import os
import torch
import random
from random import randint
import numpy as np
import cv2

from tqdm.rich import trange
from tqdm import tqdm as tqdm
from source.networks import Warper3DGS
import wandb

import sys
from utils.image_utils import erode
sys.path.append('./submodules/pgsr-pro/')
import lpips
from source.losses import ssim, l1_loss, psnr
from scene.cameras import Camera
from utils.loss_utils import lncc, get_img_grad_weight
from utils.graphics_utils import patch_offsets, patch_warp, pixelwise_cosine_similarity, guided_normal_fusion
from rich.console import Console
from rich.theme import Theme
import torch.nn.functional as F

custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red"
})

from source.corr_init import init_gaussians_with_corr, init_gaussians_with_corr_fast
from source.utils_aux import log_samples

from source.timer import Timer

torch.autograd.set_detect_anomaly(True)

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        preload_img=False, data_device = "cuda")
    return virtul_cam


class DPGSTrainer:
    def __init__(self,
                 GS: Warper3DGS,
                 training_config,
                 dataset_config,
                 dataset_white_background=False,
                 device=torch.device('cuda'),
                 log_wandb=True,
                 ):
        self.GS = GS
        self.scene = GS.scene
        self.viewpoint_stack = GS.viewpoint_stack
        self.render = GS.render
        self.gaussians = GS.gaussians
        self.pipe = GS.pipe
        self.bg =  GS.bg

        self.training_config = training_config
        self.dataset_config = dataset_config
        self.GS_optimizer = GS.gaussians.optimizer
        self.dataset_white_background = dataset_white_background

        self.training_step = 1
        self.gs_step = 0
        self.CONSOLE = Console(width=120, theme=custom_theme)
        self.saving_iterations = training_config.save_iterations
        self.evaluate_iterations = None
        self.batch_size = training_config.batch_size
        self.ema_loss_for_log = 0.0

        # Logs in the format {step:{"loss1":loss1_value, "loss2":loss2_value}}
        self.logs_losses = {}
        self.lpips = lpips.LPIPS(net='vgg').to(device)
        self.device = device
        self.timer = Timer()
        self.log_wandb = log_wandb
        self.debug_path = os.path.join(self.scene.model_path, "debug")
        self.single_path = os.path.join(self.debug_path, "single_view")
        os.makedirs(self.single_path, exist_ok=True)

    def load_checkpoints(self, load_cfg):
        # Load 3DGS checkpoint
        if load_cfg.gs:
            self.gs.gaussians.restore(
                torch.load(f"{load_cfg.gs}/chkpnt{load_cfg.gs_step}.pth")[0],
                self.training_config)
            self.GS_optimizer = self.GS.gaussians.optimizer
            self.CONSOLE.print(f"3DGS loaded from checkpoint for iteration {load_cfg.gs_step}",
                               style="info")
            self.training_step += load_cfg.gs_step
            self.gs_step += load_cfg.gs_step

    def train(self, train_cfg):
        # 3DGS training

        self.CONSOLE.print("Train 3DGS for {} iterations".format(train_cfg.gs_epochs), style="info")    
        with trange(self.training_step, self.training_step + train_cfg.gs_epochs, desc="[green]Train gaussians") as progress_bar:
            for self.training_step in progress_bar:
                radii = self.train_step_gs(max_lr=train_cfg.max_lr, no_densify=train_cfg.no_densify)
                with torch.no_grad():
                    if train_cfg.no_densify:
                        self.prune(radii)
                    else:
                        self.densify_and_prune(radii)
                    if train_cfg.reduce_opacity:
                        # Slightly reduce opacity every few steps:
                        if self.gs_step < self.training_config.densify_until_iter and self.gs_step % 10 == 0:
                            opacities_new = torch.log(torch.exp(self.GS.gaussians._opacity.data) * 0.99)
                            self.GS.gaussians._opacity.data = opacities_new
                    self.timer.pause()
                    # Progress bar
                    if self.training_step % 10 == 0:
                        progress_bar.set_postfix({"[red]Loss": f"{self.ema_loss_for_log:.{7}f}"}, refresh=True)
                    # Log and save
                    if self.training_step in self.saving_iterations:
                        self.save_model()
                    if self.evaluate_iterations is not None:
                        if self.training_step in self.evaluate_iterations:
                            self.evaluate()
                    else:
                        if (self.training_step <= 3000 and self.training_step % 500 == 0) or \
                            (self.training_step > 3000 and self.training_step % 1000 == 228) :
                            self.evaluate()

                    self.timer.start()
                    
    def evaluate(self):
        torch.cuda.empty_cache()
        log_gen_images, log_real_images = [], []
        validation_configs = ({'name': 'test', 'cameras': self.scene.getTestCameras(), 'cam_idx': self.training_config.TEST_CAM_IDX_TO_LOG},
                              {'name': 'train',
                               'cameras': [self.scene.getTrainCameras()[idx % len(self.scene.getTrainCameras())] for idx in
                                           range(0, 150, 5)], 'cam_idx': 10})
        if self.log_wandb:
            wandb.log({f"Number of Gaussians": len(self.GS.gaussians._xyz)}, step=self.training_step)
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_splat_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(self.GS(viewpoint)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(self.device), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).double()
                    psnr_test += psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).double()
                    ssim_test += ssim(image, gt_image).double()
                    lpips_splat_test += self.lpips(image, gt_image).detach().double()
                    if idx in [config['cam_idx']]:
                        log_gen_images.append(image)
                        log_real_images.append(gt_image)
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_splat_test /= len(config['cameras'])
                if self.log_wandb:
                    wandb.log({f"{config['name']}/L1": l1_test.item(), f"{config['name']}/PSNR": psnr_test.item(), \
                            f"{config['name']}/SSIM": ssim_test.item(), f"{config['name']}/LPIPS_splat": lpips_splat_test.item()}, step = self.training_step)
                self.CONSOLE.print("\n[ITER {}], #{} gaussians, Evaluating {}: L1={:.6f},  PSNR={:.6f}, SSIM={:.6f}, LPIPS_splat={:.6f} ".format(
                    self.training_step, len(self.GS.gaussians._xyz), config['name'], l1_test.item(), psnr_test.item(), ssim_test.item(), lpips_splat_test.item()), style="info")
        if self.log_wandb:
            with torch.no_grad():
                log_samples(torch.stack((log_real_images[0],log_gen_images[0])) , [], self.training_step, caption="Real and Generated Samples")
                wandb.log({"time": self.timer.get_elapsed_time()}, step=self.training_step)
        torch.cuda.empty_cache()

    def train_step_gs(self, max_lr = False, no_densify = False):
        self.gs_step += 1
        if max_lr:
            self.GS.gaussians.update_learning_rate(max(self.gs_step, 8_000))
        else:
            self.GS.gaussians.update_learning_rate(self.gs_step)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.gs_step % 1000 == 0:
            self.GS.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
      
        render_pkg = self.GS(viewpoint_cam=viewpoint_cam)
        # image = render_pkg["render"]
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
        # Loss
        # gt_image = viewpoint_cam.original_image.to(self.device)
        gt_image, gt_image_gray = viewpoint_cam.get_image()
        L1_loss = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        loss = (1.0 - self.training_config.lambda_dssim) * L1_loss + \
               self.training_config.lambda_dssim * ssim_loss
        
        # # scale loss 防止高斯膨胀过大，尤其是在遮挡或稀疏监督的区域，保持高斯球的尺寸在一个合理范围
        # if visibility_filter.sum() > 0:
        #     scale = self.GS.gaussians.get_scaling[visibility_filter]
        #     sorted_scale, _ = torch.sort(scale, dim=-1) # 每个高斯球的三个轴向尺寸进行排序，拿出最小轴向尺寸
        #     min_scale_loss = sorted_scale[...,0]
        #     loss += self.training_config.scale_loss_weight * min_scale_loss.mean()
        
        # single-view loss 局部平面性假设，单视角一致性loss
        if self.training_step > self.training_config.single_view_weight_from_iter:
            weight = self.training_config.single_view_weight
            # render_alpha = render_pkg["rendered_alpha"]
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]
            image_weight = (1.0 - get_img_grad_weight(gt_image)) # 图像梯度控制边缘贡献
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            # ∇I 归一化图像梯度，表示图像中某个像素处边缘的强度（越接近边缘越大）
            # (1− ∇I)^2 对平坦区域加大权重，在边缘区域减少权重，避免边缘误差对损失干扰太大
            # ---------- normal supervision loss ----------
            if not self.dataset_config.w_normal_prior:
                if not self.training_config.wo_image_weight:
                    # image_weight = erode(image_weight[None,None]).squeeze() 网络预测的法线与从深度图反推出的法线要一致
                    normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
                else:
                    normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
                loss += (normal_loss)
            # ---------- prior loss ----------
            else:                    
                prior_normal = viewpoint_cam.normal_prior.to('cuda')
                fusion_normal = guided_normal_fusion(depth_normal, prior_normal,         
                                                    kernel_size=15, # 调整核大小，注意这里是单个整数
                                                    eps=1e-3
                                                )
                prior_normal = viewpoint_cam.normal_prior.to('cuda')
                
                if not self.training_config.wo_image_weight:
                    normal_loss = weight * ((image_weight * (fusion_normal - normal)).abs().sum(0)).mean()
                else:
                    normal_loss = weight * ((fusion_normal - normal).abs().sum(0)).mean()

                if self.training_step % 50 == 0: # 存图可视化
                    gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    if 'app_image' in render_pkg:
                        img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    else:
                        img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    normal_show = (((normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    depth_normal_show = (((depth_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    prior_normal_show = (((prior_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    fusion_normal_show = (((fusion_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                    depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                    distance = render_pkg['rendered_distance'].squeeze().detach().cpu().numpy()
                    distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
                    distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
                    distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)
                    image_weight = image_weight.detach().cpu().numpy()
                    image_weight = (image_weight * 255).clip(0, 255).astype(np.uint8)
                    image_weight_color = cv2.applyColorMap(image_weight, cv2.COLORMAP_JET)
                 
                    # --- 将所有图像保存到指定路径 ---
                    # 构建通用的基本文件名
                    base_filename = f"{self.training_step:05d}_{viewpoint_cam.image_name}"
                    cv2.imwrite(os.path.join(self.single_path, f"{base_filename}_gt_img_show.jpg"), gt_img_show)
                    cv2.imwrite(os.path.join(self.single_path, f"{base_filename}_img_show.jpg"), img_show)
                    # 保存所有法线图
                    cv2.imwrite(os.path.join(self.single_path, f"{base_filename}_normal_show.jpg"), normal_show)
                    cv2.imwrite(os.path.join(self.single_path, f"{base_filename}_depth_normal_show.jpg"), depth_normal_show)
                    cv2.imwrite(os.path.join(self.single_path, f"{base_filename}_prior_normal_show.jpg"), prior_normal_show)
                    cv2.imwrite(os.path.join(self.single_path, f"{base_filename}_fusion_normal_show.jpg"), fusion_normal_show)
                    # 保存深度图、距离图和权重图
                    cv2.imwrite(os.path.join(self.single_path, f"{base_filename}_plane_depth_color.jpg"), depth_color)
                    cv2.imwrite(os.path.join(self.single_path, f"{base_filename}_distance_color.jpg"), distance_color)
                    cv2.imwrite(os.path.join(self.single_path, f"{base_filename}_image_weight_color.jpg"), image_weight_color)
                loss += (normal_loss)

        # multi-view loss
        if self.training_step > self.training_config.multi_view_weight_from_iter:
            # 获取最近的相机ID列表
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else self.scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
                    
            use_virtul_cam = False
            if self.training_config.use_virtul_cam and (np.random.random() < self.training_config.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=self.dataset_config.multi_view_max_dis, deg_noise=self.dataset_config.multi_view_max_angle)
                use_virtul_cam = True
            if nearest_cam is not None:
                patch_size = self.training_config.multi_view_patch_size
                sample_num = self.training_config.multi_view_sample_num
                pixel_noise_th = self.training_config.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = self.training_config.multi_view_ncc_weight # 从MVS中获得灵感，多视角光度loss
                # 取像素周围7x7的像素块，将其映射到相邻帧中，并将其转换成灰度图，使用NCC（normalized cross correlation）计算两个patch之间的光度误差

                geo_weight = self.training_config.multi_view_geo_weight # 重投影误差，多视角几何loss
                # 两次重投影误差太大，则可能是有遮挡，此时就降低该权重

                ## compute geometry consistency mask and loss
                H, W = render_pkg['plane_depth'].squeeze().shape
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

                nearest_render_pkg = self.render(nearest_cam, self.gaussians, self.pipe, self.bg,
                                            return_plane=True, return_depth_normal=False)

                pts = self.gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                map_z, d_mask = self.gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)

                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                diff = pts_projections - pixels.reshape(*pts_projections.shape)

                pixel_noise = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-8)
                # pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                
                if not self.training_config.wo_use_geo_occ_aware: 
                    d_mask = d_mask & (pixel_noise < pixel_noise_th)
                    weights = (1.0 / torch.exp(pixel_noise)).detach()
                    weights[~d_mask] = 0
                else:
                    d_mask = d_mask
                    weights = torch.ones_like(pixel_noise)
                    weights[~d_mask] = 0
                if self.training_step % 200 == 0: # 存图可视化
                    # gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    # if 'app_image' in render_pkg:
                    #     img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    # else:
                    #     img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    # normal_show = (((normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    # depth_normal_show = (((depth_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    d_mask_show = (weights.float()*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
                    d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                    # depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                    # depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                    # depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                    # depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                    # distance = render_pkg['rendered_distance'].squeeze().detach().cpu().numpy()
                    # distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
                    # distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
                    # distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)
                    # image_weight = image_weight.detach().cpu().numpy()
                    # image_weight = (image_weight * 255).clip(0, 255).astype(np.uint8)
                    # image_weight_color = cv2.applyColorMap(image_weight, cv2.COLORMAP_JET)
                    row0 = np.concatenate([gt_img_show, img_show, normal_show, distance_color], axis=1)
                    row1 = np.concatenate([d_mask_show_color, depth_color, fusion_normal_show, image_weight_color], axis=1)
                    image_to_show = np.concatenate([row0, row1], axis=0)
                    cv2.imwrite(os.path.join(self.debug_path, "%05d"%self.training_step + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)
                    if d_mask.sum() > 0:
                        geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                        loss += geo_loss
                        if use_virtul_cam is False:
                            with torch.no_grad():
                                ## sample mask
                                d_mask = d_mask.reshape(-1)
                                valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                                if d_mask.sum() > sample_num:
                                    index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                                    valid_indices = valid_indices[index]

                                weights = weights.reshape(-1)[valid_indices]
                                ## sample ref frame patch
                                pixels = pixels.reshape(-1,2)[valid_indices]
                                offsets = patch_offsets(patch_size, pixels.device)
                                ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()
                                
                                H, W = gt_image_gray.squeeze().shape
                                pixels_patch = ori_pixels_patch.clone()
                                pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                                pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                                ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                                ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                                ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                                ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]
                            
                            ## compute Homography
                            ref_local_n = render_pkg["rendered_normal"].permute(1,2,0)
                            ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

                            ref_local_d = render_pkg['rendered_distance'].squeeze()
                            # rays_d = viewpoint_cam.get_rays()
                            # rendered_normal2 = render_pkg["rendered_normal"].permute(1,2,0).reshape(-1,3)
                            # ref_local_d = render_pkg['plane_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                            # ref_local_d = ref_local_d.reshape(*render_pkg['plane_depth'].shape)

                            ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                            H_ref_to_neareast = ref_to_neareast_r[None] - \
                                torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                            ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                            H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                            H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)
                            
                            ## compute neareast frame patch
                            grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                            grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                            grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                            _, nearest_image_gray = nearest_cam.get_image()
                            sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                            sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
                            ## compute loss
                            ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)

                            mask = ncc_mask.reshape(-1)
                            ncc = ncc.reshape(-1) * weights
                            ncc = ncc[mask].squeeze()

                            if mask.sum() > 0:
                                ncc_loss = ncc_weight * ncc.mean()
                                loss += ncc_loss


        self.timer.pause() 
        self.logs_losses[self.training_step] = {"loss": loss.item(),
                                                "L1_loss": L1_loss.item(),
                                                "ssim_loss": ssim_loss.item()}
        
        if self.log_wandb:
            for k, v in self.logs_losses[self.training_step].items():
                wandb.log({f"train/{k}": v}, step=self.training_step)
        self.ema_loss_for_log = 0.4 * self.logs_losses[self.training_step]["loss"] + 0.6 * self.ema_loss_for_log
        self.timer.start()
        self.GS_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            if self.gs_step < self.training_config.densify_until_iter and not no_densify:
                self.GS.gaussians.max_radii2D[render_pkg["visibility_filter"]] = torch.max(
                    self.GS.gaussians.max_radii2D[render_pkg["visibility_filter"]],
                    render_pkg["radii"][render_pkg["visibility_filter"]])
                self.GS.gaussians.add_densification_stats(render_pkg["viewspace_points"],
                                                                     render_pkg["visibility_filter"])

        # Optimizer step
        self.GS_optimizer.step()
        self.GS_optimizer.zero_grad(set_to_none=True)
        return render_pkg["radii"]

    def densify_and_prune(self, radii = None):
        # Densification or pruning
        if self.gs_step < self.training_config.densify_until_iter:
            if (self.gs_step > self.training_config.densify_from_iter) and \
                    (self.gs_step % self.training_config.densification_interval == 0):
                size_threshold = 20 if self.gs_step > self.training_config.opacity_reset_interval else None
                self.GS.gaussians.densify_and_prune(self.training_config.densify_grad_threshold,
                                                               0.005,
                                                               self.GS.scene.cameras_extent,
                                                               size_threshold, radii)
            if self.gs_step % self.training_config.opacity_reset_interval == 0 or (
                    self.dataset_white_background and self.gs_step == self.training_config.densify_from_iter):
                self.GS.gaussians.reset_opacity()             

          

    def save_model(self):
        print("\n[ITER {}] Saving Gaussians".format(self.gs_step))
        self.scene.save(self.gs_step)
        print("\n[ITER {}] Saving Checkpoint".format(self.gs_step))
        torch.save((self.GS.gaussians.capture(), self.gs_step),
                self.scene.model_path + "/chkpnt" + str(self.gs_step) + ".pth")


    def init_with_corr(self, cfg, verbose=False, roma_model=None): 
        """
        Initializes image with matchings. Also removes SfM init points.
        Args:
            cfg: configuration part named init_wC. Check train.yaml
            verbose: whether you want to print intermediate results. Useful for debug.
            roma_model: optionally you can pass here preinit RoMA model to avoid reinit 
                it every time.  
        """
        if not cfg.use:
            return None
        N_splats_at_init = len(self.GS.gaussians._xyz)
        print("N_splats_at_init:", N_splats_at_init)
        if cfg.nns_per_ref == 1:
            init_fn = init_gaussians_with_corr_fast
        else:
            init_fn = init_gaussians_with_corr
        camera_set, selected_indices, visualization_dict = init_fn(
            self.GS.gaussians, 
            self.scene, 
            cfg, 
            self.device,                                                                                    
            verbose=verbose,
            roma_model=roma_model)

        # Remove SfM points and leave only matchings inits
        if not cfg.add_SfM_init:
            with torch.no_grad():
                N_splats_after_init = len(self.GS.gaussians._xyz)
                print("N_splats_after_init:", N_splats_after_init)
                self.gaussians.tmp_radii = torch.zeros(self.gaussians._xyz.shape[0]).to(self.device)
                mask = torch.concat([torch.ones(N_splats_at_init, dtype=torch.bool),
                                    torch.zeros(N_splats_after_init-N_splats_at_init, dtype=torch.bool)],
                                axis=0)
                self.GS.gaussians.prune_points(mask)
        with torch.no_grad():
            gaussians =  self.gaussians
            gaussians._scaling =  gaussians.scaling_inverse_activation(gaussians.scaling_activation(gaussians._scaling)*0.5)
        return visualization_dict
    

    def prune(self, radii, min_opacity=0.005):
        self.GS.gaussians.tmp_radii = radii
        if self.gs_step < self.training_config.densify_until_iter:
            prune_mask = (self.GS.gaussians.get_opacity < min_opacity).squeeze()
            self.GS.gaussians.prune_points(prune_mask)
            torch.cuda.empty_cache()
        self.GS.gaussians.tmp_radii = None



