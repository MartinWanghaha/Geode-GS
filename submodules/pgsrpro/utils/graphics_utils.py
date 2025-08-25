#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple
import torch.nn.functional as F
import kornia.filters as KF


def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz

def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
    valid_x, valid_y = torch.meshgrid(valid_x, valid_y, indexing='xy')
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz

def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1,3)
    # xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], axis=-1) @ torch.inverse(extrinsic_matrix).transpose(0,1)
    # xyz_world = xyz_world[...,:3]

    return xyz_cam

def depth_pcd2normal(xyz, offset=None, gt_image=None):
    hd, wd, _ = xyz.shape 
    if offset is not None:
        ix, iy = torch.meshgrid(
            torch.arange(wd), torch.arange(hd), indexing='xy')
        xy = (torch.stack((ix, iy), dim=-1)[1:-1,1:-1]).to(xyz.device)
        p_offset = torch.tensor([[0,1],[0,-1],[1,0],[-1,0]]).float().to(xyz.device)
        new_offset = p_offset[None,None] + offset.reshape(hd, wd, 4, 2)[1:-1,1:-1]
        xys = xy[:,:,None] + new_offset
        xys[..., 0] = 2 * xys[..., 0] / (wd - 1) - 1.0
        xys[..., 1] = 2 * xys[..., 1] / (hd - 1) - 1.0
        sampled_xyzs = torch.nn.functional.grid_sample(xyz.permute(2,0,1)[None], xys.reshape(1, -1, 1, 2))
        sampled_xyzs = sampled_xyzs.permute(0,2,3,1).reshape(hd-2,wd-2,4,3)
        bottom_point = sampled_xyzs[:,:,0]
        top_point = sampled_xyzs[:,:,1]
        right_point = sampled_xyzs[:,:,2]
        left_point = sampled_xyzs[:,:,3]
    else:
        bottom_point = xyz[..., 2:hd,   1:wd-1, :]
        top_point    = xyz[..., 0:hd-2, 1:wd-1, :]
        right_point  = xyz[..., 1:hd-1, 2:wd,   :]
        left_point   = xyz[..., 1:hd-1, 0:wd-2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point 
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(xyz_normal.permute(2,0,1), (1,1,1,1), mode='constant').permute(1,2,0)
    return xyz_normal

def normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix, offset=None, gt_image=None):
    # depth: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    # xyz_normal: (H, W, 3)
    xyz_world = depth2point_world(depth, intrinsic_matrix, extrinsic_matrix) # (HxW, 3)        
    xyz_world = xyz_world.reshape(*depth.shape, 3)
    xyz_normal = depth_pcd2normal(xyz_world, offset, gt_image)

    return xyz_normal

def normal_from_neareast(normal, offset):
    _, hd, wd = normal.shape 
    left_top_point = normal[..., 0:hd-2, 0:wd-2]
    top_point      = normal[..., 0:hd-2, 1:wd-1]
    right_top_point= normal[..., 0:hd-2, 2:wd]
    left_point   = normal[..., 1:hd-1, 0:wd-2]
    right_point  = normal[..., 1:hd-1, 2:wd]
    left_bottom_point   = normal[..., 2:hd, 0:wd-2]
    bottom_point = normal[..., 2:hd,   1:wd-1]
    right_bottom_point   = normal[..., 2:hd, 2:wd]
    normals = torch.stack((left_top_point,top_point,right_top_point,left_point,right_point,left_bottom_point,bottom_point,right_bottom_point),dim=0)
    new_normal = (normals * offset[:,None,1:-1,1:-1]).sum(0)
    new_normal = torch.nn.functional.normalize(new_normal, p=2, dim=0)
    new_normal = torch.nn.functional.pad(new_normal, (1,1,1,1), mode='constant').permute(1,2,0)
    return new_normal

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrixCenterShift(znear, zfar, cx, cy, fl_x, fl_y, w, h):
    top = cy / fl_y * znear
    bottom = -(h - cy) / fl_y * znear

    left = -(w - cx) / fl_x * znear
    right = cx / fl_x * znear

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def patch_offsets(h_patch_size, device):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
    return torch.stack(torch.meshgrid(offsets, offsets, indexing='xy')[::-1], dim=-1).view(1, -1, 2)

def patch_warp(H, uv):
    B, P = uv.shape[:2]
    H = H.view(B, 3, 3)
    ones = torch.ones((B,P,1), device=uv.device)
    homo_uv = torch.cat((uv, ones), dim=-1)

    grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
    grid_tmp = grid_tmp.reshape(B, P, 3)
    grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
    return grid

def pixelwise_cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    计算两个 3xHxW PyTorch 张量之间的逐像素余弦相似度，
    并将结果归一化到 [0, 1] 范围。
    每个像素被视为一个3维向量（对应其R, G, B或其他三个通道的值）。
    Args:
        tensor1 (torch.Tensor): 第一个输入张量，形状为 (3, H, W)。
        tensor2 (torch.Tensor): 第二个输入张量，形状为 (3, H, W)。
    Returns:
        torch.Tensor: 逐像素余弦相似度张量，形状为 (H, W)。
                      相似度值在 [0, 1] 之间。

    Raises:
        ValueError: 如果输入张量的形状不是 3xHxW 或它们的 H 和 W 维度不匹配。
    """

    # 检查输入张量的形状是否为 3xHxW
    if tensor1.dim() != 3 or tensor1.shape[0] != 3 or \
       tensor2.dim() != 3 or tensor2.shape[0] != 3:
        raise ValueError("输入张量必须是 3xHxW 的形状。")

    # 检查两个张量的 H 和 W 维度是否匹配
    # 只需要比较 H 和 W，因为通道数已经确定是3
    if tensor1.shape[1:] != tensor2.shape[1:]: # 比较 (H, W) 部分
        raise ValueError("两个输入张量必须具有相同的 H 和 W 维度。")
    # 将张量重塑为 (H*W, 3)，以便每个“像素”成为一个3维向量
    # .permute(1, 2, 0) 将 (C, H, W) 变为 (H, W, C)
    # .reshape(-1, 3) 将 (H, W, C) 变为 (H*W, C)
    tensor1_reshaped = tensor1.permute(1, 2, 0).reshape(-1, 3)
    tensor2_reshaped = tensor2.permute(1, 2, 0).reshape(-1, 3)

    # 计算逐像素点积（现在是每个 3维向量的点积）
    # (H*W, 3) @ (3, H*W) 或 (H*W, 3) 和 (H*W, 3) 的逐行点积
    # torch.sum(dim=1) 对每个 (H*W) 向量的 3 个分量求和
    dot_product = torch.sum(tensor1_reshaped * tensor2_reshaped, dim=1)

    # 计算逐像素的 L2 范数（每个 3维向量的范数）
    norm_tensor1 = torch.norm(tensor1_reshaped, p=2, dim=1)
    norm_tensor2 = torch.norm(tensor2_reshaped, p=2, dim=1)

    # 避免除以零：将分母中接近于零的值替换为一个小 epsilon 值
    epsilon = 1e-8
    denominator = norm_tensor1 * norm_tensor2
    denominator = torch.where(denominator == 0, torch.full_like(denominator, epsilon), denominator)

    # 计算逐像素余弦相似度（原始范围 [-1, 1]）
    cosine_similarity_flat = dot_product / denominator

    # 将结果裁剪到 [-1, 1] 范围内，以应对浮点误差
    cosine_similarity_flat = torch.clamp(cosine_similarity_flat, -1.0, 1.0)

    # --- 归一化到 [0, 1] 范围 ---
    # 公式: (值 + 1) / 2
    normalized_cosine_similarity_flat = (cosine_similarity_flat + 1.0) / 2.0

    # 将结果重塑回 (H, W) 的形状，因为每个像素的相似度是一个标量值
    H, W = tensor1.shape[1], tensor1.shape[2]
    normalized_cosine_similarity_matrix = normalized_cosine_similarity_flat.reshape(H, W)

    return normalized_cosine_similarity_matrix



def guided_normal_fusion(
    depth_normal: torch.Tensor,
    prior_normal: torch.Tensor,
    kernel_size: int = 11, # kernel_size for guided_blur typically an odd integer, e.g., 5x5 -> 5
    eps: float = 1e-6,
    border_type: str = 'reflect' # new parameter for guided_blur
) -> torch.Tensor:
    """
    使用 kornia.filters.guided_blur 融合 depth_normal 和 prior_normal。

    Args:
        depth_normal (torch.Tensor): 从深度图计算得到的法线，形状为 (3, H, W)，值域应为 [-1, 1]。
                                      通常表示为 (N, C, H, W) 或 (C, H, W)。
        prior_normal (torch.Tensor): 深度学习模型预测的法线，形状为 (3, H, W)，值域应为 [-1, 1]。
                                      通常表示为 (N, C, H, W) 或 (C, H, W)。
        kernel_size (int): 引导模糊核的大小 (例如，5 表示 5x5 核)。必须是奇数。
                          越大越平滑，但可能丢失细节。
        eps (float): 引导模糊的正则化参数，用于防止过拟合，避免除以零。
        border_type (str): 图像边界填充模式。可以是 'constant', 'reflect', 'replicate', 'circular'。

    Returns:
        torch.Tensor: 融合后的法线，形状为 (3, H, W)，值域为 [-1, 1]。
    """
    # 确保输入张量在正确的设备上 (CPU 或 GPU)
    device = depth_normal.device
    
    # 确保输入是浮点型
    if not depth_normal.is_floating_point():
        depth_normal = depth_normal.float()
    if not prior_normal.is_floating_point():
        prior_normal = prior_normal.float()

    # Kornia 的引导模糊期望 (N, C, H, W) 格式，如果输入是 (C, H, W) 则添加批次维度
    original_dim = depth_normal.dim()
    if original_dim == 3:
        depth_normal = depth_normal.unsqueeze(0)  # 添加批次维度
        prior_normal = prior_normal.unsqueeze(0)  # 添加批次维度

    # 检查张量形状是否匹配
    if depth_normal.shape != prior_normal.shape:
        raise ValueError(
            f"Input tensors must have the same shape. Got depth_normal: {depth_normal.shape} "
            f"and prior_normal: {prior_normal.shape}"
        )
    
    # 确保通道数为3 (RGB或XYZ)
    if depth_normal.shape[1] != 3:
        raise ValueError(f"Input tensors must have 3 channels (for X, Y, Z normal components). "
                         f"Got {depth_normal.shape[1]} channels.")
    
    # 确保 kernel_size 是奇数
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be an odd integer. Got {kernel_size}.")

    # 归一化法线向量，确保它们是单位向量
    depth_normal_norm = F.normalize(depth_normal, p=2, dim=1)
    prior_normal_norm = F.normalize(prior_normal, p=2, dim=1)

    # 应用引导模糊
    # input: depth_normal_norm (要被模糊的图像)
    # guidance: prior_normal_norm (提供结构信息的引导图像)
    # kernel_size: 整数，表示方形核的边长，例如 5 -> 5x5 核
    # eps: 正则化参数
    fused_normal = KF.guided_blur(
        input=depth_normal_norm,
        guidance=prior_normal_norm,
        kernel_size=(kernel_size, kernel_size), # kornia expects a tuple for kernel_size
        eps=eps,
        border_type=border_type
    )

    # 再次归一化融合后的法线，因为滤波过程可能会使其不再是单位向量
    fused_normal = F.normalize(fused_normal, p=2, dim=1)

    # 如果输入是 (C, H, W)，则移除批次维度
    if original_dim == 3:
        fused_normal = fused_normal.squeeze(0)

    return fused_normal