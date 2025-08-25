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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
import sys

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global_down = orig_w / 1600
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    print(f"scale {float(global_down) * float(resolution_scale)}")
                    WARNED = True
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    sys.stdout.write('\r')
    sys.stdout.write("load camera {}".format(id))
    sys.stdout.flush()

    #读取法线数据-------------
    if args.w_normal_prior:
        import torch
        import os
        from utils.general_utils import PILtoTorch
        from PIL import Image
        import torch.nn.functional as F
        #normal_path = cam_info.image_path.replace('images', args.w_normal_prior)
        normal_path = os.path.join(os.path.dirname(os.path.dirname(cam_info.image_path)), args.w_normal_prior, os.path.basename(cam_info.image_path).split('.')[0])
        
        if os.path.exists(normal_path+ '.npy'):
            _normal = torch.tensor(np.load(normal_path+ '.npy'))
            _normal = - (_normal * 2 - 1)
            resized_normal = F.interpolate(_normal.unsqueeze(0), size=resolution[::-1], mode='bicubic')
            _normal = resized_normal.squeeze(0)
            # normalize normal
            _normal = _normal.permute(1, 2, 0) @ (torch.tensor(np.linalg.inv(cam_info.R)).float())
            _normal = _normal.permute(2, 0, 1)
        elif os.path.exists(normal_path+ '.png'):
            _normal = Image.open(normal_path+ '.png')
            resized_normal = PILtoTorch(_normal, resolution)
            resized_normal = resized_normal[:3]
            _normal = - (resized_normal * 2 - 1)
            # normalize normal
            _normal = _normal.permute(1, 2, 0) @ (torch.tensor(np.linalg.inv(cam_info.R)).float())
            _normal = _normal.permute(2, 0, 1)
        else:
            print(f"Cannot find normal {normal_path}.png")
            _normal = None
    else:
        _normal = None
    #-----------------------

    #读取深度数据--------------------------
    if args.w_depth_prior:
        import cv2        
        depth_path = os.path.join(os.path.dirname(os.path.dirname(cam_info.image_path)), args.w_depth_prior, os.path.basename(cam_info.image_path).split('.')[0])
        _depth = np.load(depth_path + '.npy', allow_pickle=True)
        if len(_depth.shape) == 3:
            _depth = _depth[..., 0]
        _depth = (_depth - _depth.min()) / (_depth.max() - _depth.min())
        w, h = resolution
        _depth = cv2.resize(_depth, (w, h), interpolation=cv2.INTER_NEAREST)
        _depth = torch.from_numpy(_depth).float()

        #读取深度图与法线图对齐的置信度mask
        # depth_confidence_path = os.path.join(os.path.dirname(os.path.dirname(cam_info.image_path)), "depth_normals_mask", os.path.basename(cam_info.image_path).split('.')[0])
        # _depth_confidence = 1 - (cv2.imread(depth_confidence_path + '.jpg') / 255)[..., 0]
        # _depth_confidence = cv2.resize(_depth_confidence, (w, h), interpolation=cv2.INTER_NEAREST)
        # _depth_confidence = torch.from_numpy(_depth_confidence).unsqueeze(0)
        # _depth_confidence = _depth_confidence.float()
    else:
        _depth = None


    # return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
    #               FoVx=cam_info.FovX, FoVy=cam_info.FovY,
    #               image_width=resolution[0], image_height=resolution[1],
    #               image_path=cam_info.image_path,
    #               image_name=cam_info.image_name, uid=cam_info.global_id, 
    #               preload_img=args.preload_img, 
    #               ncc_scale=args.ncc_scale,
    #               data_device=args.data_device)
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image_width=resolution[0], image_height=resolution[1],
                  image_path=cam_info.image_path,
                  image_name=cam_info.image_name, uid=cam_info.global_id, 
                  preload_img=args.preload_img, 
                  ncc_scale=args.ncc_scale,
                  data_device=args.data_device,
                  normal=_normal,
                  depth=_depth)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
