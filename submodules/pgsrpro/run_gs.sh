#!/bin/bash

set -e

#修改-------
scene="/home/martin/code/dp-gs/data" #结尾不带"/"

output_3dgs="./output/testt" #结尾不带"/"

output_mesh="./output/testt/mesh"

logName="vip"

dataDevice="cpu"

normals="normals"

mkdir -p "${output_mesh}"

maxDepth=20.0

voxelSize=0.01

#仅渲染
time python train.py -s "${scene}" \
                     -m ${output_3dgs} \
                     --w_normal_prior normals\
                     --resolution 1\
                     --wo_image_weight\
                     --opacity_cull_threshold 0.05 \
                     --multi_view_weight_from_iter 7000 \
                     --single_view_weight_from_iter 7000 \
                     --max_abs_split_points 0\
                     --densify_until_iter 22000\
                     --iterations 30000 \
                     --exposure_compensation \
                     --data_device "${dataDevice}" > "${output_3dgs}/${logName}.txt"


# time python render.py -m ${output_3dgs} \
#                       --max_depth ${maxDepth} \
#                       --voxel_size ${voxelSize} \
#                       --iteration 30000 \
#                       --num_cluster 10 \
#                       --use_depth_filter
