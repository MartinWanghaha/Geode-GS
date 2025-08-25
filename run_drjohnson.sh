#!/bin/bash


# ====== 训练参数 ======
GS_EPOCHS=30000
NO_DENSIFY=True

# ====== 数据集路径 ======
SOURCE_PATH="/home/martin/code/dp-gs/data/tandt_db/db/drjohnson"
MODEL_PATH="/home/martin/code/dp-gs/outputs/tandt_db/db/drjohnson"
mkdir -p ${MODEL_PATH}
touch ${MODEL_PATH}/output.txt

# ====== 复制稀疏文件 ======
# 检查 sparse/0 目录是否存在
if [ -d "${SOURCE_PATH}/sparse/0" ]; then
        cp -r "${SOURCE_PATH}/sparse/0"/* "${SOURCE_PATH}/sparse/"
        echo "复制完成."
else
    echo "错误：${SOURCE_PATH}/sparse/0 目录不存在，无法复制稀疏文件."
fi

# ====== 初始化参数 ======
MATCHES_PER_REF=30000
NNS_PER_REF=5
NUM_REFS=180
RESOLUTION=-1
ROMA_MODEL=indoors
# ====== 启动训练 ======
python train.py \
  train.gs_epochs=$GS_EPOCHS \
  train.no_densify=$NO_DENSIFY \
  gs.dataset.source_path=$SOURCE_PATH \
  gs.dataset.model_path=$MODEL_PATH \
  gs.dataset.resolution=$RESOLUTION \
  init_wC.matches_per_ref=$MATCHES_PER_REF \
  init_wC.nns_per_ref=$NNS_PER_REF \
  init_wC.roma_model=$ROMA_MODEL \
  init_wC.num_refs=$NUM_REFS | tee "${MODEL_PATH}/output.txt"

