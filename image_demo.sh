#!/bin/bash

# 基于 demo/image_demo.py 的图像推理脚本
# 用法: 修改下方变量后运行 ./image_demo.sh

# ==================== 必填参数 ====================
# 输入图片路径
IMAGE="demo/demo.png"

# 配置文件路径
CONFIG="dpsn.py"

# 模型权重文件路径
CHECKPOINT="best_mIoU_iter_144000.pth"

# ==================== 可选参数 ====================
# 输出文件路径（留空则弹窗显示，不保存）
OUT_FILE="result.jpg"

# 推理设备，默认 cuda:0，可改为 cpu
DEVICE="cuda:0"

# 分割图叠加不透明度，范围 (0, 1]，默认 0.5
OPACITY=0.5

# 是否显示类别标签，默认 false。如需显示，改为 "--with-labels"
WITH_LABELS=""

# 图像标题，默认 result
TITLE="result"

# ==================== 执行推理 ====================
python demo/image_demo.py \
    "${IMAGE}" \
    "${CONFIG}" \
    "${CHECKPOINT}" \
    ${OUT_FILE:+"--out-file" "${OUT_FILE}"} \
    --device "${DEVICE}" \
    --opacity "${OPACITY}" \
    ${WITH_LABELS} \
    --title "${TITLE}"
