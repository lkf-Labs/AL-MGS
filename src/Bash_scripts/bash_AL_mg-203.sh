#!/bin/bash

echo $#
DATA_DIR="$1"
OUTPUT_DIR="$2"
SAMPLE_SET_INDEX="$3"
TRAIN_CONFIG="$4"
SAMPLING_CONFIG="$5"
SEED="$6"

echo $DATA_DIR  # 数据集根目录
echo $OUTPUT_DIR  # 代码运行过程保存文件的路径
echo $EXPERIMENT_NAME
echo $SAMPLE_SET_INDEX # 初始数据集索引（1-5）
echo $TRAIN_CONFIG  # 训练配置文件 train_config.yaml (若使用learning loss，则对应配置文件为train_learningloss_config.yaml）
echo $SAMPLING_CONFIG  # 样本选择策略配置文件
echo $SEED  # 初始化随机种子 （42）


echo "Starting task"

INIT_INDICES=$(sed "${SAMPLE_SET_INDEX}q;d" src/Configs/init_indices/prostate_indices)
echo "Init indices used: $INIT_INDICES"

python src/main.py --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --data_config data_config/data_config_mg-203.yaml --train__train_indices $INIT_INDICES --model__out_channels 1 --train__loss__normalize_fct sigmoid --train__loss__n_classes 1 --train_config $TRAIN_CONFIG --sampling_config $SAMPLING_CONFIG --seed $SEED --gpu_idx 1

echo "Reached end of job file."