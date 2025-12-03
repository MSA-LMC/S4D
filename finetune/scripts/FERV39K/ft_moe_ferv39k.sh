# pip install tensorboardX scikit-learn  einops  timm==0.6.12 opencv-python pandas matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

server=4090
pretrain_dataset='voxcelebv2+affectnet'
finetune_dataset='ferv39k+affectnet-7'
num_labels=7
ckpts=(checkpoints/pretrain/voxceleb2+AffectNet/vit_base_voxceleb2+affectnet_100.pt)
input_size=160
sr=1
model=$1
model_dir="${model}_server${pretrain_server}"

lr=1e-5
epochs=100
BATCH_SIZE=16

# mtl setting
sfer_data_set='affectnet-7'
sfer_data_rate=$2
dfer_data_rate=$3
moe_type=$5
num_experts=$6
top_k=$7
moe_layers=$8
device=$4

# 如果moe_layers为空，则默认为6-11
if [ -z "$moe_layers" ]; then
    moe_layers=6
fi

# 如果top_k为空，则默认为2
if [ -z "$top_k" ]; then
    top_k=2
fi

for ckpt in "${ckpts[@]}";
do  
    tag=${sfer_data_set}_${sfer_data_rate}_${dfer_data_rate}_${moe_type}_et${num_experts}_k${top_k}_moe_layer_id${moe_layers}-11

    OUTPUT_DIR="./saved/model/finetune/MTL/${finetune_dataset}/${pretrain_dataset}_${model_dir}/checkpoint-${ckpt}/eval_lr_${lr}_epoch_${epochs}_size${input_size}_sr${sr}#tag${tag}"
    if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p $OUTPUT_DIR
    fi
    
    DATA_PATH="data/FERV39K/2_ClipsforFaceCrop"
    TRAIN_LABEL="$(pwd)/data/FERV39K/Annotation/train.csv"
    TEST_LABEL="$(pwd)/data/FERV39K/Annotation/test.csv"
    MODEL_PATH="${ckpt}" 
    echo ${MODEL_PATH}
    echo $OUTPUT_DIR
    # batch_size can be adjusted according to number of GPUs
    CUDA_VISIBLE_DEVICES=$device python  \
        s4d/run_class_finetuning.py \
        --model ${model} \
        --data_set FERV39k \
        --nb_classes ${num_labels} \
        --data_path ${DATA_PATH} \
        --train_label_path ${TRAIN_LABEL} \
        --test_label_path ${TEST_LABEL} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size ${BATCH_SIZE} \
        --num_sample 1 \
        --input_size ${input_size} \
        --short_side_size ${input_size} \
        --save_ckpt_freq 1000 \
        --num_frames 16 \
        --sampling_rate ${sr} \
        --opt adamw \
        --lr ${lr} \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs ${epochs} \
        --dist_eval \
        --test_num_segment 2 \
        --test_num_crop 2 \
        --num_workers 8 \
        --sfer_data_set affectnet-7 \
        --sfer_data_rate ${sfer_data_rate} \
        --dfer_data_rate ${dfer_data_rate} \
        --moe_type ${moe_type} \
        --num_experts ${num_experts} \
        --top_k ${top_k} \
        --moe_layers ${moe_layers} \
       >>${OUTPUT_DIR}/nohup.out 2>&1
done
echo 'done!'
