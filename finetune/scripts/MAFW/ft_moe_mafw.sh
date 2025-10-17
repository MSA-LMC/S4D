server=4090
pretrain_dataset='voxcelebv2+affectnet'
finetune_dataset='mafw+affectnet-7'
num_labels=11
ckpts=(checkpoints/pretrain/voxceleb2+AffectNet/vit_base_voxceleb2+affectnet_100.pt)
input_size=160
sr=4
model=$1
model_dir="${model}_server${pretrain_server}"

splits=(1 2 3 4 5) # you can change it to other folds, e.g., (2,3,4,5)
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
lora_rank=$8
lora_alpha=$9

device=$4

for ckpt in "${ckpts[@]}";
do
  for split in "${splits[@]}";
  do
    tag=${sfer_data_set}_${sfer_data_rate}_${dfer_data_rate}_${moe_type}_et${num_experts}_k${top_k}_lr${lora_rank}_la${lora_alpha}_abs_soft1.0

    OUTPUT_DIR="./saved/model/finetune/MTL/${finetune_dataset}/${pretrain_dataset}_${model_dir}/${ckpt}/eval_split0${split}_lr_${lr}_epoch_${epochs}_size${input_size}_sr${sr}#tag${tag}"
    if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p $OUTPUT_DIR
    fi

    DATA_PATH="data/MAFW/frames_crop_align_224"
    TRAIN_LABEL="data/MAFW/anno/single/no_caption/set_${split}/train.csv"
    TEST_LABEL="data/MAFW/anno/single/no_caption/set_${split}/test.csv"
    # path to pre-trained model
    MODEL_PATH="${ckpt}"        # path to pre-trained model
    echo $OUTPUT_DIR
    # batch_size can be adjusted according to number of GPUs
    CUDA_VISIBLE_DEVICES=$device python \
        finetune/run_class_finetuning.py \
        --model ${model} \
        --data_set MAFW \
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
        --lora_rank ${lora_rank} \
        --lora_alpha ${lora_alpha} \
        >>${OUTPUT_DIR}/nohup.out 2>&1
    done
done
echo "Done!"

