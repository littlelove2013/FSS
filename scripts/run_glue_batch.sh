# echo "start" \
# && bash run_glue.sh cola 0 0.5 3 && bash run_glue.sh cola 1 0.1 3 \
# && bash run_glue.sh cola 1 0.3 3 && bash run_glue.sh cola 1 0.5 3 \
# && bash run_glue.sh cola 1 0.7 3 && bash run_glue.sh cola 1 0.9 3 \
# && bash run_glue.sh mrpc 0 0.5 5 && bash run_glue.sh mrpc 1 0.1 5 \
# && bash run_glue.sh mrpc 1 0.3 5 && bash run_glue.sh mrpc 1 0.5 5 \
# && bash run_glue.sh mrpc 1 0.7 5 && bash run_glue.sh mrpc 1 0.9 5 \
# && bash run_glue.sh stsb 0 0.5 3 && bash run_glue.sh stsb 1 0.1 3 \
# && bash run_glue.sh stsb 1 0.3 3 && bash run_glue.sh stsb 1 0.5 3 \
# && bash run_glue.sh stsb 1 0.7 3 && bash run_glue.sh stsb 1 0.9 3 \
# && bash run_glue.sh sst2 0 0.5 3 && bash run_glue.sh sst2 1 0.1 3 \
# && bash run_glue.sh sst2 1 0.3 3 && bash run_glue.sh sst2 1 0.5 3 \
# && bash run_glue.sh sst2 1 0.7 3 && bash run_glue.sh sst2 1 0.9 3 \
# && bash run_glue.sh wnli 0 0.5 5 && bash run_glue.sh wnli 1 0.1 5 \
# && bash run_glue.sh wnli 1 0.3 5 && bash run_glue.sh wnli 1 0.5 5 \
# && bash run_glue.sh wnli 1 0.7 5 && bash run_glue.sh wnli 1 0.9 5 \
# && bash run_glue.sh qnli 0 0.5 3 && bash run_glue.sh qnli 1 0.1 3 \
# && bash run_glue.sh qnli 1 0.3 3 && bash run_glue.sh qnli 1 0.5 3 \
# && bash run_glue.sh qnli 1 0.7 3 && bash run_glue.sh qnli 1 0.9 3 \
# && bash run_glue.sh rte 0 0.5 3 && bash run_glue.sh rte 1 0.1 3 \
# && bash run_glue.sh rte 1 0.3 3 && bash run_glue.sh rte 1 0.5 3 \
# && bash run_glue.sh rte 1 0.7 3 && bash run_glue.sh rte 1 0.9 3 \
# && bash run_glue.sh qqp 0 0.5 3 && bash run_glue.sh qqp 1 0.1 3 \
echo "start" \
&& bash run_glue.sh qqp 1 0.3 3 && bash run_glue.sh qqp 1 0.5 3 \
&& bash run_glue.sh qqp 1 0.7 3 && bash run_glue.sh qqp 1 0.9 3 \
&& bash run_glue.sh mnli 0 0.5 3 && bash run_glue.sh mnli 1 0.1 3 \
&& bash run_glue.sh mnli 1 0.3 3 && bash run_glue.sh mnli 1 0.5 3 \
&& bash run_glue.sh mnli 1 0.7 3 && bash run_glue.sh mnli 1 0.9 3 \
&& echo "Done" &

# TASK_NAME=${1}
# USE_DNS=${3}
# DNS_RATIO=${2}
# NUM_TRAINING_EPOCHS=${4}

# # 设置为fp16，则batchsize和lr都要加倍才行
# python run_glue_no_trainer.py \
#   --model_name_or_path bert-base-cased \
#   --task_name $TASK_NAME \
#   --max_length 128 \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 5 \
#   --output_dir ./tmp/$TASK_NAME/ \
#   --mixed-precision fp16 \
#   --use-dns 1 \
#   --dns-ratio 0.5 \
#   --gpus 1
