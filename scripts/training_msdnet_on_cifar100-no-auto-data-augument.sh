echo "dns-ratio is ${1} on GPU ${2}"
python ../src/train.py \
 --gpus ${2} --dns-ratio ${1} \
 --data-name cifar100 --data-root ~/datasets/cifar100 --autoaugment 0 \
 --model-name msdnet_cifar100 --model-desc dns-test-v1-no-data-augument --num-tasks 7 \
 --resume 0 --latest-checkpoint ../models/resnet18-dns-test-v1-dns_ratio-0.1-on-cifar100/save_models/latest-training-state-dict.pth.tar \
 --epochs 300 --batch-size 500 \
 --lr 0.1 --lr-decay-type multistep --lr-decay-steps 250-280-295 --lr-decay-rate 0.1 \
 --weight-decay 5e-4 \
 --useKD 0 -T 3.0 --gamma 0.9 \
 --useFD 0 --FD-loss-coefficient 0.03 \
 --model-saving-root ../models --log-root ./logs \
 --print-freq 50 --verbose 0
