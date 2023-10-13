echo "dns-ratio is ${1} on GPU ${2}"
python ../src/train.py \
 --gpus ${2} --dns-ratio ${1} \
 --data-name cifar100 --data-root ~/datasets/cifar100 --autoaugment 1 \
 --model-name vgg64_7_plane_mtl --model-desc mtl-test-v1 --num-tasks 6 \
 --resume 0 --latest-checkpoint ../models/resnet18-dns-test-v1-dns_ratio-0.1-on-cifar100/save_models/latest-training-state-dict.pth.tar \
 --epochs 100 --batch-size 500 --num-workers 8 \
 --lr 0.1 --lr-decay-type multistep --lr-decay-steps 70-85-90 --lr-decay-rate 0.1 \
 --weight-decay 1e-4 \
 --useKD 0 -T 3.0 --gamma 0.3 \
 --useFD 0 --FD-loss-coefficient 0.03 \
 --model-saving-root ../models --log-root ./logs \
 --print-freq 10 --verbose 1
