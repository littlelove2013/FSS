
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import shutil
import torch
import random
import datasets
from torch import nn,optim
import torch.nn.functional as F
from utils.train_utils import AverageMeter
import logging

formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# 创建一个日志记录器
logger = logging.getLogger('training')
logger.setLevel(logging.DEBUG)
# 创建一个文件处理器，并Formatter 添加到处理器
# file_handler = logging.FileHandler('experiments.log')
# file_handler.setFormatter(formatter)
# 创建一个控制台处理器，并将 Formatter 添加到处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
# 将处理器添加到记录器
# logger.addHandler(file_handler)
logger.addHandler(console_handler)
# logging.basicConfig(
#     level=logging.DEBUG,  # 设置日志级别，可以选择 DEBUG、INFO、WARNING、ERROR、CRITICAL
#     format='%(asctime)s [%(levelname)s] %(message)s',  # 设置日志格式
#     handlers=[
#         logging.StreamHandler()  # 输出到控制台
#     ]
# )

from arguments import arg_parser
from dns_msdnet import msdnet_cifar100, msdnet_imagenet
from dns_resnet import resnet18
from sdn_resnet import resnet18 as sdn_mtl_resnet18

# Logit Distillation Distribution Loss
def KDCrossEntropy(outputs, soft_targets,temperature):
    log_softmax_outputs = F.log_softmax(outputs/temperature, dim=1)
    softmax_targets = F.softmax(soft_targets/temperature, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

# Feature Distillation Distribution Loss
def FDDistributionLoss(output_feature,teacher_feature):
    return torch.dist(output_feature, teacher_feature)

class MTLLoss(object):
    def __init__(self, criterion=nn.CrossEntropyLoss(),
                useKD=False,loss_coefficient=0.9,temperature=3.0,
                useFD=False,feature_loss_coefficient=0.03):
        super(MTLLoss, self).__init__()
        self.criterion=criterion
        self.useKD=useKD
        self.loss_coefficient=loss_coefficient
        self.temperature=temperature
        self.useFD=useFD
        self.feature_loss_coefficient=feature_loss_coefficient
    def __call__(self,moddle_outputs,moddle_features,
                final_output,final_feature,targets):
        targets.detach_()
        loss = self.criterion(final_output, targets)
        for i in range(len(moddle_outputs)):
            m_o = moddle_outputs[i]
            m_f = moddle_features[i]
            if self.useKD:
                loss += (1-self.loss_coefficient)*self.criterion(m_o,targets)+ \
                    self.loss_coefficient*KDCrossEntropy(m_o,final_output.detach(),self.temperature)
            else:
                loss +=self.criterion(m_o,targets)
            if self.useFD:
                loss +=FDDistributionLoss(m_f,final_feature.detach())*self.feature_loss_coefficient
        return loss

def MTLLossFunc(moddle_outputs,moddle_features,
                final_output,final_feature,targets,
                criterion=nn.CrossEntropyLoss(), 
                useKD=False,loss_coefficient=0.9,temperature=3.0,
                useFD=False,feature_loss_coefficient=0.03):
    loss = criterion(final_output, targets)
    for i in range(len(moddle_outputs)):
        m_o = moddle_outputs[i]
        m_f = moddle_features[i]
        if useKD:
            loss += (1-loss_coefficient)*criterion(m_o,targets)+loss_coefficient*KDCrossEntropy(m_o,final_output,temperature)
        else:
            loss +=criterion(m_o,targets)
        if useFD:
            loss +=FDDistributionLoss(m_f,final_feature)*feature_loss_coefficient
    return loss

 # model_save_path
def save_checkpoint(state, save_path, is_best, filename, result):
    result_filename = os.path.join(save_path, 'scores.tsv')
    model_dir = os.path.join(save_path, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    logger.info("=> saving checkpoint '{}'".format(model_filename))

    torch.save(state, model_filename)

    with open(result_filename, 'w') as f:
        print('\n'.join(result), file=f)

    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if is_best:
        shutil.copyfile(model_filename, best_filename)

    logger.info("=> saved checkpoint '{}'".format(model_filename))
    return

def load_checkpoint(save_path):
    model_dir = save_path
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0]
    else:
        return None
    logger.info("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    logger.info("=> loaded checkpoint '{}'".format(model_filename))
    return state


def adjust_learning_rate(optimizer, epoch, epochs, init_lr, batch=None,
                        nBatch=None, method='multistep',
                        lr_decay_steps=None,
                        lr_decay_rate=0.1):
    if method == 'cosine':
        T_total = epochs * nBatch
        T_cur = (epoch % epochs) * nBatch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if lr_decay_steps is not None:
            # steps = [30,40,45]
            # 20 [0,0,0], [1,0,0], [1,1,0],[1,1,1]
            step = sum([epoch//i for i in lr_decay_steps])
            lr = init_lr*(lr_decay_rate**step)
        else:
            data = "cifar100"
            if data.startswith('cifar'):
                if epoch >= epochs * 0.75:
                    lr = init_lr* lr_decay_rate ** 2
                elif epoch >= epochs * 0.5:
                    lr = init_lr* lr_decay_rate
            else:
                lr = init_lr * (lr_decay_rate ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        # res.append(100.0 - correct_k.mul_(100.0 / batch_size))
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(train_loader, model, model_name,
            loss_fn, optimizer, nBlocks,
            epoch, epochs, 
            init_lr, 
            lr_decay_method="multistep",
            lr_decay_steps=None,
            lr_decay_rate=0.1,
            device=None,
            print_freq=100):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode
    model.train()

    end = time.time()
    running_lr = None
    for i, (input, target) in enumerate(train_loader):

        lr = adjust_learning_rate(optimizer, epoch, epochs, init_lr, batch=i,
                                nBatch=len(train_loader), 
                                method=lr_decay_method,
                                lr_decay_steps=lr_decay_steps,
                                lr_decay_rate=lr_decay_rate)
        # measure data loading time
        if running_lr is None:
            running_lr = lr

        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output, middle_feas = model(input_var)
        if not isinstance(output, list):
            output = [output]

        # loss = kd_loss.loss_fn_kd(output, target_var, output[-1])
        # MTLLossFunc
        if model_name.startswith("resnet18"):
            moddle_outputs = output[1:]
            moddle_features = middle_feas[1:]
            final_output = output[0]
            final_feature = middle_feas[0]
            best_acc_index = 0

        elif model_name.startswith("msdnet"):
            moddle_outputs = output[:-1]
            moddle_features = middle_feas[:-1]
            final_output = output[-1]
            final_feature = middle_feas[-1]
            best_acc_index = -1
        else:
            raise ValueError(f"Unknow model architecture {model_name}")
            exit
        
        loss = loss_fn(moddle_outputs,moddle_features,
                final_output,final_feature,target_var)
        
        losses.update(loss.item(), input.size(0))
        for j in range(len(output)):
            acc1, acc5 = accuracy(output[j].data, target, topk=(1, 5))
            top1[j].update(acc1.item(), input.size(0))
            top5[j].update(acc5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            # 输出所有中间层的精度（因为最后一个output可能对应第一层）
            ensemble = sum(moddle_outputs)/nBlocks
            ensemble.detach_()
            ensemble_acc1, ensemble_acc5 = accuracy(ensemble.data, target, topk=(1, 5))
            avg_acc1,avg_acc5 = 0,0
            acc_info='Train Epoch: [{0}][{1}/{2}] | ' \
                'Time: {batch_time.avg:.4f} ' \
                'Data: {data_time.avg:.4f} | ' \
                'Loss: {loss.val:.4f} | ' \
                'Acc '.format(epoch, i, len(train_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses)
            # logger.info()
            for block in range(nBlocks):
                acc_info+=f"{block+1}/{nBlocks}: {top1[block].avg:.4f} "
                # logger.info(f"{block+1}/{nBlocks}: {top1[block].avg:.4f} ",end="")
                avg_acc1+=top1[block].avg/nBlocks
                avg_acc5+=top5[block].avg/nBlocks
            acc_info+=f"Ensembale: {ensemble_acc1:.4f} Average: {avg_acc1:.4f} "
            logger.info(acc_info)
            
    return losses.avg, top1[best_acc_index].avg, top5[best_acc_index].avg, running_lr

def validate(val_loader, model, model_name, loss_fn, nBlocks, epoch, epochs, device,print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())
    ensemble_top1 = AverageMeter()
    ensemble_top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input = input.to(device)

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            # compute output
            output,middle_feas = model(input_var)
            if not isinstance(output, list):
                output = [output]

            # loss = kd_loss.loss_fn_kd(output, target_var, output[-1])
            # MTLLossFunc
            if model_name.startswith("resnet18"):
                moddle_outputs = output[1:]
                moddle_features = middle_feas[1:]
                final_output = output[0]
                final_feature = middle_feas[0]
                best_acc_index = 0
            elif model_name.startswith("msdnet"):
                moddle_outputs = output[:-1]
                moddle_features = middle_feas[:-1]
                final_output = output[-1]
                final_feature = middle_feas[-1]
                best_acc_index = -1
            else:
                raise ValueError(f"Unknow model architecture {model_name}")
                exit
            
            loss = loss_fn(moddle_outputs,moddle_features,
                    final_output,final_feature,target_var)

            # measure error and record loss
            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                acc1, acc5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(acc1.item(), input.size(0))
                top5[j].update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                # 输出所有中间层的精度（因为最后一个output可能对应第一层）
                ensemble = sum(moddle_outputs)/nBlocks
                ensemble.detach_()
                acc1, acc5 = accuracy(ensemble.data, target, topk=(1, 5))
                ensemble_top1.update(acc1.item(),input.size(0))
                ensemble_top5.update(acc5.item(),input.size(0))
                avg_acc1,avg_acc5 = 0,0
                acc_info='Valid Epoch: [{0}][{1}/{2}] | ' \
                    'Time: {batch_time.avg:.4f} ' \
                    'Data: {data_time.avg:.4f} | ' \
                    'Loss: {loss.val:.4f} | ' \
                    'Acc: '.format(
                        epoch, i + 1, len(val_loader),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses)
                for block in range(nBlocks):
                    acc_info+=f"{block+1}/{nBlocks}: {top1[block].avg:.4f} "
                    avg_acc1+=top1[block].avg/nBlocks
                    avg_acc5+=top5[block].avg/nBlocks
                acc_info+=f"Ensembale: {ensemble_top1.avg:.4f} Average: {avg_acc1:.4f} "
                logger.info(acc_info)

    avg_acc1,avg_acc5 = 0,0
    for j in range(nBlocks):
        logger.info(' * Acc@1 {top1.avg:.4f} Acc@5 {top5.avg:.4f}'.format(top1=top1[j], top5=top5[j]))
        avg_acc1+=top1[block].avg/nBlocks
        avg_acc5+=top5[block].avg/nBlocks
    logger.info(f' * Ensembale: Acc@1 {ensemble_top1.avg:.4f} Acc@5 {ensemble_top5.avg:.4f}')
    logger.info(f' * Average: Acc@1 {avg_acc1:.4f} Acc@5 {avg_acc5:.4f}')

    return losses.avg, top1[best_acc_index].avg, top5[best_acc_index].avg

def main(args):

    logger.info("\n# logging setting\n")
    os.makedirs(args.log_root,exist_ok=True)
    logging_file = os.path.join(args.log_root,
                    f"{args.model_name}-{args.model_desc}-dns_ratio-{args.dns_ratio}-traning.log")
    # 配置日志
    # logging.basicConfig(
    #     level=logging.DEBUG,  # 设置日志级别，可以选择 DEBUG、INFO、WARNING、ERROR、CRITICAL
    #     format='%(asctime)s [%(levelname)s] %(message)s',  # 设置日志格式
    #     handlers=[
    #         logging.FileHandler(logging_file),  # 输出到文�?
    #         logging.StreamHandler()  # 输出到控制台
    #     ]
    # )
    file_handler = logging.FileHandler(logging_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if not args.verbose:
        print(f"## Stop output to screen, please see the log in {logging_file}")
        logger.removeHandler(console_handler)
    logger.info("\n#######################Training Begin#######################\n")
    logger.info(f"## Saving log to {logging_file} for this experiments")

    logger.info("\n# Stage 1: Test enviroment\n")
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
    # 检�? GPU 是否可用  

    if torch.cuda.is_available():  
        logger.info("GPU is available")  
        # 获取 GPU 设备数量  
        num_gpus = torch.cuda.device_count()  
        logger.info(f"Having {num_gpus} GPU devices")
        # 获取 GPU 设备信息  
        for i in range(num_gpus):  
            logger.info(f"Using GPU{i}: {torch.cuda.get_device_properties(i)}")  
    else:  
        logger.info("GPU is not available")  
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    # 设置随机种子  
    torch.manual_seed(seed)  
    random.seed(seed)
    # 设置随机种子以确�? GPU 上的计算具有确定性行�?  
    torch.backends.cudnn.deterministic = True  
    logger.info(f"## Employ {device} with seed {seed} for this experiments")

    logger.info("\n# Stage 2: processing model architecture\n")
    model_name = args.model_name
    dns_ratio = max(min(args.dns_ratio,1),0)
    model = eval(model_name)(dns_ratio=dns_ratio).to(device)
    model_desc = args.model_desc
    model_save_name = f"{model_name}-{model_desc}"
    nBlocks = args.num_tasks
    logger.info(f"## Employ {model_save_name} model with dns-raio={dns_ratio:.2f} for this experiments")

    logger.info("\n# Stage 3: preprocessing dataset\n")
    # torch_weights_path=f"{os.environ['HOME']}/models/torch/weights"
    data_root=args.data_root
    dataset_name = args.data_name
    batch_size = args.batch_size
    num_workers = args.num_workers
    autoaugment=True
    if dataset_name.startswith("cifar10"):
        trainloader,testloader,classes = datasets.cifar.get_cifar_dataloader(
            train_batch_size=batch_size,val_batch_size=batch_size,
            root=data_root,dataset=dataset_name,
            autoaugment=autoaugment)
    elif dataset_name.startswith("imagenet"):
        trainloader,testloader = datasets.imagenet.get_dataset(
            data_path=data_root,batch_size=batch_size, workers=num_workers, 
            parse_type="torch",prefetch=True)
    else:
        logger.error(f"unkown data name {dataset_name}")
        exit
    logger.info(f"## Employ {dataset_name} dataset for this experiments")

    logger.info("\n# Stage 4: preparing optimizer and loss function \n")
    useKD = args.useKD
    useFD = args.useFD
    loss_coefficient=args.gamma
    feature_loss_coefficient=args.FD_loss_coefficient
    epochs=args.epochs
    autoaugment=args.autoaugment
    temperature=args.T
    batchsize=args.batch_size
    init_lr=args.lr
    lr_decay_method=args.lr_decay_type
    lr_decay_rate = args.lr_decay_rate
    lr_decay_steps = [int(v) for v in args.lr_decay_steps.split("-")]
    momentum = args.momentum
    weight_decay=args.weight_decay
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=init_lr, weight_decay=weight_decay, momentum=momentum)
    elif args.optimizer == "rmsprop":
        raise ValueError("Not Implemenetation for {args.optimizer} optimizer")
    elif args.optimizer == "adam":
        raise ValueError("Not Implemenetation for {args.optimizer} optimizer")
    # init = False
    
    loss_fn = MTLLoss(nn.CrossEntropyLoss(),useFD,loss_coefficient,temperature,
                      useFD,feature_loss_coefficient)
    
    logger.info(f'## Employ {args.optimizer} optimizer with (init_lr={init_lr},weight_decay={weight_decay},momentum={momentum})'
                f'and MTLLoss with (useFD={useFD},temperature={temperature},loss_coefficient={loss_coefficient},useFD={useFD},feature_loss_coefficient={feature_loss_coefficient}) for this experiments')
    
    logger.info("\n# Stage 5: resume from save model state dict \n")
    latest_checkpoint = args.latest_checkpoint
    resume = args.resume
    start_epoch=0
    if resume:
        if latest_checkpoint == "":
            logger.error("Please provide latest_checkpoint for resume training")
            exit
        else:
            # logger.info(f"Resume training from {latest_checkpoint}")
            state_dict = torch.load(latest_checkpoint)
            # 如果loading的是model state dict，那就直接loading
            try:
                start_epoch=0
                model.load_state_dict(state_dict)
            except Exception as e:
                # 如果loading的是trainig state dict，还需要还原optimizer和现�?
                # 'epoch': epoch,
                # 'model_name': model_name,
                # 'state_dict': model.state_dict(),
                # 'best_acc1': best_acc1,
                # 'optimizer': optimizer.state_dict(),
                start_epoch = state_dict["epoch"]
                model.load_state_dict(state_dict["state_dict"])
                optimizer.load_state_dict(state_dict["optimizer"])  
            finally:
                logger.info(f"## Resume training from {latest_checkpoint} checkpoint and epoch {start_epoch}")
    else:
        logger.info(f"## Training from Strach")

    # traning process
    best_acc1 = 0.0
    best_epoch = start_epoch
    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_acc1'
              '\tval_acc1\ttrain_acc5\tval_acc5']
    
    logger.info(f"\n# Stage 5: training from {start_epoch} to {epochs} epochs with total {len(trainloader)*(epochs-start_epoch)} iterations\n")
    for epoch in range(start_epoch, epochs):
        logger.info(f"#################epoch {epoch}#################")
        logger.info("lr=%.6f \n"%(optimizer.param_groups[0]['lr']))
        # train for one epoch
        train_loss, train_acc1, train_acc5, lr = train(trainloader, model, model_name,loss_fn, optimizer, 
                                                       nBlocks, epoch, epochs, init_lr, 
                                                       lr_decay_method,lr_decay_steps,lr_decay_rate,
                                                       device,
                                                       print_freq=args.print_freq)
        # train_loss, train_acc1, train_acc5, lr = 0,1,1,0.1

        # evaluate on validation set
        val_loss, val_acc1, val_acc5 = validate(testloader, model, model_name, loss_fn,nBlocks,epoch, epochs,
                                                device, print_freq=args.print_freq)
        # val_loss, val_acc1, val_acc5 = 0, 1, 1

        # save scores to a tsv file, rewrite the whole file to prevent
        # accidental deletion
        scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 6)
                    .format(epoch, lr, train_loss, val_loss,
                            train_acc1, val_acc1, train_acc5, val_acc5))

        is_best = val_acc1 > best_acc1
        if is_best:
            best_acc1 = val_acc1
            best_epoch = epoch
            logger.info('Best var_acc1 {}'.format(best_acc1))

        os.makedirs(args.model_saving_root,exist_ok=True)
        save_path = os.path.join(args.model_saving_root,f"{model_name}-{model_desc}-dns_ratio-{dns_ratio}-on-{dataset_name}")
        # save_path = f"../models/{model_name}-{model_desc}-dns_ratio-{dns_ratio}-on-{dataset_name}/"
        model_filename = 'latest-training-state-dict.pth.tar' 
        save_checkpoint({
            'epoch': epoch,
            'arch': model_name,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, save_path, is_best, model_filename, scores)

    # verifying and saving process
    logger.info('Best val_acc1: {:.4f} at epoch {}'.format(best_acc1, best_epoch))
    save_path = os.path.join(args.model_saving_root,f"{model_name}-{model_desc}-dns_ratio-{dns_ratio}-on-{dataset_name}")
    save_file_name = os.path.join(args.model_saving_root,f"{model_name}-{model_desc}-dns_ratio-{dns_ratio}-on-{dataset_name}-top1-{best_acc1}.pth")
    # torch.save(model.state_dict(),save_file_name)
    logger.info(f"training state dict are saved to folder {save_path}")
    state_dict = torch.load(os.path.join(save_path,"save_models","model_best.pth.tar"))
    torch.save(state_dict["state_dict"],save_file_name)
    logger.info(f"save best model weight is saved to {save_file_name}")

    logger.info("\n#######################Training Finished#######################\n")
    logger.removeHandler(file_handler)

if __name__ == '__main__':
    args = arg_parser.parse_args()
    main(args)
