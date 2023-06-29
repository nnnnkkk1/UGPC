import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# val import val_calculate_metric
from vnet import VNet
from Utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/mnt/UGPC/Training_Set', help='Name of Experiment')#’Training_Set‘地址
parser.add_argument('--exp', type=str,  default='UAMT-ada4-ema/decay-0.9999', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=1000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')       #可以改成0.1试试---太大了
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=6666, help='random seed')          #1337  改成1266后效果不错[ 0.84834708  0.7521455   2.91170339 10.40884597  0.89779118  0.82762218 0.98920743]
#1200效果不好     666效果好 [0.8573699   0.75694815 2.65368517 9.86923351 0.92795734 0.8072097 0.99323964]
parser.add_argument('--gpu', type=str,  default='2', help='GPU to use')

parser.add_argument('--pseudo', action='store_true', default=True, help='generate the pseudo label')
parser.add_argument('--pseudo_rect', action='store_true', default=False, help='Rectify the pseudo label')
parser.add_argument('--threshold', type=float, default=0.90, help='pseudo label threshold')
parser.add_argument('--T', type=float, default=1)
parser.add_argument('--ratio', type=float,  default=0.10, help='model noise ratio')
parser.add_argument('--dropout_rate', type=float,  default=0.5)
### costs
parser.add_argument('--ema_decay', type=float,  default=0.9999, help='ema_decay')
parser.add_argument('--consistency', type=float,  default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')

parser.add_argument('--num_labeled', type=int, default=16, help='number of labeled')
parser.add_argument('--num_class', type=int, default=2, help='number of classes')

args = parser.parse_args()

num_labeled = args.num_labeled
train_data_path = args.root_path



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = args.num_class
patch_size = (112, 112, 80)

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)



# def update_variance(pred1, pred2, loss_origin):
#
#     mse_distance = losses.softmax_mse_loss
#     loss_mse = torch.sum(mse_distance(pred1, pred2), dim=1)
#     exp_loss_kl = torch.exp(-loss_mse)
#     loss_rect = torch.mean(loss_origin * exp_loss_kl) + torch.mean(loss_mse)
#     return loss_rect
#
def update_variance(pred1, pred2, loss_origin):
    sm = nn.Softmax(dim=1)
    log_sm = nn.LogSoftmax(dim=1)
    mse_distance = losses.softmax_mse_loss
    loss_mse = torch.sum(mse_distance(log_sm(pred1), sm(pred2)), dim=1)
    exp_loss_mse = torch.exp(-loss_mse)
    loss_rect = torch.mean(loss_origin * exp_loss_mse) + torch.mean(loss_mse)
    return loss_rect

def update_consistency_loss(pred1, pred2):
    if args.pseudo:
        criterion = nn.CrossEntropyLoss(reduction='none')
        pseudo_label2 = torch.softmax(pred2.detach() / args.T, dim=1)
        max_probs2, targets2 = torch.max(pseudo_label2, dim=1)
        loss_ce1 = criterion(pred1, targets2)
        loss1 = update_variance(pred1, pred2, loss_ce1)
        pseudo_label1 = torch.softmax(pred1.detach() / args.T, dim=1)
        max_probs1, targets1 = torch.max(pseudo_label1, dim=1)
        loss_ce2 = criterion(pred2, targets1)
        loss2 = update_variance(pred1, pred2, loss_ce2)
        loss = (loss1 + loss2)*0.5
    return loss



if __name__ == "__main__":
    snapshot_path = "../model/{}/{}_labeled".format(
        args.exp, args.num_labeled)#项目文件夹地址
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')


    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False, has_dropout=False):
        # Network definition
        if has_dropout:
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True,
                       dropout_rate=args.dropout_rate)
        else:
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    model = create_model(has_dropout=True)  # student model
    ema_model = create_model(ema=True, has_dropout=True)  # teacher model

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))

    all_idxs = list(range(0, len(db_train)))
    np.random.seed(6)
    unlabeled_idxs = np.random.choice(all_idxs, len(db_train) - args.num_labeled, replace=False)
    labeled_idxs = np.setdiff1d(all_idxs, unlabeled_idxs)
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)


    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr

    # bestperformance = 0
    # with open('/mnt/UGPC/val.list', 'r') as f:
    #     val_image_list = f.readlines()#验证集list地址
    # val_image_list = [args.root_path+'/' + item.replace('\n', '') + "/mri_norm2.h5" for item in val_image_list]

    model.train()
    ema_model.train()
    time1 = time.time()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            ratio = args.ratio
            noise = torch.clamp(torch.randn_like(volume_batch) * ratio, -(2 * ratio), (2 * ratio))
            noise_strong = torch.clamp(torch.randn_like(volume_batch) * ratio, -(3 * ratio), (3 * ratio))
            student_inputs = volume_batch + noise
            ema_inputs = volume_batch + noise
            outputs = model(student_inputs)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)


            loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])   # 只取labeld的output
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            supervised_loss = 0.5 * (loss_seg + loss_seg_dice)  # only on labeled data


            # 计算consisitency loss
            consistency_loss = update_consistency_loss(outputs, ema_output)


            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/supervised_loss', supervised_loss, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)


            logging.info('iteration %d : loss : %f supervised_loss: %f consistency_loss: %f consistency_weight: %f' %
                         (iter_num, loss.item(), supervised_loss.item(), consistency_loss.item(), consistency_weight))


            # if iter_num < 1000 and iter_num % 20 == 0:
            #     valdice  = val_calculate_metric(model,val_image_list,num_classes)
            #     performance = valdice
            #     if performance >= bestperformance:
            #         bestperformance = performance
            #         save_mode_path = os.path.join(snapshot_path, 'bestpe formance' + '.pth')
            #         torch.save(model.state_dict(), save_mode_path)
            #         logging.info("iter_{}_save model".format(iter_num))
            #         model.train()
            # if iter_num >=1000 and iter_num % 200 == 0:
            #     valdice  = val_calculate_metric(model,val_image_list,num_classes)
            #     performance = valdice
            #     if performance >= bestperformance:
            #         bestperformance = performance
            #         save_mode_path = os.path.join(snapshot_path, 'bestperformance' + '.pth')
            #         torch.save(model.state_dict(), save_mode_path)
            #         logging.info("iter_{}_save model".format(iter_num))
            #         model.train()

            ## change lr
            if iter_num % 2500 == 0:             #原来是2500
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                #lr_ = base_lr * （1.0 - iter_num / max_iterations） ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if (iter_num % 1000 == 0) & (iter_num >= 1000):
                save_mode_path = os.path.join(snapshot_path, 'ada_iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break

    time2 = time.time()
    total_time = (time2 - time1) / 3600
    print('total train time:', total_time)

    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
    torch.save(model.state_dict(), save_mode_path)

    logging.info("save model to {}".format(save_mode_path))
    writer.close()
