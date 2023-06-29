import os
import argparse
import torch
from vnet import VNet
from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/mnt/UGPC/Training_Set/', help='Name of Experiment')#train_set地址
parser.add_argument('--model', type=str,  default='ugpc', help='model_name')
parser.add_argument('--gpu', type=str,  default='2', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# snapshot_path = "../model/"+FLAGS.model+"/"
test_save_path = "/mnt/3333/model/prediction/"+FLAGS.model+"_post/"#项目地址
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

with open(FLAGS.root_path + '/../test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path +item.replace('\n', '')+"/mri_norm2.h5" for item in image_list]


def test_calculate_metric(save_model_path):
    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    net.load_state_dict(torch.load(save_model_path))
    print("init weight from {}".format(save_model_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric

if __name__ == '__main__':
    model_save_path = '/mnt/model/UAMT-ada4-ema/decay-0.9999/16_labeled/iter_1000.pth'#权重地址：bestperformance.path
    metric = test_calculate_metric(model_save_path)
    print(metric)