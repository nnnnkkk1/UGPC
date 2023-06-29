import os
import argparse
from val_util import val_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str,  default='2', help='GPU to use')
FLAGS = parser.parse_args()


def val_calculate_metric(net,image_list,num_classes):

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    net.eval()

    dice,_,_,_,_,_,_ = val_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=False)

    return dice
