import torch
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--weight_decay', default=1e-2, type=float)
parser.add_argument('--opt', default='adamw', type=str)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--model_saved_path', type=str, default='./models_saved')
parser.add_argument('--imagenet_path', type=str, default='/home/cxc/Downloads/xiaowei_stuff/CP-DCMM-Transformer/CPDCMM_ViT/data/ImageNet/ImageNet1K/')
#/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/CP_ViT_ImageNet/ImageNet1K/
parser.add_argument('--tinyImagenet_path', type=str, default='/home/cxc/Downloads/xiaowei_stuff/CP-DCMM-Transformer/CPDCMM_ViT/data/TinyImageNet/TinyImageNet/')
#/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/CP_ViT_TinyImageNet/TinyImageNet/
parser.add_argument('--ADNI_path', type=str, default='/media/xw_stuff/CP-DCMM-Transformer/adni')
parser.add_argument('--SIIMACR_path', type=str, default='/data/xw_stuff/CP-DCMM-Transformer/SIIM-ACR')
parser.add_argument('--INbreast_path', type=str, default='/data/xw_stuff/CP-DCMM-Transformer/INbreast')
parser.add_argument('--ChestXray_path', type=str, default='/data/xw_stuff/CP-DCMM-Transformer/chest_xray')
parser.add_argument('--log_step', type=int, default=2000, help='log_step')
parser.add_argument('--batch_size', type=int, default=1)  #batch size on each gpu
parser.add_argument('--te_batch_size', type=int, default=1)  #batch size on each gpu
parser.add_argument('--warm_up', default=4, type=int)
parser.add_argument('--layer', default=11, type=int)
parser.add_argument('--strength', default=0.0, type=float)
parser.add_argument('--gpu_id', default='0', type = str, help="gpu id")
parser.add_argument('--res', default=224, type = int,
                        help="image resolution")
parser.add_argument('--patch_size', default=16, type = int,
                        help="patch size")
parser.add_argument('--scale', default='small', type = str,
                        help="model scale")
parser.add_argument('--datasets', default='TinyImageNet', type = str,
                        help="choose dataset to experiment, ImageNet, TinyImageNet, Cifar100, Cifar10, INbreast, SIIMACR")
parser.add_argument('--num_classes', default=3, type = int,
                        help="number of class categories")
parser.add_argument('--training', default=0, type = int,
                        help="injection in training stage.")
parser.add_argument('--inference', default=0, type = int,
                        help="injection in inference stage.")
parser.add_argument('--num_patches', default=196, type = int,
                        help="number of patches.")

parser.add_argument('--use_cp', default=1, type = int,
                        help="use or not use cp-dcmm")     

parser.add_argument('--lamda_e', default=0.1, type = float,
                        help="pi range (0,1)")     
parser.add_argument('--lamda_d', default=0.0, type = float,
                        help=" degree corrected loss, theta range (0.1), sigmoid is enough, so this term to be 0")     
parser.add_argument('--lamda_s', default=0.0, type = float,
                        help="affinity matrix B range (0,1), sigmoid is enough, so this term to be 0")     

#Mixed-membership cluster number
parser.add_argument("--n_clusters", type= int, default = 100)

parser.add_argument("--cp_bias_scale", type= float, default = 10)

opt = parser.parse_args()

expected_patches = (opt.res // opt.patch_size) ** 2
if opt.num_patches != expected_patches:
    print(f"Error: --num_patches ({opt.num_patches}) does not match expected value ({expected_patches}).")
    sys.exit(1)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
#cuda = True if torch.cuda.is_available() else False
#Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#device = torch.device('cuda' if cuda else 'cpu')

#if torch.cuda.device_count() > 1:
#          print("Let's use", torch.cuda.device_count(), "GPUs!")
