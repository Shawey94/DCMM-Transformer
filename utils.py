import pickle
import random
import os
import torch
import numpy as np
import networkx as nx
from itertools import combinations, groupby
from parameters import *
import torch.distributed as dist

import torch.nn.functional as F
import matplotlib.pyplot as plt


def visualize_attention(attn_ori, cp_bias_ori, B_aff, degree_term, cp_bias, attn_cp, batch_idx=0, head_idx=0, title_prefix=""):
    """
    Visualize the attention maps.
    Args:
        attn_orig: Tensor [B, H, N, N] — raw attention before bias
        cp_bias:   Tensor [B, H, N, N] — structural bias z_i * z_j
        attn_cp:   Tensor [B, H, N, N] — final attention after CP bias
    """

    # Convert tensors to numpy
    attn_orig_np = attn_ori[batch_idx, head_idx].detach().cpu().numpy()
    cp_bias_np =  cp_bias[batch_idx, head_idx].detach().cpu().numpy()  #opt.cp_bias_scale *
    attn_cp_np = attn_cp[batch_idx, head_idx].detach().cpu().numpy()
    degree_term = degree_term[batch_idx, head_idx].detach().cpu().numpy()
    cp_bias_ori = cp_bias_ori[batch_idx, head_idx].detach().cpu().numpy()
    B_aff = B_aff[batch_idx, head_idx].detach().cpu().numpy()

    vmin = min(attn_orig_np.min(), attn_cp_np.min())
    vmax = max(attn_orig_np.max(), attn_cp_np.max())

    fig, axs = plt.subplots(1, 6, figsize=(15, 4))

    im0 = axs[0].imshow(attn_orig_np, cmap='viridis', )         #vmin=vmin, vmax=vmax
    axs[0].set_title(f"{title_prefix}Original Attention batch idx{batch_idx} at head{head_idx}")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 =axs[1].imshow(cp_bias_ori, cmap='viridis', )
    axs[1].set_title(f"{title_prefix}CP Bias Ori batch idx{batch_idx} at head{head_idx}")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 =axs[2].imshow(degree_term, cmap='viridis', )
    axs[2].set_title(f"{title_prefix}degree_term batch idx{batch_idx} at head{head_idx}")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    im3 =axs[3].imshow(cp_bias_np, cmap='viridis', )
    axs[3].set_title(f"{title_prefix}CP Bias batch idx{batch_idx} at head{head_idx}")
    fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

    im4 = axs[4].imshow(attn_cp_np, cmap='viridis', )
    axs[4].set_title(f"{title_prefix}Attention + CP Bias batch idx{batch_idx} at head{head_idx}")
    fig.colorbar(im4, ax=axs[4], fraction=0.046, pad=0.04)

    im5 = axs[5].imshow(B_aff, cmap='viridis', )
    axs[5].set_title(f"{title_prefix} B matrix idx{batch_idx} at head{head_idx}")
    fig.colorbar(im5, ax=axs[5], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xlabel("Key Tokens")
        ax.set_ylabel("")

    plt.tight_layout()
    plt.show()

class SampleGraphSparseGraph(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        A = torch.bernoulli(torch.clamp(input+0.01, min=0, max=1)).requires_grad_(True)
        ctx.save_for_backward(A)
        return A
        
    def backward(ctx, grad_output):
        A, = ctx.saved_tensors
        return F.hardtanh(A*grad_output)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    # pdb.set_trace()
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

if __name__ == '__main__':
    
    img = torch.from_numpy(np.random.rand(1,1,224,224))
    indice = np.random.randint(16,size=(1, 4))
    print(indice)
    img = re_org_img(img,indice)

    #
    proj = torch.nn.Conv2d(1, 1, kernel_size=2, stride=2)
    x = torch.from_numpy(np.random.rand(1,1,6,4)).float()
    print(x)
    x = proj(x)
    print(x)
    
    '''

    x = torch.Tensor(2, 8, 3)
    indice = torch.randint(0, 8, (2, 1, 2))
    print(x)
    print(indice)
    x = re_org_pathch_embeds(x, indice)
    print(x)
    '''
