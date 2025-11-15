
import torch
from torch import nn
from torch._C import set_flush_denormal
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import nibabel as nib
import numpy as np
import os
from torch.autograd import Variable
from parameters import *
import seaborn as sns
import matplotlib.pyplot as plt
#from dataloader import *
from timm.models.vision_transformer import partial
from timm.models.layers.weight_init import trunc_normal_
from timm.models.vision_transformer import LayerScale
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg, named_apply, adapt_input_conv, checkpoint_seq
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import Mlp
from utils import *
import sys

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message

def LinearQ(x):
    hidden_states_copy = x.clone()
    hidden_states_copy = torch.cat( (hidden_states_copy[1:,1:,:], hidden_states_copy[0,1:,:].unsqueeze(0)),dim=0 )
    noise = hidden_states_copy - x[:,1:,:].clone()
    x[:,1:,:] = x[:,1:,:] + opt.strength * noise  
    return x

def init_weights_vit_moco(module: nn.Module, name: str = ''):
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


        self.cp_bias_scale = nn.Parameter(torch.tensor(opt.cp_bias_scale,  dtype=torch.float32)) 
        self.n_clusters = opt.n_clusters

        # CP-DCMM group assignment projection from query
        self.group_proj = nn.Linear(self.head_dim, self.n_clusters)

        # Degree scalar projection from query
        self.theta_proj = nn.Linear(self.head_dim, 1)

        # Affinity matrix B: [H, K, K]
        self.affinity_B = nn.Parameter(torch.randn(self.num_heads, self.n_clusters, self.n_clusters)  )

    def forward(self, x, flag, injection=False, cp_dcmm = 1):
        
        if(flag and injection):
            x = LinearQ(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        _, h, _, _ = q.shape
        _, _, n, _ = v.shape

        attn = (q @ k.transpose(-2, -1)) * self.scale 
        
        z_constraint = torch.tensor(0.0, device=x.device)
        B_constraint = torch.tensor(0.0, device=x.device)
        theta_constraint = torch.tensor(0.0, device=x.device)
        cp_bias = torch.zeros_like(attn).to(device=x.device)

        if cp_dcmm and flag:
            attn_ori = attn.clone()
            # Group assignment z from Q: [B, H, N, K]
            z_logits = self.group_proj(q)  # [B, H, N, K]
            z = F.softmax(z_logits, dim=-1)

            # z_i B z_j^T: compute structural bias
            B_aff = F.sigmoid(self.affinity_B).unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, K, K]
            cp_bias_ori = torch.einsum("bhik,bhkl,bhjl->bhij", z, B_aff, z).tanh()

            # Degree correction from Q
            theta = F.relu(self.theta_proj(q).squeeze(-1))  # [B, H, N] non-negative
            theta_i = theta.unsqueeze(-1)           # [B, H, N, 1]
            theta_j = theta.unsqueeze(-2)           # [B, H, 1, N]
            degree_term = theta_i * theta_j         # [B, H, N, N]

            # CP-DCMM bias
            #cp_bias = self.cp_bias_scale * cp_bias_ori * degree_term

            # CP-DCMM bias
            cp_bias = self.cp_bias_scale *  degree_term * cp_bias_ori 

            # Add to original attention
            attn = attn + cp_bias   # add CP bias

            z_constraint = - (z * torch.log(z + 1e-8)).sum(dim=-1).mean()
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)


        #if (cp_dcmm and injection and flag):  # Only save during training to avoid redundant writes during eval
            
            #visualize_attention(attn_ori, cp_bias_ori, B_aff, degree_term, self.cp_bias_scale * cp_bias, attn, batch_idx=0, head_idx=0)
            # save_dir = '/media/xw_stuff/CP-DCMM-Transformer/CPDCMM-VIT_2_ExtraLoss_LastLayer/graphs_saved'  # Change this to your desired path
            # os.makedirs(save_dir, exist_ok=True)
            
            # torch.save(degree_term.detach().cpu(), os.path.join(save_dir, f'degree_term_dataset_{opt.datasets}.pt'))
            # torch.save(B_aff.detach().cpu(), os.path.join(save_dir, f'B_aff_dataset_{opt.datasets}.pt'))
            # torch.save(cp_bias.detach().cpu(), os.path.join(save_dir, f'cp_mask_dataset_{opt.datasets}.pt'))

        return x, z_constraint, B_constraint, theta_constraint, cp_bias

class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, flag = False, injection=False, cp_dcmm = 1):  # default: not add noise
        x_attn, z_entropy, B_sparse, theta_norm, dcmm_mask = self.attn(self.norm1(x), flag, injection, cp_dcmm)
        x = x + self.drop_path1(self.ls1(x_attn))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, z_entropy, B_sparse, theta_norm, dcmm_mask

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=opt.res, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size) # Batch * 3 * 224 *224 -> Batch * 768 * 14 * 14
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, no_embed_class=False, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x, noise_layer_index, strength):
        x = self.patch_embed(x)

        if(noise_layer_index == -1):  #-1 means after embedding before attn
            hidden_states_copy = x.detach()
            hidden_states_copy = torch.cat( (hidden_states_copy[1:,:,:], hidden_states_copy[0,:,:].unsqueeze(0)),dim=0 )
            x = x + strength * (hidden_states_copy - x.detach())

        x = self._pos_embed(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            for i in range(len(self.blocks)):
                if( i == noise_layer_index ):
                    x = self.blocks[i](x, True, strength) # add noise to the layer
                else:
                    x = self.blocks[i](x, False) # not add noise to the layer
            #x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    '''
    def forward(self, x, noise_layer_index, strength):
        x = self.forward_features(x, noise_layer_index, strength)
        x = self.forward_head(x)
        return x
    '''

class ViT(VisionTransformer):
    def __init__(self, patch_size, num_classes, embed_dim, depth, num_heads, ):
        super().__init__(patch_size=patch_size, num_classes=num_classes, embed_dim = embed_dim, depth = depth, num_heads = num_heads, img_size = opt.res)
        self.layer = depth  
    
    def forward(self, x, noise_layer_index = 11, injection=True, cp_dcmm = 1):   #default to the last layer
        x = self.patch_embed(x)
        B, num_patches = x.shape[0], x.shape[1]

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim = 1)
        x = self.pos_drop(x + self.pos_embed)

        total_entropy_loss = 0
        total_B_sparse = 0
        total_theta_norm = 0
        for i in range(self.layer):
            if(i == noise_layer_index):
                x, z_entropy, B_sparse, theta_norm, dcmm_mask = self.blocks[i](x, True, injection, cp_dcmm)
            else:
                x, z_entropy, B_sparse, theta_norm, dcmm_mask = self.blocks[i](x, False, injection, cp_dcmm)
            total_entropy_loss = total_entropy_loss + z_entropy
            total_B_sparse = total_B_sparse + B_sparse
            theta_norm = theta_norm + total_theta_norm
        x = self.norm(x)
        x = self.fc_norm(x)
        return self.head(x[:,0]), total_entropy_loss, total_B_sparse, theta_norm, dcmm_mask








if __name__ == '__main__':

    print('debug purpose.')

    
