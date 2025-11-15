# [DCMM-Transformer: Degree-Corrected Mixed-Membership Attention for Medical Imaging](https://arxiv.org/pdf/2411.07794), AAAI 2026

### updates (11/15/2025)
<!--  Add the environment requirements to reproduce the results.  --> 

<p align="left"> 
<img width="800" src="https://github.com/Shawey94/DCMM-Transformer/blob/main/image.png">
</p>

### Environment (Python 3.8.12)
```
# Install Anaconda (https://docs.anaconda.com/anaconda/install/linux/)
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh

# Install required packages
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch
pip install tqdm==4.50.2
pip install tensorboard==2.8.0
# apex 0.1
conda install -c conda-forge nvidia-apex
pip install scipy==1.5.2
pip install timm==0.6.13
pip install ml-collections==0.1.0
pip install scikit-learn==0.23.2
```

### Pretrained ViT
Download the following models and put them in `checkpoint/`
- ViT-B_16 [(ImageNet-21K)](https://storage.cloud.google.com/vit_models/imagenet21k/ViT-B_16.npz?_ga=2.49067683.-40935391.1637977007)
- ViT-B_16 [(ImageNet)](https://console.cloud.google.com/storage/browser/_details/vit_models/sam/ViT-B_16.npz;tab=live_object)
- ViT-S_16 [(ImageNet)](https://storage.googleapis.com/vit_models/sam/ViT-S_16.npz)


<!-- 

 --> 

### Datasets:

- [ChestXray](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- SIIM-ACR
- [INbreast](https://pubmed.ncbi.nlm.nih.gov/22078258/)
- [ADNI](https://adni.loni.usc.edu/)
- [EyeDisease](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)

### Training:

All commands can be found in `script.txt`. An example:
```
python Main_wTop5.py --lr 0.00005 --epochs 100 --batch_size 128 --te_batch_size 128 --layer 11 --gpu_id 0 --res 224 --patch_size 16 --scale small --datasets INbreast --num_classes 3 --training 1 --inference 0 --strength 0.0 --use_cp 1 --cp_bias_scale 10 --n_clusters 100 --lamda_e 0.1 --lamda_s 0.0 --lamda_d 0.0

```

<!-- 

 --> 

### Citation:
```
@article{cheng2024DCMM,
  title={DCMM-Transformer: Degree-Corrected Mixed-Membership Attention for Medical Imaging},
  author={Cheng, Huimin and Yu, Xiaowei and Wu, Shushan and Fang, Luyang and Cao, Chao and Zhang, Jing and Liu, Tianming and Zhu, Dajiang and Zhong, Wenxuan and Ma, Ping},
  journal={arXiv preprint arXiv:2411.07794},
  year={2025}
}
```
Our code is largely borrowed from [NoisyNN](https://github.com/Shawey94/NoisyNN) and [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
