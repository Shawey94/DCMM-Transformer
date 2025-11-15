
from ViT import *
import numpy as np
import itertools
import time
from parameters import *
import torch
from Accdataloader import *
from utils import *
import timm
from timm.loss import LabelSmoothingCrossEntropy
import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
import logging
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./exp_log')

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')

transform_train = transforms.Compose([
        transforms.RandomResizedCrop((opt.res, opt.res), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

transform_test = transforms.Compose([
        transforms.Resize((opt.res, opt.res)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

transform = transforms.Compose([
        transforms.Resize((opt.res, opt.res)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

transform_adni = transforms.Compose([
        transforms.Resize((opt.res, opt.res)),
        #transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def train():

    if(opt.scale == 'tiny'):
        Layers = 12
        HiddenSize = 192
        Heads = 3
        MLPSize = 768   #MLP ratio 4
    elif(opt.scale == 'small'):
        Layers = 12
        HiddenSize = 384
        Heads = 12
        MLPSize = 1536   #MLP ratio 4
    elif(opt.scale == 'base'):
        Layers = 12
        HiddenSize = 768
        Heads = 12
        MLPSize = 3072
    elif(opt.scale == 'large'):
        Layers = 24
        HiddenSize = 1024
        Heads = 16
        MLPSize = 4096
    elif(opt.scale == 'huge'):
        Layers = 32
        HiddenSize = 1280
        Heads = 16
        MLPSize = 5120

    vit = ViT(patch_size = opt.patch_size, num_classes=opt.num_classes, embed_dim=HiddenSize, depth=Layers, num_heads=Heads)
    vit.to(device)

    saved_models = os.listdir(opt.model_saved_path)
    flag = 1   
    for saved_model in saved_models:

        saved_acc = float(saved_model.split('_')[1])
        saved_top5 = float(saved_model.split('_')[3])
        saved_lr = saved_model.split('_')[5]
        saved_bs = saved_model.split('_')[7]
        saved_layer = saved_model.split('_')[9]
        saved_scale = saved_model.split('_')[10]
        saved_res = int(saved_model.split('_')[11])
        saved_patch_size = int(saved_model.split('_')[12])
        saved_datasets = saved_model.split('_')[13]
        saved_cpdcmm = int((saved_model.split('_')[15]))
        saved_cpscale = float((saved_model.split('_')[17]))
        saved_clusters = int((saved_model.split('_')[19]))
        saved_lambdae = float((saved_model.split('_')[21]))
        saved_lambdas = float((saved_model.split('_')[23]))
        saved_lambdad = float((saved_model.split('_')[25]).split('.')[0])

        if( str(opt.lr)==saved_lr and str(opt.batch_size)==saved_bs and str(opt.layer)==saved_layer and opt.scale==saved_scale and opt.res==saved_res and \
            opt.patch_size==saved_patch_size and saved_datasets == opt.datasets and saved_cpdcmm == opt.use_cp and saved_cpscale == opt.cp_bias_scale and saved_clusters == opt.n_clusters \
            and saved_lambdae == opt.lamda_e and\
            saved_lambdas == opt.lamda_s and saved_lambdad == opt.lamda_d):
            #vit.load_state_dict(torch.load(opt.model_saved_path+'/'+saved_model, ))
            last_th = saved_acc
            #last_th = 0
            print('threshold: ', last_th)
            flag = 1
            break
    
    if(flag):
        print('pretrained mode: '+'vit_'+opt.scale+'_patch'+str(opt.patch_size)+'_'+str(opt.res))
        if(opt.scale == 'huge'):
            vit_pretrain = timm.create_model('vit_'+opt.scale+'_patch'+str(opt.patch_size)+'_'+str(opt.res)+'_in21k', pretrained=True, num_classes = opt.num_classes)
        else:
            vit_pretrain = timm.create_model('vit_'+opt.scale+'_patch'+str(opt.patch_size)+'_'+str(opt.res), pretrained=True, num_classes = opt.num_classes)  #vit_small_patch16_224, vit_base_patch16_224
        vit_pretrain_weight = vit_pretrain.state_dict()
        vit.load_state_dict(vit_pretrain_weight, strict=False)   
        #if local_rank == 0:
        #    dist.barrier()
        last_th = 0

    '''
    for k,v in vit.named_parameters():
        print('layer:', k)
        if('attn' not in k and 'head' not in k and 'patch_embed' not in k and 'cls_token' not in k and 'pos_embed' not in k):
            print('freeze layer')
            v.requires_grad = False
    '''
    print('model parameters:', sum(param.numel() for param in vit.parameters()) /1e6)
    #for name, param in vit.named_parameters():
    #    if(param.requires_grad):
    #        print(name)
    
    if(opt.datasets == 'ImageNet'):
        tra_dataloader, te_dataloader = get_loader(opt.imagenet_path)
    elif(opt.datasets == 'TinyImageNet'):
        tra_dataloader, te_dataloader = get_loader(opt.tinyImagenet_path)
    elif(opt.datasets == 'Cifar100'):
        train_data = datasets.CIFAR100(root='/media/xw_stuff/CP-DCMM-Transformer/CPDCMM-VIT_2/data', train=True,transform=transform_train,download=True)
        test_data =datasets.CIFAR100(root='/media/xw_stuff/CP-DCMM-Transformer/CPDCMM-VIT_2/data',train=False,transform=transform_test,download=True)
        tra_dataloader = DataLoader(dataset=train_data,batch_size=opt.batch_size,shuffle=True,)
        te_dataloader = DataLoader(dataset=test_data,batch_size=opt.batch_size,shuffle=False,)
    elif(opt.datasets == 'Cifar10'):
        train_data = datasets.CIFAR10(root='/media/xw_stuff/CP-DCMM-Transformer/CPDCMM-VIT_2/data', train=True,transform=transform_train,download=True)
        test_data =datasets.CIFAR10(root='/media/xw_stuff/CP-DCMM-Transformer/CPDCMM-VIT_2/data',train=False,transform=transform_test,download=True)
        tra_dataloader = DataLoader(dataset=train_data,batch_size=opt.batch_size,shuffle=True,)
        te_dataloader = DataLoader(dataset=test_data,batch_size=opt.batch_size,shuffle=False,)
    elif(opt.datasets == 'INbreast'):
        dataset = CustomedINbreast(root_dir= opt.INbreast_path+'/train', transform=transform_train)
        tra_dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)

        dataset = CustomedINbreast(root_dir= opt.INbreast_path+'/test', transform=transform_test)
        te_dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size, shuffle= False)
    elif(opt.datasets == 'SIIMACR'):
        # All image paths
        all_image_paths = sorted([
            os.path.join(opt.SIIMACR_path, fname)
            for fname in os.listdir(opt.SIIMACR_path)
            if fname.endswith('.jpg')
        ])

        # Shuffle and split
        random.seed(42)  # for reproducibility
        random.shuffle(all_image_paths)
        train_paths = all_image_paths[:1000]
        test_paths = all_image_paths[1000:]

            # Datasets
        train_dataset = SIIMACRDataset(train_paths, transform=transform_train)  #transform_train
        test_dataset = SIIMACRDataset(test_paths, transform=transform_test)

        # Dataloaders
        tra_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        te_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    elif(opt.datasets == 'ADNI'):
        # Set random seed for reproducibility
        generator = torch.Generator().manual_seed(42)

        dataset = ADNIDataset(opt.ADNI_path, label_csv = '/media/xw_stuff/CP-DCMM-Transformer/AD_labels.csv', transform=transform_adni)

        total_len = len(dataset)
        train_len = int(0.9 * total_len)
        test_len = total_len - train_len

        train_dataset, test_dataset = random_split(dataset, [train_len, test_len], generator=generator)

        tra_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        te_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    elif(opt.datasets == 'ChestXray'):
        train_dataset = datasets.ImageFolder(root=opt.ChestXray_path+'/train', transform=transform_train)  #transform_train
        test_dataset = datasets.ImageFolder(root=opt.ChestXray_path+'/test', transform=transform_test)

        # Dataloaders
        tra_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
        te_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)


    
    start_t = time.time()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, vit.parameters()), 
                                        lr=opt.lr, weight_decay=opt.weight_decay,)
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, vit.parameters()), 
    #                                    lr=opt.lr, weight_decay=opt.weight_decay, momentum = 0.9)
    #optimizer = create_optimizer(opt, vit)
    #scheduler = CosineAnnealingLR(optimizer, T_max=opt.epochs/2, eta_min=1e-6 )
    scheduler = CosineLRScheduler(optimizer, t_initial=opt.epochs, lr_min=1e-8, warmup_t=opt.warm_up)
    # loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction=opt.cross_loss_para)
    loss_fn = LabelSmoothingCrossEntropy(0.1)

    for epoch in range(opt.epochs):
        vit.train()

        num_steps_per_epoch = len(tra_dataloader)
        num_updates = epoch * num_steps_per_epoch

        batch_time = AverageMeter()  # forward prop. + back prop. time
        accs = AverageMeter()
        epoch_tra_loss = AverageMeter()  
        
        total = 0
        correct = 0
        #tra_dataloader.sampler.set_epoch(epoch)  # randomize the training data
        pbar = tqdm(enumerate(tra_dataloader),total=len(tra_dataloader),)
        #for i, (tra_transformed_normalized_img, tra_labels) in enumerate(tra_dataloader):
        for i, (tra_transformed_normalized_img, tra_labels) in pbar:
            batchSize = tra_transformed_normalized_img.shape[0]
            #print('current lr: ', (optimizer.state_dict()['param_groups'][0]['lr']))

            #------------------------------------------------------
            tra_transformed_normalized_img = tra_transformed_normalized_img.float().to(device)
            
            outputs, pi_loss, B_loss, theta_loss, dcmm_mask = vit(tra_transformed_normalized_img, noise_layer_index = opt.layer, injection=opt.training, cp_dcmm = opt.use_cp) 
            cls_loss = loss_fn(outputs, tra_labels.cuda())

            total_loss = cls_loss + opt.lamda_e * pi_loss + opt.lamda_s * B_loss + opt.lamda_d * theta_loss

            #acc---------------------------------------------------------------------
            _, predictions = torch.max(outputs, 1)

            total += tra_labels.size(0)
            correct += (predictions == tra_labels.to(device)).sum().item()
            #------------------------------------------------------------------------

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step() 
            scheduler.step_update(num_updates=num_updates)      

            epoch_tra_loss.update(total_loss.detach())
            accs.update(correct/total)
            batch_time.update(time.time() - start_t)
            
            #pbar.set_description(f"epoch {epoch + 1} iter {i}: train loss {cls_loss.item():.3f}. lr {scheduler.get_last_lr()[0]:e}")
            # Print log info     
            
            if i % 200 == 5:
                # print('======================== print results \t' + time.asctime(time.localtime(time.time())) + '=============================')
                print('Train Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Classification_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'.format(epoch, i, len(tra_dataloader),
                                                                            batch_time=batch_time,
                                                                            loss=epoch_tra_loss,
                                                                            top1=accs,))
        #logger.info(f'Total local training loss: {epoch_tra_loss.avg}, acc: {accs.avg}', main_process_only= True)
          

        scheduler.step(epoch+1)
        writer.add_scalar('clssification_train_loss', epoch_tra_loss.avg, epoch)
        writer.add_scalar('train_acc', accs.avg, epoch)


        #############################################################################

        accs = AverageMeter()
        epoch_te_loss = AverageMeter()
        accs_top5 = AverageMeter()  

        total = 0
        correct = 0
        correct_top5 = 0
        vit.eval()
        for i,(te_transformed_normalized_img, te_labels) in enumerate(te_dataloader):
            te_transformed_normalized_img = te_transformed_normalized_img.float().cuda()

            with torch.no_grad():
                outputs, pi_loss, B_loss, theta_loss, dcmm_mask = vit(te_transformed_normalized_img, noise_layer_index = opt.layer, injection=opt.inference, cp_dcmm = opt.use_cp) 
                cls_loss = loss_fn( (outputs), (te_labels.to(device)) )
       
                #acc---------------------------------------------------------------------
                _, predictions = torch.max(outputs, 1)

                total += (te_labels).size(0)
                correct += ( (predictions) == (te_labels.to(device)) ).sum().item()  #.cpu()
                #------------------------------------------------------------------------
                epoch_te_loss.update( cls_loss.detach() )
                accs.update(correct/total)  #acc = correct/total

                # Get top 5 predictions
                #_, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)

                # Check if te_labels are in the top 5 predictions
                # correct_top5 += (top5_pred == te_labels.view(-1, 1).to(device)).sum().item()
                # accs_top5.update(correct_top5/total)  #acc = correct/total
        
            # Print log info
            #print('te batch size', len(te_labels))
            if i % 100 == 0:  #opt.log_step
                # print('======================== print results \t' + time.asctime(time.localtime(time.time())) + '=============================')
                # print('Test Epoch: [{0}][{1}/{2}]\t'                 
                #     'Classification_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                #     'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                #     'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(epoch, i, len(te_dataloader),
                #                                                                 loss=epoch_te_loss,
                #                                                                 top1=accs,
                #                                                                 top5=accs_top5))
                print('Test Epoch: [{0}][{1}/{2}]\t'                 
                    'Classification_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'.format(epoch, i, len(te_dataloader),
                                                                                loss=epoch_te_loss,
                                                                                top1=accs,))
        writer.add_scalar('clssification_test_loss', epoch_te_loss.avg, epoch)
        writer.add_scalar('test_acc', accs.avg, epoch)  #add_scalar
        # writer.add_scalar('test_top5 acc', accs_top5.avg, epoch)  #add_scalar
        #logger.info(f'Total local eval loss: {epoch_te_loss.avg}, acc: {accs.avg}', main_process_only= True)              
        if( accs.avg > last_th):
            last_th = (accs.avg) #.cpu().item()
            # last_th_top5 = accs_top5.avg
            
            # torch.save(vit.state_dict(),  opt.model_saved_path + '/'+'acc_'+str(last_th) + '_top5_'+str(last_th_top5)+ '_lr_'+str(opt.lr)+'_bs_'+str(opt.batch_size)+'_layer_'+str(opt.layer)+'_'+opt.scale+'_'+str(opt.res)+'_'+str(opt.patch_size)+'_'+\
            #     opt.datasets +'_CPDCMM_'+str(opt.use_cp)+'_CPScale_'+str(opt.cp_bias_scale)+'_clusters_'+str(opt.n_clusters)+\
            #     '_lambdae_'+str(opt.lamda_e)+'_lambdas_'+str(opt.lamda_s)+'_lambdad_'+str(opt.lamda_d)+'.pkl') 
            
            torch.save(vit.state_dict(),  opt.model_saved_path + '/'+'acc_'+str(last_th) + '_top5_'+str(0)+ '_lr_'+str(opt.lr)+'_bs_'+str(opt.batch_size)+'_layer_'+str(opt.layer)+'_'+opt.scale+'_'+str(opt.res)+'_'+str(opt.patch_size)+'_'+\
                opt.datasets +'_CPDCMM_'+str(opt.use_cp)+'_CPScale_'+str(opt.cp_bias_scale)+'_clusters_'+str(opt.n_clusters)+\
                '_lambdae_'+str(opt.lamda_e)+'_lambdas_'+str(opt.lamda_s)+'_lambdad_'+str(opt.lamda_d)+'.pkl') 
            
        #print('%d epoch done' % epoch)
        #time.sleep(0.03)
    writer.close()

if __name__ == '__main__':

    set_seed(42)

    #if(not os.path.exists(save_path)):
    #    os.makedirs(save_path)

    train()

    new_acc = 0
    for file in os.listdir('./models_saved'):
        saved_acc = float(file.split('_')[1])
        saved_top5 = float(file.split('_')[3])
        saved_lr = file.split('_')[5]
        saved_bs = file.split('_')[7]
        saved_layer = file.split('_')[9]
        saved_scale = file.split('_')[10]
        saved_res = int(file.split('_')[11])
        saved_patch_size = int(file.split('_')[12])
        saved_datasets = file.split('_')[13]
        saved_cpdcmm = int((file.split('_')[15]))
        saved_cpscale = float((file.split('_')[17]))
        saved_clusters = int((file.split('_')[19]))
        saved_lambdae = float((file.split('_')[21]))
        saved_lambdas = float((file.split('_')[23]))
        saved_lambdad = float((file.split('_')[25]).split('.')[0])
        
        if( new_acc < saved_acc and str(opt.lr)==saved_lr and str(opt.layer)==saved_layer and \
            opt.scale==saved_scale and opt.res==saved_res and opt.patch_size==saved_patch_size and saved_datasets == opt.datasets and \
            saved_cpdcmm == opt.use_cp and saved_cpscale == opt.cp_bias_scale and saved_clusters == opt.n_clusters and saved_lambdae == opt.lamda_e and\
            saved_lambdas == opt.lamda_s and saved_lambdad == opt.lamda_d):
            new_acc = saved_acc

    for file in os.listdir('./models_saved'):
        saved_acc = float(file.split('_')[1])
        saved_top5 = float(file.split('_')[3])
        saved_lr = file.split('_')[5]
        saved_bs = file.split('_')[7]
        saved_layer = file.split('_')[9]
        saved_scale = file.split('_')[10]
        saved_res = int(file.split('_')[11])
        saved_patch_size = int(file.split('_')[12])
        saved_datasets = file.split('_')[13]
        saved_cpdcmm = int((file.split('_')[15]))
        saved_cpscale = float((file.split('_')[17]))
        saved_clusters = int((file.split('_')[19]))
        saved_lambdae = float((file.split('_')[21]))
        saved_lambdas = float((file.split('_')[23]))
        saved_lambdad = float((file.split('_')[25]).split('.')[0])

        if( new_acc > saved_acc and str(opt.lr)==saved_lr and str(opt.layer)==saved_layer and opt.scale==saved_scale and \
            opt.res==saved_res and opt.patch_size==saved_patch_size and saved_datasets == opt.datasets and \
            saved_cpdcmm == opt.use_cp and saved_cpscale == opt.cp_bias_scale and saved_clusters == opt.n_clusters and saved_lambdae == opt.lamda_e and\
            saved_lambdas == opt.lamda_s and saved_lambdad == opt.lamda_d):
            os.remove('./models_saved/'+file)





