# 検証用コード

import os
import random
import numpy as np
import omegaconf
from tqdm import tqdm
import timm

import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig, OmegaConf

# 自作ネットワークの定義
from utils import make_shape
from perform import perform
from testdataset import SetData, MotionDetectionDataset
from cnn import FourLayerCNN, LeNet, CustomEfficientNet, EfficientNetB0
from wandb_CNNs import save_model, take_log
from ResNet import resnet18, resnet34, resnet50, resnet101, resnet152, resnet50_cam
from ConvNeXt import convnext_tiny, ConvNeXtLee
from net import OrientationNet
from wandb_utils import take_detect_log
#from utils.set_func import *


def perform(model, train_loader, criterion, optimizer, scheduler, device):
    loss_total = 0
    accuracy_total = 0
    count = 0
    scaler = GradScaler(enabled=True)

    for step, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images, labels = images.to(device), labels.to(device)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            #添加正则化项
#             l2_loss = 0
#             for param in model.parameters():
#                 l2_loss += torch.norm(param, p=2)  # L2正则化
#             #定义超参数
#             l2_lambda = 1e-3
#             loss += l2_lambda * l2_loss

        with torch.no_grad():
            accuracy = torch.mean(
                (torch.max(outputs, dim=1)[1] == labels).float())

        if optimizer is not None:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        loss_total += loss.item() * len(images)
        accuracy_total += accuracy.item() * len(images)
        count += len(images)

    loss_total /= count
    accuracy_total /= count

    return loss_total, accuracy_total


# -----------------------------------------------------------
'''Seed setting'''


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.torch.backends.cudnn.benchmark = False
    torch.torch.backends.cudnn.enabled = True
    # 決定論的アルゴリズムを使用する
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
# -----------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path='../config',
    config_name='config-optuna.yaml'
)
def main(cfg):
    # GPU指定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # SEED値固定
    seed_everything(seed=cfg.model_num_seed)

    # DataSetの設定
    setData = SetData(
        cfg.data.object_array,
        cfg.data.path,
        cfg.data.img_size,
        cfg.seed,
        cfg.data.is_noise,
        cfg.data.noise_num
    )

    #experiment_name = f'EfN-{cfg["runtime"]}'                                         #run detect revise

    # wandbを開始する設定

    run_wandb = wandb.init(
        project = cfg.wandb.project_name,#f'RGB-{cfg["modeltype"]}',                                                              #run detect revise
        group = cfg.wandb.group_name,#'noise_no_connection(noise_10per)',                                                                      #run detect revise
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True),
    )

    cfg = wandb.config

    #学習に必要なmodel, dataloader, loss, optimizer, schedulerの定義
    # 検知だけなのでvalid_loaderでok
    valid_loader = setData.set_valid_data_Loader(
        batch_size=cfg.batch_size,
        #model=cfg.model,
        #img_size=cfg.data.img_size
    )

    # Modelを作成
    if cfg["modeltype"] == 'AVS':
        model = OrientationNet().to(device)
        
    elif cfg["modeltype"] == 'LeNet':
        model = LeNet().to(device)
        
    elif cfg["modeltype"] == '4LCNN':
        model = FourLayerCNN().to(device)
        
    elif cfg["modeltype"] == 'EfNB0':
        model = EfficientNetB0(num_classes = 8).to(device)
        
    elif cfg["modeltype"] == 'ResNet':
        model = resnet50_cam(num_classes = 8).to(device)
     
    elif cfg["modeltype"] == 'ConvNeXt':
        model = convnext_tiny().to(device)

    # model load
    model.load_state_dict(torch.load(f'./saved_model_GrayScale/{cfg["modeltype"]}/{cfg["modeltype"]}-TTR{cfg["datarate"]}_{cfg["runtime"]}_{cfg["wandb"]["experiment_name"]}.pth'))
    # loss
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                              optimizer, eta_min=cfg["scheduler"]["name"], T_max=cfg["scheduler"]["T_max"])


    # 検証の実行
    wandb.watch(model, criterion, log="all", log_freq=100)

    model.eval()
    valid_loss, valid_acc = perform(
        model, valid_loader, criterion, None, scheduler, device)

    take_detect_log(valid_acc, 5)
    print(f"test acc : {valid_acc:.5f}")

    wandb.finish()
    return valid_loss


if __name__ == "__main__":
    wandb.finish()
    main()
