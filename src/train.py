import os
import time
import random
import pickle
from typing import Dict
import numpy as np
import omegaconf
from tqdm import tqdm

import wandb
from wandb import log_artifact

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import hydra
from omegaconf import DictConfig, OmegaConf

import argparse

# 自作ネットワークの定義 net.pyから呼び出し
from net import OrientationNet
from dataset import SetData, MotionDetectionDataset
from perform import perform
from wandb_utils import save_param_img_table, save_model, take_log

import copy

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
    config_path='../config/',
    config_name='config-optuna.yaml'
)
def main(cfg):
    # device 指定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # seed 固定
    seed_everything(seed=cfg.seed)

    # dataset回りのクラス呼び出し
    setData = SetData(
        cfg.data.object_array,
        cfg.data.path,
        cfg.data.img_size,
        cfg.seed,
        cfg.data.is_noise,
        cfg.data.noise_num
    )

    # wandbを開始する設定
    if cfg.wandb.is_sweep == False:
        run_wandb = wandb.init(
            entity = cfg.wandb.entity,  # ここはチームに応じて設定
            project = cfg.wandb.project_name,
            group = cfg.wandb.group_name,
            # name=experiment_name, #experiment_nameは必要であれば追加
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True),
            save_code=cfg.wandb.is_save_code,
        )
    else:
        run_wandb = wandb.init(
            entity=cfg.wandb.entity,  # ここはチームに応じて設定
            project=f'RGB-{cfg["modeltype"]}',
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True),
        )

    cfg = wandb.config

    #学習に必要なmodel, dataloader, loss, optimizer, schedulerの定義
    train_loader = setData.set_train_data_Loader(batch_size=cfg.batch_size)
    valid_loader = setData.set_valid_data_Loader(batch_size=cfg.batch_size)

    # Define Model
    if cfg["model"]["name"] == 'Dmodel':
        model = OrientationNet(
            dendrite=cfg["model"]["dendrite"],
            init_w_mul=cfg["model"]["init_w_mul"],
            init_w_add=cfg["model"]["init_w_add"],
            init_q=cfg["model"]["init_q"],
            k=cfg["model"]["k"]
        ).to(device)

    # Param Count
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(params)  # 121898

    # loss
    print(cfg["loss"])
    if cfg["loss"] == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif cfg["loss"] == 'MSE':
        criterion = nn.MSELoss()
    else:
        raise Exception(
            f'指定されたLoss-{cfg["loss"]}-は追加されていません.このエラー文の場所に必要なModuleを追加してください.')

    # optimizer
    if cfg["optimizer"]["name"] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
    else:
        raise Exception(
            f'指定されたOptimizer-{cfg["optimizer"]["name"]}-は追加されていません.このエラー文の場所に必要なModuleを追加してください.')

    # scheduler
    if cfg["scheduler"]["name"] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, eta_min=cfg["scheduler"]["eta_min"], T_max=cfg["scheduler"]["T_max"])
    else:
        raise Exception(
            f'指定されたScheduler-{cfg["scheduler"]["name"]}-は追加されていません.このエラー文の場所に必要なModuleを追加してください.')

    # Dmodelでの学習の場合、初期形状の保存
    if cfg["model"]["name"] == 'Dmodel':
        init_model = copy.deepcopy(model)

    # 记录上次保存模型的epoch
    last_save_epoch = 0
    start_time = time.time()
    # 记录上次的训练准确率
    last_train_acc = 0

    # 存储最近5个epoch的训练准确率
    recent_train_accs = []
    
    # 学習の実行
    wandb.watch(model, criterion, log="all", log_freq=100)

    for epoch in range(cfg["epoch"]):
        model.train()
        train_loss, train_acc = perform(
            model, train_loader, criterion, optimizer, scheduler, device)

        model.eval()
        valid_loss, valid_acc = perform(
            model, valid_loader, criterion, None, scheduler, device)

        take_log(train_loss, train_acc, valid_loss, valid_acc, epoch)
        # 保存模型
        if epoch % 50 == 0:
            last_save_epoch = epoch
            save_model(model, cfg, run_wandb, epoch)
            if cfg["model"]["name"] == 'Dmodel':
                save_param_img_table('learned', model)
             
       # 检查是否超过10个小时
        elapsed_time = time.time() - start_time
        if elapsed_time > 8 * 60 * 60:  # 8h over
            print("Training time exceeded 8 hours. Forcefully terminating.")
            break
            
            
        # 检查是否满足停止条件
        recent_train_accs.append(train_acc)
        if len(recent_train_accs) >= 5:
            recent_std = np.std(recent_train_accs[-5:])
            if train_acc > 0.9999 and recent_std < 0.00001:
                print(f"Training stopped with stable Acc. Train accuracy: {train_acc:.5f}, Recent std: {recent_std:.5f}")
                break
                
        if len(recent_train_accs) >= 5:
            recent_std = np.std(recent_train_accs[-5:])
            if train_acc < 0.1000 and recent_std < 0.1:
                print(f"Training stopped with very low Acc. Train accuracy: {train_acc:.5f}, Recent std: {recent_std:.5f}")
                break

        # 更新上次的训练准确率
        last_train_acc = train_acc

    # Dmodelでの学習の場合、学習後形状の保存
    if cfg["model"]["name"] == 'Dmodel':
        save_param_img_table('learned', model)
        save_param_img_table('init', init_model)

    # Modelの保存
    save_model(model, cfg, run_wandb, "final")

    wandb.finish()
    return valid_loss


if __name__ == "__main__":
    wandb.finish()
    main()
