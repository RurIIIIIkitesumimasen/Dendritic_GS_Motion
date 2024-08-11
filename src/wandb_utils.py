import os
import random
import pickle
import numpy as np
from tqdm import tqdm

import wandb
from wandb import log_artifact

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from utils import make_shape


def save_param_img_table(save_time, model):

    img = make_shape(model).to('cpu').detach().numpy().copy()
    img = img.repeat(30, axis=2).repeat(30, axis=3)
    img = torch.from_numpy(img)
    img_grid = torchvision.utils.make_grid(img[:90], nrow=90)
    init_images = wandb.Image(
        img_grid, caption="0:cons=0,1:direct,2:inverse,3:cons=1")
    wandb.log({f"{save_time} synaptic shape": init_images})

    W = model.state_dict()['dconvSynaps.W'].to('cpu')
    q = model.state_dict()['dconvSynaps.q'].to('cpu')

    init_weight_table = wandb.Table(
        data=list(W[0].transpose(1, 0)),
        columns=[#'OOUpLeft', 'OOUp', 'OOUpRight', 'OOLeft', 'Mid',
                 #'OORight', 'OODownLeft', 'OODown', 'OODownRight',
                 'UpLeft', 'Up', 'UpRight', 'Left', 'Center',
                 'Right', 'DownLeft', 'Down', 'DownRight']
    )
    wandb.log({f'{save_time}_weight': init_weight_table})

    init_bias_table = wandb.Table(
        data=list(q[0].transpose(1, 0)),
        columns=[#'OOUpLeft', 'OOUp', 'OOUpRight', 'OOLeft', 'Mid',
                 #'OORight', 'OODownLeft', 'OODown', 'OODownRight',
                 'UpLeft', 'Up', 'UpRight', 'Left', 'Center',
                 'Right', 'DownLeft', 'Down', 'DownRight']
    )
    wandb.log({f'{save_time}_bias': init_bias_table})

    del init_weight_table, init_bias_table, W, q

    torch.cuda.empty_cache()


# def save_model(model, config, run_wandb):
#     # encoded_path = r'./ODsave/DRGBnew-{config["runtime"]}.pth'.encode('utf-8')
#    # encoded_path = r'\DRGBall-{config["runtime"]}.pth'.encode('utf-8')
#     # torch.save(model.state_dict(), encoded_path)
#     torch.save(model.state_dict(), f'./ODsave/DRGBnew-{config["runtime"]}.pth')#{config["model"]["name"]}.pth')
#     artifact = wandb.Artifact('model', type='model')
#     artifact.add_file('./ODsave/DRGBnew-{config["runtime"]}.pth')
#     run_wandb.log_artifact(artifact)
#     os.remove(f'./model-{config["model"]["name"]}.pth')
# 保存模型的函数修改为接受epoch参数
def save_model(model, config, run_wandb, epoch):
    # 注意修改保存的文件名
    torch.save(model.state_dict(), f'./saved_model/{config["modeltype"]}-TTR{config["datarate"]}_{config["runtime"]}_{config["wandb"]["experiment_name"]}.pth')
    artifact = wandb.Artifact(f'model_epoch_{epoch}', type='model')
    artifact.add_file(f'./saved_model/{config["modeltype"]}-TTR{config["datarate"]}_{config["runtime"]}_{config["wandb"]["experiment_name"]}.pth')
    run_wandb.log_artifact(artifact)

def take_log(train_loss, train_acc, valid_loss, valid_acc, epoch):
    wandb.log({"epoch": epoch, "train_loss": train_loss,
               "test_loss": valid_loss}, step=epoch)
    wandb.log(
        {"epoch": epoch, "train_acc": train_acc,
         "test_acc": valid_acc}, step=epoch)
    print(f"{epoch} : train --- loss: {train_loss:.5f} acc: {train_acc:.5f}, test --- loss: {valid_loss:.5f} acc: {valid_acc:.5f}")


def take_detect_log(valid_acc, epoch):
    wandb.log(
        {"epoch": epoch, "test_acc": valid_acc}, step=epoch)
    print(f"test acc : {valid_acc:.5f}")
    #print(f"{epoch} : train --- loss: {train_loss:.5f} acc: {train_acc:.5f}, test --- loss: {valid_loss:.5f} acc: {valid_acc:.5f}")
