from tqdm import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

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


# def perform(model, train_loader, criterion, optimizer, scheduler, device):
#     loss_total = 0
#     accuracy_total = 0
#     count = 0
#     scaler = GradScaler(enabled=True)

#     for step, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
#         onehot = torch.eye(8)[labels].to(device)
#         labels = torch.argmax(onehot, dim=1)
#         images, labels = images.to(device), labels.to(device)

#         with autocast():
#             outputs = model(images)
#             loss = criterion(outputs, labels)
# #             #添加正则化项
# #             l2_loss = 0
# #             for param in model.parameters():
# #                 l2_loss += torch.norm(param, p=2)  # L2正则化
# #             #定义超参数
# #             l2_lambda = 1e-3
# #             loss += l2_lambda * l2_loss
# # #             #print(loss)

#         with torch.no_grad():
#             accuracy = torch.mean(
#                 (torch.max(outputs, dim=1)[1] == labels).float())

#         if optimizer is not None:
#             optimizer.zero_grad()
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             scheduler.step()

#         loss_total += float(loss.detach()) * len(images)
#         accuracy_total += float(accuracy.detach()) * len(images)
#         count += len(images)

#     loss_total = loss_total / len(train_loader)

#     return loss_total / count, accuracy_total / count
