import torch
import torch.nn as nn
import numpy as np
from torchmetrics.functional import accuracy

import random
import logging
import data_loader
from student_net import optimizer
from tqdm import tqdm
import torchmetrics
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
# from ui

import teacher_net
import student_net
import Loss_function_set



# 知识蒸馏KD训练
def train_KD(model, train_loader, optimizer, epochs, train_mode):
    logger.info(f"Training {train_mode} work is beginning......")
    model.cuda()
    model.train()

    teacher_model=teacher_net.model1
    teacher_model.cuda()
    teacher_model.eval()

    best_acc = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0.0

        # progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")  # 进度条

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, ncols=66):
            optimizer.zero_grad()
            images ,labels = images.cuda(), labels.cuda()
            outputs = model(images)

            with torch.no_grad():
                outputs_teacher = teacher_model(images)

            loss_function_kd=Loss_function_set.loss_fn_kd

            # loss = loss_function(outputs,labels)
            loss = loss_function_kd(outputs=outputs, labels=labels,teacher_outputs=outputs_teacher)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            running_acc += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader.dataset)
        # 打印结果
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        val_loss, val_acc = validate(model, val_loader=data_loader.load_data(type="val"))

        print(f"Val_Loss: {val_loss:.4f}, Val_Accuracy: {val_acc:.4f}")

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            # torch.save(model.state_dict(),'./model_params/best_teacher_model_params.pt')
            torch.save(model.state_dict(), './model_params/best_kd_params.pt')
            logger.info(f"Best model renewed! Validation accuracy: {val_acc:.4f}")


    writer.close()

# 普通单个网络训练
def train_norm(model, train_loader, optimizer, epochs, train_mode):
    logger.info(f"Training {train_mode} work is beginning......")
    # model.cuda()
    model.train()

    best_acc = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0.0

        # progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")  # 进度条

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False, ncols=66):
            optimizer.zero_grad()
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)

            # loss = loss_function(outputs,labels)
            loss = Loss_function_set.loss_fn_norm(outputs=outputs, labels=labels)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("Loss is NaN or Inf! Skipping this batch.")
                continue
            # print(f"Loss: {loss.item()}")  # 观察是否为 NaN 或 Inf

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            running_acc += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader.dataset)
        # 打印结果
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        val_loss, val_acc = validate(model, val_loader=data_loader.load_data(type="val"))

        print(f"Val_Loss: {val_loss:.4f}, Val_Accuracy: {val_acc:.4f}")

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            # torch.save(model.state_dict(),'./model_params/best_teacher_model_params.pt')
            torch.save(model.state_dict(), f'./model_params/best_{train_mode}_params.pt')
            logger.info(f"Best model renewed! Validation accuracy: {val_acc:.4f}")

    writer.close()

def validate(model, val_loader):
    """
    Perform validation on the validation set.
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = Loss_function_set.loss_fn_norm(outputs, labels)
            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            running_acc += (preds == labels).sum().item()

    val_loss = running_loss / len(val_loader.dataset)
    val_acc = running_acc / len(val_loader.dataset)
    return val_loss, val_acc

##主通道入口
if __name__ == '__main__':
    # 如果GPU可用，就使用GPU，否则C    PU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    logger.info("Loading the datasets...")
    models = [ "model_ResNet50"]

    # model_use = "model_ResNet50" #  model_MyLenet5    model_AlexNet   model_VGG16    model_ResNet50

    #选择训练模式，是知识蒸馏/教师网络训练/学生网络训练
    # train_mode="knowledge_distillation"
    # train_mode="teachernet_train"
    train_mode="studentnet_train"

    for model_use in models:

        logger.info(f"Training {model_use} work is beginning...")

        #tensorboard监控启动
        writer = SummaryWriter(f'./tensorboard_runs/FUSAR_{train_mode}_{model_use}_logs')

        #设置随机数种子
        random.seed(230)

        # 打印模型结构
        # print("Model structure:")
        # print(student_net.model)
        #
        # print(teacher_net.model)
        # Create the input data pipeline

        #训练模式动态调整
        # train(model=teacher_net.model1,train_loader=data_loader.load_data(type="train"),optimizer=teacher_net.optimizer_teacher,epochs=50)
        if train_mode == "knowledge_distillation":
            train_KD(model=student_net.model2,train_loader=data_loader.load_data(type="train"),optimizer=student_net.optimizer_stu,epochs=50)
        elif train_mode == "studentnet_train":
            train_norm(model=getattr(student_net,model_use).to(device),train_loader=data_loader.load_data(type="train"),
                       optimizer=torch.optim.Adam((getattr(student_net,model_use).parameters()), lr=1e-3), epochs=50,train_mode=train_mode)
        elif train_mode == "teachernet_train":
            train_norm(model=getattr(teacher_net,model_use).to(device),train_loader=data_loader.load_data(type="train"),
                       optimizer=torch.optim.Adam((getattr(teacher_net,model_use).parameters()), lr=1e-4), epochs=50,train_mode=train_mode)
        else:
            pass


# if __name__ == "__main__":
#     train()
