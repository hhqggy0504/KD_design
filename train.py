import torch
import torch.nn as nn
import numpy as np
import torchvision.models._utils
from torchmetrics.functional import accuracy
import torch.optim.lr_scheduler as lr_scheduler
import random
import logging
import data_loader
# from student_net import optimizer
from tqdm import tqdm
import torchmetrics
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
# from ui
import torch.nn.functional as F
import teacher_net
import student_net
import Loss_function_set

import Design_KD_student
import Design_KD_teacher

from DCT import feature_model

# 知识蒸馏KD训练
def train_KD(student_model, teacher_model,train_loader, optimizer, epochs, train_mode,model_use,knowledge_type):
    logger.info(f"Training {train_mode} work is beginning......")
    student_model.train()

    teacher_model.load_state_dict(torch.load(f'./model_params/222{dataset_use}_best_model_resnet101_teachernet_train_params.pt'))
    teacher_model.cuda()
    teacher_model.eval()
    feature_model.cuda()
    best_acc = 0.0
    alpha_start = 0.7
    alpha_end = 0.25

    alpha_curve = 0.2 + 0.6 * np.exp(-((np.arange(200) - 100) / 30) ** 2)
    alpha_values = np.concatenate([alpha_curve, np.full(200, 0.1)])  # 300 轮 Alpha

    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0.0
        # knowledge_type = "Feature_Based"
        # knowledge_type = "Relation_Based"
        # progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")  # 进度条
        alpha = alpha_values[epoch]

        for teacher_images, student_images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, ncols=66):
            optimizer.zero_grad()
            # images ,labels = images.cuda(), labels.cuda()
            teacher_images, student_images, labels = teacher_images.cuda(), student_images.cuda(), labels.cuda()

            outputs,stu_feature1,stu_feature2,stu_feature3,stu_feature4 = student_model(student_images)

            # domain_feature=feature_model(images)
            # print(domain_feature.shape)

            with torch.no_grad():
                outputs_teacher = teacher_model(teacher_images)
            if knowledge_type == 'Response_Based':
                loss_function_kd=Loss_function_set.loss_fn_kd
                loss = loss_function_kd(outputs=outputs, labels=labels, teacher_outputs=outputs_teacher,alphas=alpha)
            elif knowledge_type == 'Feature_Based':
                # lambda_ce=0.3
                lambda_l1=0.5
                loss_function_kd=Loss_function_set.feature_distillation_loss
                l1_loss=0.0
                # stu_domain_feature=feature_model(stu_feature2)
                loss_function_kd = Loss_function_set.loss_fn_kd
                loss1 = loss_function_kd(outputs=outputs, labels=labels, teacher_outputs=outputs_teacher,alphas=alpha )

                extract_list = ["layer1"]

                extract_result = Design_KD_teacher.FeatureExtractor(teacher_model, extract_list)
                extract_result = extract_result(teacher_images)
                student_result = [stu_feature1]
                # print(extract_result(images))
                # print(domain_feature.shape)
                # print(stu_domain_feature.shape)
                # print("extract_result:::::::::::::::::::::::::::",extract_result[2].shape[1])
                # print("student_result:::::::::::::::::::::::::::::",student_result[3].shape[1])
                t_feat = extract_result[0]  # 教师模型的特征
                s_feat = student_result[0]  # 学生模型的特征

                # **1. 1×1 卷积进行通道对齐**
                conv = nn.Conv2d(in_channels=s_feat.shape[1], out_channels=t_feat.shape[1], kernel_size=1,
                                 bias=False).cuda()
                s_feat_aligned = conv(s_feat)  # 学生特征通道对齐

                # **2. AdaptiveAvgPool2d 进行空间尺寸匹配**
                adaptive_pool = nn.AdaptiveAvgPool2d((t_feat.shape[2], t_feat.shape[3]))
                s_feat_resized = adaptive_pool(s_feat_aligned)  # 学生特征尺寸匹配

                # print(s_feat_resized.shape)
                # print(t_feat.shape)

                # **3. 计算 L1 Loss**
                l1_loss = F.l1_loss(t_feat,s_feat_resized)


                loss =  loss1+l1_loss*0.3


            # print(outputs)
            # print(outputs_teacher)
            # loss = loss_function(outputs,labels)


            loss.backward()
            optimizer.step()
            running_loss += loss.item() * student_images.size(0)

            preds = torch.argmax(outputs, dim=1)
            running_acc += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader.dataset)
        # 打印结果
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        val_loss, val_acc = validate(student_model, val_loader=data_loader.load_data(type="val"))

        print(f"Val_Loss: {val_loss:.4f}, Val_Accuracy: {val_acc:.4f}")

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            # torch.save(model.state_dict(),'./model_params/best_teacher_model_params.pt')
            torch.save(student_model.state_dict(), f'./model_params/111{dataset_use}_{model_use}_{train_mode}_best_kd_params.pt')
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

        for teacher_images, student_images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False, ncols=66):
            optimizer.zero_grad()
            images, labels = teacher_images.cuda(), labels.cuda()
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
            torch.save(model.state_dict(), f'./model_params/222{dataset_use}_best_{model_use}_{train_mode}_params.pt')
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
            outputs,stu_feature1,stu_feature2,stu_feature3,stu_feature4 = model(images)
            # outputs = model(images)
            loss = Loss_function_set.loss_fn_norm(outputs, labels)
            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            running_acc += (preds == labels).sum().item()

    val_loss = running_loss / len(val_loader.dataset)
    val_acc = running_acc / len(val_loader.dataset)
    return val_loss, val_acc

##主通道入口
if __name__ == '__main__':
    # 如果GPU可用，就使用GPU，否则CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    logger.info("Loading the datasets...")

    dataset_use="MSTAR"
    # dataset_use = "FUSAR"

    # models=["model_resnet18"]
    models = ["model_resnet101"]
    # models = [ "model_Lenet5", "model_AlexNet", "model_VGG16","model_ResNet50"]# "model_Lenet5", "model_AlexNet", "model_VGG16","model_ResNet50"

    # model_use = "model_ResNet50" #  model_MyLenet5    model_AlexNet   model_VGG16    model_ResNet50
    # 选择训练模式，是知识蒸馏/教师网络训练/学生网络训练
    train_mode = "knowledge_distillation"
    # train_mode="teachernet_train"
    # train_mode="studentnet_train"

    # knowledge_type = "Response_Based"
    knowledge_type = "Feature_Based"
    # knowledge_type = "Relation_Based"

    # 设置随机数种子
    random.seed(3407)
    torch.manual_seed(3407)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(3407)
        torch.cuda.manual_seed_all(3407)




    for model_use in models:

        logger.info(f"Training {model_use} work is beginning...")

        #tensorboard监控启动
        writer = SummaryWriter(f'./tensorboard_runs/{dataset_use}_{train_mode}_{model_use}_KD_logs')

        warmup_epochs = 20  # Warm-up 轮数
        lr_init = 1e-3  # Warm-up 开始时的学习率
        lr_warmup_max = 1e-2  # Warm-up 结束时的学习率
        epochs=300
        # 计算 Warm-up 期间学习率的缩放比例
        warmup_factor = (lr_warmup_max / lr_init) - 1


        # 定义 Warm-up + Cosine Annealing 学习率策略
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # 线性 Warm-up 增长（从 1e-4 线性增长到 1e-3）
                return 1 + warmup_factor * (epoch / (warmup_epochs - 1))
            else:
                # 余弦退火（从 1e-3 逐步衰减）
                return 0.5 * (1 + np.cos((epoch - warmup_epochs) / (epochs - warmup_epochs) * np.pi))


        # 打印模型结构
        # print("Model structure:")
        # print(student_net.model)
        #
        # print(teacher_net.model)
        # Create the input data pipeline

        #训练模式动态调整
        # train(model=teacher_net.model1,train_loader=data_loader.load_data(type="train"),optimizer=teacher_net.optimizer_teacher,epochs=50)
        if train_mode == "knowledge_distillation":
            student_model = Design_KD_student.model_resnet18.to(device)
            teacher_model=Design_KD_teacher.model_resnet101.to(device)
            optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr_init)
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
            train_KD(student_model=student_model,teacher_model=teacher_model,train_loader=data_loader.load_data(type="train"),
                     optimizer=optimizer,epochs=epochs,train_mode=train_mode,model_use=model_use,knowledge_type=knowledge_type)
        elif train_mode == "studentnet_train":
            model = getattr(Design_KD_student, model_use).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # T_max 设为训练周期数
            train_norm(model=model,train_loader=data_loader.load_data(type="train"),
                       optimizer=optimizer, epochs=50,train_mode=train_mode)
            new_model=torchvision.models._utils.IntermediateLayerGetter(model,{"pre":1,"layer1_first":2,"layer1_next":3,"layer2_first":4,"layer2_next":5,"layer3_first":6,"layer3_next":7,"layer4_first":8,"layer4_next":9})
            # out = new_model(img)
            # tensor_ls = [(k, v) for k, v in out.items()]
        elif train_mode == "teachernet_train":
            model = getattr(Design_KD_teacher, model_use).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # T_max 设为训练周期数
            train_norm(model=model, train_loader=data_loader.load_data(type="train"),
                     optimizer=optimizer, epochs=50, train_mode=train_mode)
        else:
            pass


# if __name__ == "__main__":
#     train()
