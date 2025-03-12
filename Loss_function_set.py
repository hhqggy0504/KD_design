import torch.nn as nn
import torch.nn.functional as F
import logging

def loss_fn_norm(outputs, labels):
    # print(f"Output shape: {outputs.shape}")
    # print(f"Labels shape: {labels.shape}")
    return nn.CrossEntropyLoss()(outputs, labels)

def loss_fn_kd(outputs, labels, teacher_outputs,alphas):
    # hyperparameters for KD
 ###0.5best now
    temperatures =  4.5

    logging.info("Searching hyperparameters...")
    # logging.info("alphas: {}".format(alphas))
    # logging.info("temperatures: {}".format(temperatures))

    # for alpha in alphas:
    #     for temperature in temperatures:
    #         # [Modify] the relevant parameter in params (others remain unchanged)
    #         params.alpha = alpha
    #         params.temperature = temperature

            # # Launch job (name has to be unique)
            # job_name = "alpha_{}_Temp_{}".format(alpha, temperature)
            # launch_training_job(args.parent_dir, job_name, params)

    alpha = alphas
    T = temperatures
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


def feature_distillation_loss(teacher_feats, student_feats):
    """
    计算多层特征对齐损失
    Args:
        teacher_feats: List[Tensor] 教师模型特征列表
        student_feats: List[Tensor] 学生模型特征列表
    Returns:
        loss: Tensor 特征蒸馏总损失
    """
    total_loss = 0.0
    for t_feat, s_feat in zip(teacher_feats, student_feats):
        # 方法1：MSE损失
        total_loss += F.l1_loss(s_feat, t_feat)

    return total_loss / len(teacher_feats)  # 平均各层损失