import torch.nn as nn
import torch.nn.functional as F
import logging

def loss_fn_norm(outputs, labels):
    # print(f"Output shape: {outputs.shape}")
    # print(f"Labels shape: {labels.shape}")
    return nn.CrossEntropyLoss()(outputs, labels)

def loss_fn_kd(outputs, labels, teacher_outputs):
    # hyperparameters for KD
    alphas = 0.5
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