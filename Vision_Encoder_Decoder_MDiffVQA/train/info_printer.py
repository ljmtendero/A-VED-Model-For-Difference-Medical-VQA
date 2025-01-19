import torch
import torchvision

from multiprocessing import cpu_count
from datetime import datetime


def print_soft_hard_info() -> str:
    info =  '='*10 + ' Software and Hardware Information ' + '='*10 + '\n' + \
            ' '*4 + 'PyTorch Version: ' + torch.__version__ + '\n' + \
            ' '*4 + 'Torchvision Version: ' + torchvision.__version__ + '\n' + \
            ' '*4 + 'CPU Count: ' + str(cpu_count()) + '\n' + \
            ' '*4 + 'CUDA Available: ' + str(torch.cuda.is_available()) + '\n'
    
    if torch.cuda.is_available():
        info += ' '*8 + 'Device Count: ' + str(torch.cuda.device_count()) + '\n' + \
                ' '*8 + 'Device Name: ' + torch.cuda.get_device_name(0) + '\n'
    info += '='*55 + '\n'

    return info


def print_train_info(batch_size, num_epochs, num_workers) -> str:
    info =  '='*10 + ' Training Information ' + '='*10 + '\n' + \
            ' '*4 + 'Batch Size: ' + str(batch_size) + '\n' + \
            ' '*4 + 'Number of Epochs: ' + str(num_epochs) + '\n' + \
            ' '*4 + 'Number of Workers: ' + str(num_workers) + '\n' + \
            '='*42 + '\n'
    
    return info


def print_model_info(model) -> str:
    info =  '='*10 + ' Model Information ' + '='*10 + '\n' + \
            ' '*4 + 'Model Name: ' + model.__class__.__name__ + '\n' + \
            ' '*4 + 'Number of Trainable Parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)) + '\n' + \
            ' '*4 + 'Number of Non-Trainable Parameters: ' + str(sum(p.numel() for p in model.parameters() if not p.requires_grad)) + '\n' + \
            ' '*4 + 'Number of Total Parameters: ' + str(sum(p.numel() for p in model.parameters())) + '\n' + \
            ' '*8 + '(Encoder) Number of Parameters: ' + str(sum(p.numel() for p in model.encoder.parameters())) + '\n' + \
            ' '*8 + '(Decoder) Number of Parameters: ' + str(sum(p.numel() for p in model.decoder.parameters())) + '\n\n' + \
            ' '*4 + 'Processor Type: ' + model.processor_name + '\n' + \
            ' '*4 + 'Encoder Architecture: ' + model.encoder_name + '\n\n' + \
            ' '*4 + 'Decoder Architecture: ' + model.decoder.__class__.__name__ + '\n\n' + \
            ' '*4 + str(model.decoder.config) + '\n' + \
            '='*50 + '\n'
    
    return info


def print_hyperparams_info(loss_func, optimizer, lr_scheduler) -> str:
    info =  '='*10 + ' Hyperparameters Information ' + '='*10 + '\n' + \
            ' '*4 + 'Loss Function: ' + loss_func.__class__.__name__ + '\n' + \
            ' '*4 + 'Optimizer: ' + optimizer.__class__.__name__ + '\n' + \
            ' '*8 + '\n'.join([f'{" "*8}{k}: {v}' for k, v in optimizer.defaults.items()]) + '\n' + \
            ' '*4 + 'Learning Rate Scheduler: ' + lr_scheduler.__class__.__name__ + '\n' + \
            ' '*8 + '\n'.join([f'{" "*8}{k}: {v}' for k, v in lr_scheduler.__dict__.items() if k != 'optimizer' and not k.startswith('_')]) + '\n' + \
            '='*49 + '\n'
    
    return info