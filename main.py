import os
import glob
import datetime
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.utils.tensorboard import SummaryWriter

from losses import dice
from utils import train, losses, metrics, augmentation, functional
from encoder import _preprocessing as F
from sklearn.model_selection import train_test_split
from model import PSPNet
from dataset.dataset import MyDataset

def adam_optimizer(model, lr, weight_decay):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        apply_weight_decay = weight_decay
        if 'bias' in key or 'bn' in key:
            apply_weight_decay = 0
            print('set weight decay=0 for {}'.format(key))
        params += [{'params': [value], 'weight_decay': apply_weight_decay}]
    optimizer = torch.optim.Adam(params, lr)
    return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
    コマンドライン引数
    """
    parser.add_argument('-i', '--images',  default= './data/CamVid/images',
                        help='image directory')
    parser.add_argument('-m', '--masks', default= './data/CamVid/masks',
                        help='masks directory')
    parser.add_argument('-o', '--output', default='./models/mobilev1.pth',
                        help='model output directory')
    parser.add_argument('-rt', '--resume_train', default=None,
                        help='model to resume training')
    parser.add_argument('-lr', '--learning_rate', type=float,
                        default=0.001, help='learning rate (default=0.001)')
    parser.add_argument('-ss', '--step_size', type=int,
                        default=22, help='lr step size (default=25)')
    parser.add_argument('-e', '--epoch', type=int, default=100,
                        help='training epochs (default=100)')
    parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help='training batch size (default=8)')

    FLAGS = parser.parse_args()

    device = 'cuda'
    train_dir = os.path.join(FLAGS.images, 'train')
    valid_dir = os.path.join(FLAGS.images, 'valid')
    tr_masks_dir = os.path.join(FLAGS.masks, 'train')
    va_masks_dir = os.path.join(FLAGS.masks, 'valid')

    # Class labels for cityscape dataset
    CLASSES = ['car']
#     CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
#                'tree', 'signsymbol', 'fence', 'car',
#                'pedestrian', 'bicyclist', 'unlabelled']
    
    # Load model
    if FLAGS.resume_train: 
        model = torch.load(FLAGS.resume_train)
    else:
        model = PSPNet(
            preupsample=False,
            upsampling=8,
            encoder_name="mobilenetv1",
            encoder_weights=False,
            encoder_depth=5,
            psp_out_channels=512,              # PSP out channels after concat not yet final
            psp_use_batchnorm=True,
            psp_dropout=0.2,
            in_channels=3,
            classes=len(CLASSES),
            activation='sigmoid',     # Optional[Union[str, callable]]
            dilated=False,
#             aux_params={'classes': 1, 'height': 320,
#                         'width': 320, 'dropout': 0.2},  # Opt
        )
    
    # Define parameters
    loss = losses.DiceLoss()
    metrics = [metrics.IoU(threshold=0.5),
               metrics.Accuracy(),
               metrics.Recall()]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=FLAGS.learning_rate),
    ])

    # optimizer = adam_optimizer(model, FLAGS.learning_rate, weight_decay=1e-4)

    lr_step_size = FLAGS.step_size
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.6)
    # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=FLAGS.epoch * len(self.all_img_names) // FLAGS.batch_size)

    # Create augmentation and preprocessing pipelines
    pl_aug = augmentation.get_training_augmentation()
    pl_prepro = F.get_preprocessing(F.normalize_input)

    # Create Data Loaders
    train_dataset = MyDataset(train_dir, tr_masks_dir, classes=CLASSES, augmentation=pl_aug, preprocessing=pl_prepro)
    valid_dataset = MyDataset(valid_dir, va_masks_dir, classes=CLASSES, augmentation=pl_aug, preprocessing=pl_prepro)

    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)


    # Load model and data
    train_epoch = train.TrainEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )
    valid_epoch = train.ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    # Run training
    max_score = 0
    lr_history = []
    today = datetime.datetime.today().strftime('%y%m%d')
    
    num_logs = len(glob.glob(f'./logs/{today}_*'))
    writer = SummaryWriter(log_dir=f'./logs/{today}_{num_logs+1}')
    for e in range(FLAGS.epoch):
        functional.adjust_learning_rate(e, train_epoch.optimizer, lr_step_size)

        print('\nEpoch: {}'.format(e))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # Update lr
        lr_history.append(train_epoch.optimizer.param_groups[0]['lr'])
        print('lr: {}'.format(lr_history[-1]))
        scheduler.step()
        
        # Tensorboard
        for metric in train_logs:
            writer.add_scalar('train/'+metric, train_logs[metric], e)
            writer.add_scalar('valid/'+metric, valid_logs[metric], e)
        writer.add_scalar('lr/lr', lr_history[-1], e)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, FLAGS.output)
            print('Model saved!')
    
    writer.close()
