import os
import argparse

from tqdm import tqdm
import pandas as pd
import joblib
import glob

from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn

import torch.optim as optim

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from Networks.net2zhujiaxiacaiyang import MODEL as net

from losses import ssim_ir, ssim_vi,RMI_ir,RMI_vi,Hub_vi,Hub_ir
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

device = torch.device('cuda:0')
use_gpu = torch.cuda.is_available()
print(use_gpu)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='zhujiaxiacaiyangloss1', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--weight', default=[1, 1, 1, 2.5, 1, 1], type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--alpha', default=300, type=int,
                        help='number of new channel increases per depth (default: 300)')
    args = parser.parse_args()

    return args





class GetDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):

        ir = self.imageFolderDataset[index]#pet
        vi = ir.replace("SPECT","MRI")#mri

        ir = Image.open(ir).convert('L')
        vi = Image.open(vi).convert('L')

        if self.transform is not None:
            # tran = transforms.ToTensor()
            tran = transforms.Compose([transforms.ToTensor()])
            ir=tran(ir)

            vi= tran(vi)

            input = torch.cat((ir, vi), -3)


            return input, ir,vi

    def __len__(self):
        return len(self.imageFolderDataset)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader_ir,train_loader_vi, model, criterion_ssim_ir, criterion_ssim_vi,criterion_RMI_ir,criterion_RMI_vi,criterion_Hub_ir,criterion_Hub_vi,optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    losses_ssim_ir = AverageMeter()
    losses_ssim_vi = AverageMeter()
    losses_RMI_ir = AverageMeter()
    losses_RMI_vi= AverageMeter()
    losses_Hub_ir = AverageMeter()
    losses_Hub_vi = AverageMeter()
    weight = args.weight
    model.train()

    for i, (input,ir,vi)  in tqdm(enumerate(train_loader_ir), total=len(train_loader_ir)):

        if use_gpu:
            input = input.cuda()

            ir=ir.cuda()
            vi=vi.cuda()


        else:
            input = input
            ir=ir
            vi=vi

        out = model(input)


        loss_ssim_ir= weight[0] * criterion_ssim_ir(out, ir)
        loss_ssim_vi= weight[1] * criterion_ssim_vi(out, vi)
        loss_RMI_ir= weight[2] * criterion_RMI_ir(out,ir)
        loss_RMI_vi = weight[3] * criterion_RMI_vi(out,vi)
        loss_Hub_ir = weight[4] * criterion_Hub_ir(out, ir)
        loss_Hub_vi = weight[5] * criterion_Hub_vi(out, vi)
        loss = loss_ssim_ir + loss_ssim_vi+loss_RMI_ir+ loss_RMI_vi+loss_Hub_vi+loss_Hub_ir

        losses.update(loss.item(), input.size(0))
        losses_ssim_ir.update(loss_ssim_ir.item(), input.size(0))
        losses_ssim_vi.update(loss_ssim_vi.item(), input.size(0))
        losses_RMI_ir.update(loss_RMI_ir.item(), input.size(0))
        losses_RMI_vi.update(loss_RMI_vi.item(), input.size(0))
        losses_Hub_ir.update(loss_Hub_ir.item(), input.size(0))
        losses_Hub_vi.update(loss_Hub_vi.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('loss_ssim_ir', losses_ssim_ir.avg),
        ('loss_ssim_vi', losses_ssim_vi.avg),
        ('loss_RMI_ir', losses_RMI_ir.avg),
        ('loss_RMI_vi', losses_RMI_vi.avg),
        ('loss_Hub_ir', losses_Hub_ir.avg),
        ('loss_Hub_vi', losses_Hub_vi.avg),
    ])

    return log

def plot_losses(log,args):
    # Plotting loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(log['epoch'], log['loss'], label='Total Loss')
    plt.plot(log['epoch'], log['loss_ssim_ir'], label='SSIM IR Loss')
    plt.plot(log['epoch'], log['loss_ssim_vi'], label='SSIM VI Loss')
    plt.plot(log['epoch'], log['loss_RMI_ir'], label='RMI IR Loss')
    plt.plot(log['epoch'], log['loss_RMI_vi'], label='RMI VI Loss')
    plt.plot(log['epoch'], log['loss_Hub_ir'], label='Hub IR Loss')
    plt.plot(log['epoch'], log['loss_Hub_vi'], label='Hub VI Loss')

    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig('models/%s/loss_curves.png' % args.name)
    plt.show()

def main():
    args = parse_args()

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    cudnn.benchmark = True

    training_dir_ir = "./S1PECT-MRI_patches/train/SPECT"
    folder_dataset_train_ir = glob.glob(training_dir_ir + '/*.png')
    training_dir_vi = "./S1PECT-MRI_patches/train/MRI"
    folder_dataset_train_vi= glob.glob(training_dir_vi + '/*.png')

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))
                                          ])

    dataset_train_ir = GetDataset(imageFolderDataset=folder_dataset_train_ir,
                                                  transform=transform_train)
    dataset_train_vi = GetDataset(imageFolderDataset=folder_dataset_train_vi,
                                  transform=transform_train)
    train_loader_ir = DataLoader(dataset_train_ir,
                              shuffle=True,
                              batch_size=args.batch_size)
    train_loader_vi = DataLoader(dataset_train_vi,
                                 shuffle=True,
                                 batch_size=args.batch_size)
    model = net(in_channel=2)
    if use_gpu:
        model = model.cuda()
        model.cuda()

    else:
        model = model
    criterion_ssim_ir = ssim_ir
    criterion_ssim_vi = ssim_vi
    criterion_RMI_ir = RMI_ir
    criterion_RMI_vi=RMI_vi
    criterion_Hub_ir = Hub_ir
    criterion_Hub_vi = Hub_vi
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps)

    log = pd.DataFrame(index=[],
                       columns=['epoch',

                                'loss',
                                'loss_ssim_ir',
                                'loss_ssim_vi',
                                'loss_RMI_ir',
                                'loss_RMI_vi',
                                'loss_Hub_ir',
                                'loss_Hub_vi',
                                ])

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch+1, args.epochs))

        train_log = train(args, train_loader_ir,train_loader_vi, model, criterion_ssim_ir, criterion_ssim_vi,criterion_RMI_ir,criterion_RMI_vi, criterion_Hub_ir,criterion_Hub_vi,optimizer, epoch)     # 训练集

        print('loss: %.4f - loss_ssim_ir: %.4f - loss_ssim_vi: %.4f - loss_RMI_ir: %.4f - loss_RMI_vi: %.4f - loss_Hub_ir: %.4f - loss_Hub_vi: %.4f '
              % (train_log['loss'],
                 train_log['loss_ssim_ir'],
                 train_log['loss_ssim_vi'],
                 train_log['loss_RMI_ir'],
                 train_log['loss_RMI_vi'],
                 train_log['loss_Hub_ir'],
                 train_log['loss_Hub_vi'],
                 ))

        tmp = pd.Series([
            epoch + 1,

            train_log['loss'],
            train_log['loss_ssim_ir'],
            train_log['loss_ssim_vi'],
            train_log['loss_RMI_ir'],
            train_log['loss_RMI_vi'],
            train_log['loss_Hub_ir'],
            train_log['loss_Hub_vi'],

        ], index=['epoch', 'loss', 'loss_ssim_ir', 'loss_ssim_vi', 'loss_RMI_ir', 'loss_RMI_vi', 'loss_Hub_ir', 'loss_Hub_vi'])

        # log = log.append(tmp, ignore_index=True)
        log = pd.concat([log, tmp], ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), 'models/%s/model_{}.pth'.format(epoch+1) %args.name)

    plot_losses(log,args)


if __name__ == '__main__':
    main()




# True
# Config -----
# name: zhujiaxiacaiyangloss1
# epochs: 800
# batch_size: 5
# lr: 0.001
# weight: [1, 1, 1, 2.5]
# betas: (0.9, 0.999)
# eps: 1e-08
# alpha: 300
# ------------
# Epoch [1/800]
#   0%|          | 0/3264 [00:00<?, ?it/s]/home/he/cx/MATRxiugai/losses/__init__.py:444: UserWarning: torch.cholesky is deprecated in favor of torch.linalg.cholesky and will be removed in a future PyTorch release.
# L = torch.cholesky(A)
# should be replaced with
# L = torch.linalg.cholesky(A)
# and
# U = torch.cholesky(A, upper=True)
# should be replaced with
# U = torch.linalg.cholesky(A).mH
# This transform will produce equivalent results for all valid (symmetric positive definite) inputs. (Triggered internally at ../aten/src/ATen/native/BatchLinearAlgebra.cpp:1692.)
#   x = torch.cholesky(x)
# 100%|██████████| 3264/3264 [18:07<00:00,  3.00it/s]
# loss: 0.3736 - loss_ssim_ir: 0.8203 - loss_ssim_vi: 0.2031 - loss_RMI_ir: 0.4892 - loss_RMI_vi: -1.1641 - loss_Hub_ir: 0.0207 - loss_Hub_vi: 0.0043
# Epoch [2/800]
# 100%|██████████| 3264/3264 [18:08<00:00,  3.00it/s]
# loss: -0.3484 - loss_ssim_ir: 0.7667 - loss_ssim_vi: 0.0674 - loss_RMI_ir: 0.4913 - loss_RMI_vi: -1.6991 - loss_Hub_ir: 0.0236 - loss_Hub_vi: 0.0017
# Epoch [3/800]
# 100%|██████████| 3264/3264 [18:08<00:00,  3.00it/s]
# loss: -0.4947 - loss_ssim_ir: 0.7483 - loss_ssim_vi: 0.0342 - loss_RMI_ir: 0.4923 - loss_RMI_vi: -1.7954 - loss_Hub_ir: 0.0246 - loss_Hub_vi: 0.0012
#   0%|          | 0/3264 [00:00<?, ?it/s]Epoch [4/800]
# 100%|██████████| 3264/3264 [18:09<00:00,  3.00it/s]
# loss: -0.5337 - loss_ssim_ir: 0.7452 - loss_ssim_vi: 0.0277 - loss_RMI_ir: 0.4927 - loss_RMI_vi: -1.8253 - loss_Hub_ir: 0.0250 - loss_Hub_vi: 0.0010
#   0%|          | 0/3264 [00:00<?, ?it/s]Epoch [5/800]
# 100%|██████████| 3264/3264 [18:06<00:00,  3.00it/s]
# loss: -0.5708 - loss_ssim_ir: 0.7435 - loss_ssim_vi: 0.0240 - loss_RMI_ir: 0.4930 - loss_RMI_vi: -1.8574 - loss_Hub_ir: 0.0252 - loss_Hub_vi: 0.0009
#   0%|          | 0/3264 [00:00<?, ?it/s]Epoch [6/800]
# 100%|██████████| 3264/3264 [18:03<00:00,  3.01it/s]
# loss: -0.5898 - loss_ssim_ir: 0.7424 - loss_ssim_vi: 0.0220 - loss_RMI_ir: 0.4931 - loss_RMI_vi: -1.8735 - loss_Hub_ir: 0.0254 - loss_Hub_vi: 0.0008
#   0%|          | 0/3264 [00:00<?, ?it/s]Epoch [7/800]
#  22%|██▏       | 707/3264 [03:58<14:21,  2.97it/s]