import os, sys
import torch
# torch.set_default_tensor_type(torch.DoubleTensor)

import numpy as np
import scipy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import wandb
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim as optim
from scipy.sparse.linalg import spsolve
import torchvision
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import networks
import forwardOps
import argparse
from datetime import datetime
import medmnist
from medmnist import OrganCMNIST
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
time_ = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
parser = argparse.ArgumentParser()
parser.add_argument('--cluster', type=int, default=0)
parser.add_argument('--datapath', type=str, default='/amedmnist')
parser.add_argument('--savepath', type=str, default='/checkpoints')
parser.add_argument('--regnet', type=str, default='pgd')  # can be 'LA', 'resnet', 'unet' or 'pgd'
parser.add_argument('--channels', type=int, default=64)
parser.add_argument('--layers', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--task', type=str, default='radon')  # can be deblur, radon, mask
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-5)
parser.add_argument('--cglsIter', type=int, default=16)
parser.add_argument('--solveIter', type=int, default=1)

args = parser.parse_args()
exp_name = args.task + '_stl10_' + args.regnet + '_layers_' + str(args.layers) + '_chan_' + str(
    args.channels) + '_cglsIter_' + str(
    args.cglsIter) + '_netIter_' + str(args.solveIter) + time_
wandb.init(config=args, entity='username', name=exp_name)
config = wandb.config

if not os.path.exists(args.savepath):
    os.mkdir(args.savepath)

## Random seed:###
import random
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

#####

path = args.datapath
num_workers = 6 if args.cluster else 0
train_data = OrganCMNIST(root=path, split="train", download=True, transform=torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Resize(size=96)]))
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
test_data = OrganCMNIST(root=path, split="train", download=True, transform=torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Resize(size=96)]))
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=num_workers)

learnEmb = False
if args.regnet == 'resnet':
    learnEmb = True
if args.regnet == 'resnetFO':
    learnEmb = True
elif args.regnet == 'unet':
    learnEmb = False
elif args.regnet == 'LA':
    learnEmb = True
elif args.regnet == 'hyperunet':
    learnEmb = True
elif args.regnet == 'pgd':
    learnEmb = False

if args.task == 'deblur':
    forOp = forwardOps.blurFFT(args.channels, 3, learnEmb=learnEmb, dim=96, device='cuda')
elif args.task == 'radon':
    forOp = forwardOps.radonTransform(args.channels, 3, learnEmb=learnEmb, device='cuda')
elif args.task == 'mask':
    ind = torch.randperm(96 ** 2, device=device)
    ind = ind[:1000]
    forOp = forwardOps.maskImage(ind, imsize=[96, 96], embdsize=args.channels, learnEmb=learnEmb)

if args.regnet == 'resnet':
    regNet = networks.resNet(num_layers=args.layers, nopen=args.channels)
    dataProj = networks.CGLS(forOp, CGLSit=args.cglsIter, eps=1e-2)
if args.regnet == 'resnetFO':
    args.cglsIter = 1
    regNet = networks.resNetFO(num_layers=args.layers, nopen=args.channels)
    dataProj = networks.CGLS(forOp, CGLSit=1, eps=1e-2)
elif args.regnet == 'unet':
    args.cglsIter = 1
    regNet = networks.UNet(channels=[32, 64, 128, 256], embed_dim=256, ncIn=3)
    dataProj = networks.CGLS(forOp, CGLSit=1, eps=1e-2)
elif args.regnet == 'LA':
    regNet = networks.leastActionNet(nlayers=args.layers, nchanels=args.channels, nfixPointIter=2, imsize=[96, 96])
    dataProj = networks.CGLS(forOp, CGLSit=args.cglsIter, eps=1e-2)
elif args.regnet == 'hyperunet':
    regNet = networks.hyperUNet(args.channels)
    dataProj = networks.CGLS(forOp, CGLSit=args.cglsIter, eps=1e-2)

if args.regnet == 'pgd':
    args.cglsIter = 1
    forOp = forwardOps.radonTransform(3, 3, learnEmb=learnEmb, device='cuda')
    # forOp = forwardOps.blurFFT(3, 3, learnEmb=False, dim=96, device='cuda')
    regNet = networks.resNetFO(num_layers=args.layers, nopen=args.channels, embed_proj=True)
    dataProj = networks.CGLS(forOp, CGLSit=1, eps=1e-2)
    net = (networks.neuralProximalGradient(regNet, dataProj, forOp, niter=args.solveIter))
else:
    net = (networks.inerseSolveNet(regNet, dataProj, forOp, niter=args.solveIter))

if args.cluster:
    if not os.path.exists(args.savepath):
        os.mkdir(args.savepath)
    log_filename = os.path.join(args.savepath, exp_name + '.txt')
    sys.stdout = open(log_filename, "w")
    sys.stderr = sys.stdout

    print(
        "**********************************************************************************")
    file2Open = "inverseProblemsTests.py"
    print("DRIVER CODE:")
    f = open(file2Open, "r")
    for line in f:
        print(line, end='', flush=True)

print("exp name=", exp_name)
print("args:", args, flush=True)

net.to(device)

optimizer = optim.Adam([{'params': net.parameters()}], lr=args.lr, weight_decay=args.wd)
epochs = 1000
hist = torch.zeros(epochs, device=device)


def eval_test(alpha=0):
    print(':::::::::::::::::Started TEST:::::::::::::::::')
    net.eval()
    avhist = 0
    fithist = 0
    epss = 0
    niters = max(net.dataProj.nCGLSiter, net.niter)
    with torch.no_grad():
        for batch_idx, (img, lbl) in enumerate(test_loader):
            img = img.to(device)
            img = torch.cat([img, img, img], dim=1)
            ftrue = img
            dtrue = net.forOp(ftrue, emb=False)
            noise = torch.randn_like(dtrue)
            Dn = dtrue + alpha * (torch.mean(torch.abs(dtrue)) / 100 * noise)
            eps = ((Dn - dtrue).norm() / Dn.norm())
            net.dataProj.eps = 1.2 * eps
            X, Xref, R = net(Dn)
            Dp = net.forOp(X, emb=False)

            lossX = F.mse_loss(X, ftrue) / F.mse_loss(torch.zeros_like(ftrue), ftrue)
            lossD = F.mse_loss(Dp, Dn) / F.mse_loss(torch.zeros_like(Dn), Dn)
            loss = lossX + lossD
            epss += eps
            avhist += torch.sqrt(loss)
            fithist += torch.sqrt(lossD)

        print('TEST loss = %3.2e    datafit = %3.2e   avgEps = %3.2e' % (
            avhist / (batch_idx), fithist / (batch_idx), epss / (batch_idx)))
    net.train()
    print(':::::::::::::::::Finished TEST:::::::::::::::::')
    net.dataProj.nCGLSiter = 1  # args.cglsIter
    return (avhist / (batch_idx)), fithist / (batch_idx)


for j in range(epochs):
    avhist = 0
    fithist = 0
    avXrefLoss = 0
    avXloss = 0
    avDloss = 0
    avEps = 0
    cnt = 0
    for batch_idx, (img, lbl) in enumerate(train_loader):
        optimizer.zero_grad()
        img = img.to(device)
        img = torch.cat([img, img, img], dim=1)
        ftrue = img
        dtrue = net.forOp(ftrue, emb=False)
        noise = torch.randn_like(dtrue)
        alpha = 10 + 5 * torch.rand((1)).to(device)
        Dn = dtrue + alpha * (torch.mean(torch.abs(dtrue)) / 100 * noise)
        eps = ((Dn - dtrue).norm() / Dn.norm())
        net.dataProj.eps = 1.2 * eps
        X, Xref, R = net(Dn)
        Dp = net.forOp(X, emb=False)

        lossX = F.mse_loss(X, ftrue) / F.mse_loss(torch.zeros_like(ftrue), ftrue)
        lossD = F.mse_loss(Dp, dtrue) / F.mse_loss(torch.zeros_like(dtrue), dtrue)  # optimize for this objective
        lossXref = F.mse_loss(Xref, ftrue) / F.mse_loss(torch.zeros_like(ftrue), ftrue)
        train_residual = torch.sqrt(F.mse_loss(Dp, Dn) / F.mse_loss(torch.zeros_like(Dn), Dn))

        loss = lossX + lossD + lossXref

        avDloss += lossD.item()
        avXloss += lossX.item()
        avXrefLoss += lossXref.item()
        avEps += eps.item()
        loss.backward()
        optimizer.step()
        avhist += torch.sqrt(loss)
        fithist += torch.sqrt(lossD)
        if batch_idx % 10 == 9:
            print('epoch = %3d   batch_idx = %d    loss = %3.2e    lossD = %3.2e    datafit= %3.2e    eps = %3.2e' % (
                j, batch_idx, loss, lossD, train_residual, eps),
                  flush=True)

    # scheduler.step()
    hist[j] = avhist / (batch_idx)
    print('==== epoch = %3d    aveloss = %3.2e    avefit = %3.2e' % (
        j, avhist / (batch_idx), fithist / (batch_idx)), flush=True)
    print(' ')

    torch.save({'epoch': j,
                'net_dict': net.state_dict(),
                'error_fit': avhist / (batch_idx),
                'data_fit': fithist / (batch_idx),
                'optimizer_dict': optimizer.state_dict(),
                'regNet_dict': regNet.state_dict(),
                'dataProj_dict': dataProj.state_dict()}, os.path.join(args.savepath, exp_name + '.pth'))

    test_error = 0
    test_residual = 0
    metrics = {
        "train_error": avhist / (batch_idx),
        "train_residual": fithist / (batch_idx),
        "test_error": test_error,
        "test_residual": test_residual,
        "avDloss": avDloss / (batch_idx),
        "avXloss": avXloss / (batch_idx),
        "avXrefloss": avXrefLoss / (batch_idx),
        "avEps": avEps / (batch_idx),
    }
    wandb.log(metrics)

print('done')
