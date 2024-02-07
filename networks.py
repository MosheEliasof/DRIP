import os, sys
import torch
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim as optim
from scipy.sparse.linalg import spsolve

import torchvision
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt


def project_to_data(forProb, b, x0, niter=3, tol=1e-3):
    # iterate for the solution min \|b-Ax\|^2
    # starting from x0
    x = x0.clone()
    r = b - forProb(x)
    for i in range(niter):
        g = forProb.adjoint(r)
        Ag = forProb(g)
        delta = torch.norm(Ag) ** 2
        gamma = torch.norm(g) ** 2
        alpha = gamma / delta

        x = x + alpha * g
        r = r - alpha * Ag

        if torch.norm(r) / torch.norm(b) < tol:
            return x, r

    return x, r


class CGLS(nn.Module):
    def __init__(self, forOp, CGLSit=100, eps=1e-2, device='cuda'):
        super(CGLS, self).__init__()
        self.forOp = forOp
        self.nCGLSiter = CGLSit
        self.eps = eps

    def forward_landweber(self, b, xref, zref=[]):
        x = xref

        r = b - self.forOp(x)
        if r.norm() / b.norm() < self.eps:
            return x, r
        s = self.forOp.adjoint(r)
        for k in range(self.nCGLSiter):
            g = self.forOp.adjoint(r)
            Ag = self.forOp(g)
            delta = torch.norm(Ag) ** 2
            gamma = torch.norm(g) ** 2
            alpha = gamma / delta

            x = x + alpha * g
            r = r - alpha * Ag

            if torch.norm(r) / torch.norm(b) < self.eps:
                return x, r

        return x, r

    def forward(self, b, xref, zref=[]):
        x = xref

        r = b - self.forOp(x)
        if r.norm() / b.norm() < self.eps:
            return x, r
        s = self.forOp.adjoint(r)

        # Initialize
        p = s
        norms0 = torch.norm(s)
        gamma = norms0 ** 2

        for k in range(self.nCGLSiter):
            q = self.forOp(p)
            delta = torch.norm(q) ** 2
            alpha = gamma / delta

            x = x + alpha * p
            r = r - alpha * q

            # print(k, r.norm().item() / b.norm().item())
            if r.norm() / b.norm() < self.eps:
                return x, r

            s = self.forOp.adjoint(r)

            norms = torch.norm(s)
            gamma1 = gamma
            gamma = norms ** 2
            beta = gamma / gamma1
            p = s + beta * p

        return x, r


class resNet(nn.Module):
    def __init__(self, num_layers, nopen):
        super(resNet, self).__init__()

        self.h = 0.1
        self.num_layers = num_layers

        self.Kin = nn.Parameter(nn.init.xavier_uniform_(torch.empty(nopen, nopen, 7, 7)))
        self.K = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_layers, nopen, nopen, 3, 3)))
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            self.bns.append(nn.BatchNorm2d(nopen))

    def forward(self, Z, Zall=[]):
        Zold = Z.clone()
        Z = F.conv2d(Z, self.Kin, padding=self.Kin.shape[-1] // 2)

        for i in range(self.num_layers):
            dZ = F.conv2d(Z, self.K[i], padding=self.K[i].shape[-1] // 2)
            dZ = F.instance_norm(dZ)
            # dZ = self.bns[i](dZ)
            dZ = F.leaky_relu(dZ, negative_slope=0.2)
            dZ = F.conv_transpose2d(dZ, self.K[i], padding=self.K.shape[-1] // 2)

            tmp = Z.clone()
            Z = 2 * Z - Zold - 1 * dZ
            Zold = tmp

        # close
        return Z, Z


class resNetFO(nn.Module):
    def __init__(self, num_layers, nopen, embed_proj=False):
        super(resNetFO, self).__init__()
        self.embed_proj = embed_proj
        if self.embed_proj:
            # self.embed = torch.nn.Conv2d(in_channels=3, out_channels=nopen, kernel_size=3, padding=1)
            # self.proj = torch.nn.Conv2d(in_channels=nopen, out_channels=3, kernel_size=1, padding=0)
            self.proj = Embed(embdsize=nopen, nin=3)
            self.embed = Embed(embdsize=3, nin=nopen)

            self.bn_embed = torch.nn.BatchNorm2d(nopen)

        self.h = 0.1
        self.num_layers = num_layers

        self.K = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_layers, nopen, nopen, 3, 3)))
        self.K2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_layers, nopen, nopen, 3, 3)))

    def forward(self, Z, Zall=[]):
        if self.embed_proj:
            Z = self.embed(Z)
        for i in range(self.num_layers):
            dZ = F.conv2d(Z, self.K[i], padding=self.K[i].shape[-1] // 2)
            dZ = F.instance_norm(dZ)
            dZ = F.leaky_relu(dZ, negative_slope=0.2)
            if self.embed_proj:
                dZ = F.conv2d(dZ, self.K2[i], padding=self.K[i].shape[-1] // 2)
            Z = Z + dZ
        # close
        if self.embed_proj:
            Z = self.proj(Z)
        return Z, Z


class leastActionNet(nn.Module):
    def __init__(self, nlayers, nchanels, nfixPointIter, imsize):
        super(leastActionNet, self).__init__()

        self.K = nn.Parameter(nn.init.xavier_uniform_(torch.empty(nlayers, nchanels, nchanels, 3, 3)))
        self.nlayers = nlayers
        self.nfixPointIter = nfixPointIter
        self.X0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, nchanels, imsize[0], imsize[1])))

    def layer(self, Z, K):
        dZ = F.conv2d(Z, K, padding=K.shape[-1] // 2)
        dZ = F.instance_norm(dZ)
        dZ = F.leaky_relu(dZ, negative_slope=0.2)
        dZ = F.conv_transpose2d(dZ, K, padding=K.shape[-1] // 2)
        return dZ

    def getNoNlinRHS(self, Z, XN):
        Y = torch.zeros_like(Z)
        for i in range(Y.shape[0]):
            Y[i] = -self.layer(Z[i].clone(), self.K[i])
        Y[-1] = Y[-1].clone() + XN
        Y[0] = Y[0].clone() + self.X0

        return Y

    def triDiagSolve(self, Z):
        # forward pass
        nlayers = Z.shape[0]
        Y = torch.zeros_like(Z)
        Y[0] = np.sqrt(1 / 2.0) * Z[0].clone()
        for i in range(1, nlayers):
            a = np.sqrt((i + 1) / (i + 2))
            b = np.sqrt((i) / (i + 1))
            Y[i] = a * (b * Y[i - 1].clone() + Z[i].clone())
        # backward pass
        W = torch.zeros_like(Z)
        a = np.sqrt(nlayers / (nlayers + 1))
        W[-1] = a * Y[-1].clone()
        for i in np.flip(range(nlayers - 1)):
            a = np.sqrt((i + 1) / (i + 2))
            W[i] = a * (a * W[i + 1].clone() + Y[i].clone())

        return W

    def forward(self, X, Z=[]):
        if len(Z) == 0:
            Z = torch.zeros_like(X).unsqueeze(0).repeat_interleave(self.nlayers, dim=0)

        for k in range(self.nfixPointIter):
            Z = self.getNoNlinRHS(Z, X)
            Z = self.triDiagSolve(Z)
        return Z[-1], Z


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class UNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, channels=[32, 64, 128, 256], ncIn=64, embed_dim=256):
        super().__init__()

        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(ncIn, channels[0], 3, stride=1, bias=False, padding=1)
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False, padding=1)
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False, padding=1)
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False, padding=1)
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False, padding=1, output_padding=1)
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, padding=1,
                                         output_padding=1)
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, padding=1,
                                         output_padding=1)
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], ncIn, 3, stride=1, padding=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, xall=[]):
        # Encoding path
        h1 = self.conv1(x)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        return h, h1


class hyperUNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture.
    min sum 0.5*|W_j u_j - u_{j+1}|^2 + phi(u_j)
    2*u_j - W_j^T u_{j+1} - W_{j-1} u_{j-1} + \grad \phi(u_j) = 0
    u_{j+1} = 2 W_j*u_j - W_j*W_{j-1}*u_{j-1} + W_j*\grad \phi(u_j)

    """

    def __init__(self, in_channels=64):
        super().__init__()

        # Encoding layers where the resolution decreases
        self.crsOp = [0, 1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0]
        self.in_channels = in_channels
        channels = [self.in_channels, self.in_channels, self.in_channels * 4, self.in_channels * 4,
                    self.in_channels * 4 * 4, self.in_channels * 4 * 4
            , self.in_channels * 4 * 4 * 4, self.in_channels * 4 * 4 * 4, self.in_channels * 4 * 4,
                    self.in_channels * 4 * 4, self.in_channels * 4, self.in_channels * 4, self.in_channels]

        self.K = nn.ParameterList()
        self.bns = nn.ModuleList()
        for i in range(len(self.crsOp)):
            Ki = nn.Parameter(torch.nn.init.xavier_uniform_(torch.randn(channels[i], channels[i], 3, 3) * 1e-3))
            self.K.append(Ki)
            self.bns.append(torch.nn.BatchNorm2d(channels[i]))

        self.K00 = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.randn(self.in_channels, self.in_channels, 3, 3)) * 1e-4)
        self.K01 = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.randn(self.in_channels, self.in_channels, 3, 3)) * 1e-4)
        # self.K00 = nn.Parameter(torch.nn.init.xavier_uniform_(torch.randn(32, 3, 3, 3))*1e-4)
        # self.K01 = nn.Parameter(torch.nn.init.xavier_uniform_(torch.randn(3, 32, 3, 3))*1e-4)
        # self.proj = torch.nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, padding=1, stride=1)

    def spaceToChannel(self, X):
        X11 = X[:, :, 0::2, 0::2]
        X12 = X[:, :, 0::2, 1::2]
        X21 = X[:, :, 1::2, 0::2]
        X22 = X[:, :, 1::2, 1::2]
        Z = torch.cat((X11, X12, X21, X22), dim=1)
        return Z

    def channelToSpace(self, X):
        nc = X.shape[1]
        nc4 = nc // 4

        X11 = X[:, :nc4, :, :]
        X12 = X[:, nc4:2 * nc4, :, :]
        X21 = X[:, 2 * nc4:3 * nc4, :, :]
        X22 = X[:, 3 * nc4:, :, :]

        Xf = torch.zeros(X.shape[0], nc4, X.shape[2] * 2, X.shape[3] * 2, device=X.device)
        Xf[:, :, 0::2, 0::2] = X11
        Xf[:, :, 0::2, 1::2] = X12
        Xf[:, :, 1::2, 0::2] = X21
        Xf[:, :, 1::2, 1::2] = X22

        return Xf

    def crs(self, X, state):
        if state == 0:
            return X
        elif state == 1:
            return self.spaceToChannel(X)
        else:
            return self.channelToSpace(X)

    def doubleLayer(self, X, K, idx=None):
        dX = F.conv2d(X, K, padding=K.shape[-1] // 2)
        if idx is not None:
            dX = self.bns[idx](dX)
        else:
            dX = F.instance_norm(dX)
        dX = F.leaky_relu(dX, negative_slope=0.2)
        dX = F.conv_transpose2d(dX, K, padding=K.shape[-1] // 2)

        return dX

    def forward(self, X, Xall=[]):
        crsOp = self.crsOp
        Xold = F.conv2d(X, self.K00, padding=1)
        Xold = F.instance_norm(Xold)
        Xold = F.leaky_relu(Xold, negative_slope=0.2)
        Xold = F.conv2d(Xold, self.K01, padding=1)
        Xold = X + Xold
        for i in range(len(self.K)):
            dX = self.doubleLayer(X, self.K[i], idx=i)

            temp = X.clone()
            if i > 0:
                Y = self.crs(Xold, crsOp[i - 1])
            else:
                Y = Xold

            X = 2 * self.crs(X, crsOp[i]) - self.crs(Y, crsOp[i]) + self.crs(dX, crsOp[i])

            Xold = temp

        return X, Xold


from forwardOps import Embed


class neuralProximalGradient(nn.Module):
    def __init__(self, regNet, dataProj, forOp, niter=1):
        super(neuralProximalGradient, self).__init__()
        self.net = regNet
        self.dataProj = dataProj
        self.forOp = forOp
        self.niter = niter

        self.proj = Embed(embdsize=3, nin=3)
        self.mu = torch.nn.Parameter(torch.Tensor([0.01]))

    def forward(self, D):
        # initial recovey
        Z = self.forOp.adjoint(D, emb=False)
        Az = self.forOp.forward(Z, emb=False)
        alpha = (D * Az).mean(dim=(1,2,3), keepdim=True) / (Az * Az).mean(dim=(1,2,3), keepdim=True)
        Z = alpha*Z
        # Zref = torch.zeros((D.shape[0], 3, D.shape[2], D.shape[-1])).to(D.device)
        # Z = self.forOp.adjoint(D)
        # Zref = torch.zeros_like(Z)
        # Z, R = self.dataProj(D, Zref)
        # Zall = []
        Zall = []

        for i in range(self.niter):
            # network
            Zref, Zall = self.net(Z, Zall)
            #Zref = Z
            R = D - (self.forOp.forward(Zref, emb=False))
            #print("Iter:", i, ", Rnorm/Dnorm:", (R.norm()/D.norm()).item())
            G = self.forOp.adjoint(R, emb=False)
            Ag = self.forOp.forward(G, emb=False)
            mu = (R * Ag).mean(dim=(1,2,3), keepdim=True) / (Ag * Ag).mean(dim=(1,2,3), keepdim=True)
            Z = Zref + mu * G


        X = Z
        Xref = Zref
        return X, Xref, torch.Tensor([0])


class inerseSolveNet(nn.Module):
    def __init__(self, regNet, dataProj, forOp, niter=1):
        super(inerseSolveNet, self).__init__()
        self.net = regNet
        self.dataProj = dataProj
        self.forOp = forOp
        self.niter = niter

    def forward(self, D):
        # initial recovey
        Z = self.forOp.adjoint(D)
        Zref = torch.zeros_like(Z)
        Z, R = self.dataProj(D, Zref)
        Zall = []

        for i in range(self.niter):
            # network
            Zref, Zall = self.net(Z, Zall)
            # data projection
            Z, R = self.dataProj(D, Zref)

        X = self.forOp.Emb(Z)
        Xref = self.forOp.Emb(Zref)
        return X, Xref, R
