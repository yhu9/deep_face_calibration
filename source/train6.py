
import itertools
import argparse
import os

import torch
import torch.optim

from model3 import PointNet
from test6 import test
import dataloader
import util

####################################################

parser = argparse.ArgumentParser(description="training arguments")
parser.add_argument("--out", default="model.pt")
parser.add_argument("--model", default="")
parser.add_argument("--device",default='cpu')
parser.add_argument("--opt",default=False, action="store_true")
args = parser.parse_args()

####################################################


def train(modelin=args.model, modelout=args.out,device=args.device,opt=args.opt):

    # define model, dataloader, 3dmm eigenvectors, optimization method
    calib_net = PointNet(n=1)
    sfm_net = PointNet(n=199)
    if modelin != "":
        calib_path = os.path.join('model','calib_' + modelin)
        sfm_path = os.path.join('model','sfm_' + modelin)
        pretrained1 = torch.load(calib_path)
        pretrained2 = torch.load(sfm_path)
        calib_dict = calib_net.state_dict()
        sfm_dict = sfm_net.state_dict()

        pretrained1 = {k: v for k,v in pretrained1.items() if k in calib_dict}
        pretrained2 = {k: v for k,v in pretrained2.items() if k in sfm_dict}
        calib_dict.update(pretrained1)
        sfm_dict.update(pretrained2)

        calib_net.load_state_dict(pretrained1)
        sfm_net.load_state_dict(pretrained2)

    calib_net.to(device=device)
    sfm_net.to(device=device)
    opt1 = torch.optim.Adam(calib_net.parameters(),lr=1e-3)
    opt2 = torch.optim.Adam(sfm_net.parameters(),lr=1e-3)

    # dataloader
    data = dataloader.Data()
    loader = data.batchloader
    batch_size = data.batchsize

    # mean shape and eigenvectors for 3dmm
    mu_lm = torch.from_numpy(data.mu_lm).float()#.to(device=device)
    mu_lm[:,2] = mu_lm[:,2] * -1
    mu_lm = torch.stack(batch_size * [mu_lm.to(device=device)])
    shape = mu_lm
    lm_eigenvec = torch.from_numpy(data.lm_eigenvec).float().to(device=device)
    sigma = torch.from_numpy(data.sigma).float().detach().to(device=device)
    sigma = torch.diag(sigma.squeeze())
    lm_eigenvec = torch.mm(lm_eigenvec, sigma)
    lm_eigenvec = torch.stack(batch_size * [lm_eigenvec])

    M = data.M
    N = data.N

    # main training loop
    best = 10000
    for epoch in itertools.count():
        for j,batch in enumerate(loader):

            # get the input and gt values
            x_cam_gt = batch['x_cam_gt'].to(device=device)
            shape_gt = batch['x_w_gt'].to(device=device)
            fgt = batch['f_gt'].to(device=device)
            x_img = batch['x_img'].to(device=device)
            #beta_gt = batch['beta_gt'].to(device=device)
            #x_img_norm = batch['x_img_norm']
            x_img_gt = batch['x_img_gt'].to(device=device).permute(0,2,1,3)
            batch_size = fgt.shape[0]

            one = torch.ones(batch_size,M*N,1).to(device=device)
            x_img_one = torch.cat([x_img,one],dim=2)
            x_cam_pt = x_cam_gt.permute(0,1,3,2).reshape(batch_size,6800,3)
            x = x_img.permute(0,2,1)
            #x = x_img.permute(0,2,1).reshape(batch_size,2,M,N)

            ptsI = x_img_one.reshape(batch_size,M,N,3).permute(0,1,3,2)[:,:,:2,:]

            # if just optimizing
            if not opt:
                # calibration
                f = calib_net(x) + 300
                K = torch.zeros((batch_size,3,3)).float().to(device=device)
                K[:,0,0] = f.squeeze()
                K[:,1,1] = f.squeeze()
                K[:,2,2] = 1

                # sfm
                betas = sfm_net(x)
                betas = betas.unsqueeze(-1)
                shape = mu_lm + torch.bmm(lm_eigenvec,betas).squeeze().view(batch_size,N,3)
                shape = shape - shape.mean(1).unsqueeze(1)

                #import pptk
                #import numpy as np
                #s = shape.detach().cpu().numpy()
                #sgt = shape_gt.detach().cpu().numpy()
                #pts = np.concatenate((sgt[0],s[0]))
                #color = np.zeros(pts.shape)
                #color[:68,2] = 255
                #v = pptk.viewer(pts,color)
                #v.set(point_size=1.1)

                opt1.zero_grad()
                opt2.zero_grad()
                f_error = torch.mean(torch.abs(f - fgt))
                #error2d = torch.mean(torch.abs(pred - x_img_gt))
                error3d = torch.mean(torch.norm(shape - shape_gt,dim=2))
                error = f_error + error3d
                error.backward()
                opt1.step()
                opt2.step()

                print(f"best: {best:.2f} | f_error: {f_error.item():.3f} | error3d: {error3d.item():.3f} | f/fgt: {f[0].item():.1f}/{fgt[0].item():.1f} | f/fgt: {f[1].item():.1f}/{fgt[1].item():.1f} | f/fgt: {f[2].item():.1f}/{fgt[2].item():.1f} | f/fgt: {f[3].item():.1f}/{fgt[3].item():.1f} ")
                continue

        # save model and increment weight decay
        ferror = test(modelin=args.out,outfile=args.out,optimize=False)
        if ferror < best:
            best = ferror
            print("saving!")
            torch.save(sfm_net.state_dict(), os.path.join('model','sfm_'+modelout))
            torch.save(calib_net.state_dict(), os.path.join('model','calib_'+modelout))
        #decay.step()


####################################################################################3
if __name__ == '__main__':
    train()

