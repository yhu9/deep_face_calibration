
import itertools
import argparse
import os

import torch
import torch.optim

from model4 import PointNet
from test5 import test
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
    calib_net = PointNet(n=1,N=68)
    sfm_net = PointNet(n=199,N=68)
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
    opt1 = torch.optim.Adam(calib_net.parameters(),lr=1e-4)
    opt2 = torch.optim.Adam(sfm_net.parameters(),lr=1e-2)

    # dataloader
    loader = dataloader.SyntheticLoader()
    #data = dataloader.Data()
    #loader = data.batchloader
    #batch_size = data.batchsize
    batch_size = 100

    # mean shape and eigenvectors for 3dmm
    #data3dmm = dataloader.SyntheticLoader()
    #mu_lm = torch.from_numpy(data3dmm.mu_lm).float().detach()
    #mu_lm[:,2] = mu_lm[:,2]*-1
    #lm_eigenvec = torch.from_numpy(data3dmm.lm_eigenvec).float().detach()
    #sigma = torch.from_numpy(data3dmm.sigma).float().detach()
    #sigma = torch.diag(sigma.squeeze())
    #lm_eigenvec = torch.mm(lm_eigenvec, sigma)

    mu_lm = torch.from_numpy(loader.mu_lm).float()
    mu_lm[:,2] = mu_lm[:,2] * -1
    mu_lm = torch.stack(batch_size * [mu_lm])
    shape = mu_lm
    lm_eigenvec = torch.from_numpy(loader.lm_eigenvec).float().to(device=device)
    lm_eigenvec = torch.stack(batch_size * [lm_eigenvec])

    M = 100
    N = 68

    # main training loop
    best = 1000
    for epoch in itertools.count():
        for j,batch in enumerate(loader):

            # get the input and gt values
            x_cam_gt = batch['x_cam_gt'].to(device=device)
            shape_gt = batch['x_w_gt'].to(device=device)
            fgt = batch['f_gt'].to(device=device)
            x_img = batch['x_img'].to(device=device)
            #beta_gt = batch['beta_gt'].to(device=device)
            #x_img_norm = batch['x_img_norm']
            x_img_gt = batch['x_img_gt'].to(device=device).permute(1,0,2)
            batch_size = fgt.shape[0]

            # train single view model
            #x = x_img.permute(1,0)
            #x = x_img.permute(0,2,1).reshape(batch_size,2,M,N)

            ptsI = x_img.reshape(M,N,2).permute(0,2,1)
            x = ptsI

            # if just optimizing
            if not opt:
                # calibration
                f = calib_net(x) + 300
                f = f.mean()
                K = torch.zeros((M,3,3)).float().to(device=device)
                K[:,0,0] = f.squeeze()
                K[:,1,1] = f.squeeze()
                K[:,2,2] = 1

                # sfm
                betas = sfm_net(x)
                betas = betas.unsqueeze(-1)
                shape = mu_lm + torch.bmm(lm_eigenvec,betas).squeeze().view(M,N,3)
                shape = shape - shape.mean(1).unsqueeze(1)
                shape = shape.mean(0)

                opt1.zero_grad()
                opt2.zero_grad()
                f_error = torch.mean(torch.abs(f - fgt))
                #error2d = torch.mean(torch.abs(pred - x_img_gt))
                error3d = torch.mean(torch.abs(shape - shape_gt))
                error = f_error + error3d
                error.backward()
                opt1.step()
                opt2.step()


                print(f"f_error: {f_error.item():.3f} | error3d: {error3d.item():.3f} | f/fgt: {f.item():.1f}/{fgt.item():.1f} ")
                continue

        # save model and increment weight decay
        torch.save(sfm_net.state_dict(), os.path.join('model','sfm_model.pt'))
        torch.save(calib_net.state_dict(), os.path.join('model','calib_model.pt'))
        ferror = test(modelin='model.pt',outfile=args.out,optimize=False)
        if ferror < best:
            best = ferror
            print("saving!")
            torch.save(sfm_net.state_dict(), os.path.join('model','sfm_'+modelout))
            torch.save(calib_net.state_dict(), os.path.join('model','calib_'+modelout))
        sfm_net.train()
        calib_net.train()
        #decay.step()


####################################################################################3
if __name__ == '__main__':
    train()

