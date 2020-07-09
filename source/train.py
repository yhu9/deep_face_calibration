
import itertools
import argparse
import os

import torch
import torch.optim
import numpy as np
import pptk
import scipy.io

#from logger import Logger
from model import Model1
import dataloader
import util

####################################################

parser = argparse.ArgumentParser(description="training arguments")
parser.add_argument("--out", default="model.pt")
parser.add_argument("--model", default="")
#parser.add_argument("--log",type=int,default=1)
#parser.add_argument("--logname",default="log")
args = parser.parse_args()

####################################################

def train(modelin=args.model, modelout=args.out):
    # define logger
    #torch.manual_seed(6)
    #if log:
    #    logger = Logger(logname)
    # define model, dataloader, 3dmm eigenvectors, optimization method
    torch.manual_seed(2)
    calib_net = Model1(k=1,feature_transform=False)
    sfm_net = Model1(k=199,feature_transform=False)
    #if modelin != "":
    #    model.load_state_dict(torch.load(modelin))
    opt1 = torch.optim.Adam(calib_net.parameters(),lr=1e-1)
    opt2 = torch.optim.Adam(sfm_net.parameters(),lr=1e-1)

    # dataloader
    #data = dataloader.Data()
    #loader = data.batchloader
    #loader = dataloader.BIWILoader()
    loader = dataloader.SyntheticLoader()

    # mean shape and eigenvectors for 3dmm
    mu_lm = torch.from_numpy(loader.mu_lm).float()
    #mu_lm[:,2] = mu_lm[:,2] * -1
    shape = mu_lm
    lm_eigenvec = torch.from_numpy(loader.lm_eigenvec).float()

    # main training loop
    for epoch in itertools.count():
        for j, data in enumerate(loader):

            M = loader.M
            N = loader.N

            # get the input and gt values
            x_cam_gt = data['x_cam_gt']
            shape_gt = data['x_w_gt']
            fgt = data['f_gt']
            x_img = data['x_img']
            x_img_gt = data['x_img_gt']

            x_img_pts = x_img.reshape((M,N,2)).permute(0,2,1)
            one = torch.ones(M*N,1)
            x_img_one = torch.cat([x_img,one],dim=1)
            x_cam_pt = x_cam_gt.permute(0,2,1).reshape(M*N,3)
            x = x_img_one.permute(1,0)

            # get initial values for betas and alphas of EPNP
            ptsI = x_img.reshape((M,N,2)).permute(0,2,1)

            fvals = []
            errors = []
            for outerloop in itertools.count():

                # calibration
                shape = shape.detach()
                for iter in itertools.count():
                    opt1.zero_grad()

                    # focal length prediction
                    f,_,_ = calib_net(x.unsqueeze(0))
                    f = f + 300
                    K = torch.zeros((3,3)).float()
                    K[0,0] = f
                    K[1,1] = f
                    K[2,2] = 1

                    # RMSE between GT and predicted shape
                    rmse = torch.norm(shape_gt - shape,dim=1).mean().detach()

                    # error f
                    error_f = torch.mean(torch.abs(f - fgt))

                    # differentiable PnP pose estimation
                    km,c_w,scaled_betas, alphas = util.EPnP(ptsI,shape,K)
                    Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
                    error2d = util.getReprojError2(ptsI,shape,R,T,K,show=False,loss='l1')
                    loss = error2d.mean() + error_f
                    if iter > 20 and prev_loss < loss:
                        break
                    else:
                        prev_loss = loss
                    loss.backward()
                    opt1.step()
                    print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.item():.1f}/{fgt[0].item():.1f} | rmse: {rmse.item():.2f}")

                # sfm
                f = f.detach()
                for iter in itertools.count():
                    opt2.zero_grad()

                    # shape prediction
                    betas,_,_ = sfm_net(x.unsqueeze(0))
                    shape = torch.sum(betas * lm_eigenvec,1)
                    shape = shape.reshape(68,3) + mu_lm
                    K = torch.zeros((3,3)).float()
                    K[0,0] = f
                    K[1,1] = f
                    K[2,2] = 1

                    # RMSE between GT and predicted shape
                    rmse = torch.norm(shape_gt - shape,dim=1).mean().detach()

                    # differentiable PnP pose estimation
                    km,c_w,scaled_betas, alphas = util.EPnP(ptsI,shape,K)
                    Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
                    error2d = util.getReprojError2(ptsI,shape,R,T,K,show=False,loss='l2')
                    loss = error2d.mean()
                    if iter > 20 and prev_loss < loss:
                        break
                    else:
                        prev_loss = loss
                    loss.backward()
                    opt2.step()
                    print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.item():.1f}/{fgt[0].item():.1f} | rmse: {rmse.item():.2f}")


                if outerloop == 2: break


            # get errors
            reproj_errors2 = util.getReprojError2(ptsI,shape,R,T,K)
            reproj_errors3 = util.getReprojError3(x_cam_gt,shape,R,T)
            rel_errors =  util.getRelReprojError3(x_cam_gt,shape,R,T)

            reproj_error = reproj_errors2.mean()
            reconstruction_error = reproj_errors3.mean()
            rel_error = rel_errors.mean()
            f_error = torch.abs(fgt - f) / fgt

            print(f"f/fgt: {f[0].item():.3f}/{fgt.item():.3f} |  f_error_rel: {f_error.item():.4f}  | rmse: {reconstruction_error.item():.4f}  | rel rmse: {rel_error.item():.4f}    | 2d error: {reproj_error.item():.4f}")
            #end for

        torch.save(sfm_net.state_dict(), os.path.join('model','sfm_'+modelout))
        torch.save(calib_net.state_dict(), os.path.join('model','calib_'+modelout))

####################################################################################3
if __name__ == '__main__':
    train()

