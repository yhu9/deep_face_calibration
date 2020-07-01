
import itertools
import argparse

import torch
import torch.optim
import numpy as np
import pptk

from logger import Logger
from model import Model1
import dataloader
import util

####################################################

parser = argparse.ArgumentParser(description="training arguments")
parser.add_argument("--out", default="model/model.pt")
parser.add_argument("--model", default="")
parser.add_argument("--log",type=int,default=1)
parser.add_argument("--logname",default="log")
args = parser.parse_args()

####################################################

def train(modelin=args.model, modelout=args.out,log=args.log,logname=args.logname):
    # define logger
    #torch.manual_seed(6)
    if log:
        logger = Logger(logname)

    # define model, dataloader, 3dmm eigenvectors, optimization method
    torch.manual_seed(2)
    model = Model1(k=199,feature_transform=False)
    if modelin != "":
        model.load_state_dict(torch.load(modelin))
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
    decay = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

    # dataloader
    #data = dataloader.Data()
    #loader = data.batchloader
    loader = dataloader.SyntheticLoader()

    # mean shape and eigenvectors for 3dmm
    mu_lm = torch.from_numpy(loader.mu_lm).float().cuda()
    #mu_lm[:,2] = mu_lm[:,2] * -1
    #shape = mu_lm.detach().cuda()
    lm_eigenvec = torch.from_numpy(loader.lm_eigenvec).float().cuda()

    M = loader.M
    N = loader.N

    # main training loop
    for epoch in itertools.count():
        #for j,batch in enumerate(loader):

        np.random.seed(2)
        for j, data in enumerate(loader):

            # get the input and gt values
            x_cam_gt = data['x_cam_gt'].cuda()
            x_w_gt = data['x_w_gt'].cuda()
            fgt = data['f_gt'].cuda()
            beta_gt = data['beta_gt'].cuda()
            x_img = data['x_img'].cuda()
            #x_img_norm = data['x_img_norm']
            x_img_gt = data['x_img_gt'].cuda()

            #batch_size = fgt.shape[0]
            x_img_pts = x_img.reshape((M,N,2)).permute(0,2,1)

            one = torch.ones(M*N,1).cuda()
            x_img_one = torch.cat([x_img,one],dim=1)
            x_cam_pt = x_cam_gt.permute(0,2,1).reshape(6800,3)

            # run the model
            x = x_img_one.permute(1,0)

            # get initial values for betas and alphas of EPNP
            ptsI = x_img.reshape((M,N,2)).permute(0,2,1)

            v = pptk.viewer([0,0,0])
            v.set(point_size=1)
            # optimize using EPNP+GN
            for iter in itertools.count():
                optimizer.zero_grad()

                # model output
                betas,_,_ = model(x.unsqueeze(0))
                shape = torch.sum(betas * lm_eigenvec,1)
                shape = shape.reshape(68,3) + mu_lm

                K = torch.zeros((3,3)).float().cuda()
                K[0,0] = 400
                K[1,1] = 400
                K[2,2] = 1

                # differentiable pose estimation
                km,c_w,scaled_betas, alphas = util.EPnP(ptsI,shape,K)
                Xc, R, T, _ = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
                loss = util.getReprojError2(ptsI,shape,R,T,K).mean()
                loss.backward()
                optimizer.step()

                #visualize shape
                v.clear()
                pts = shape.detach().cpu().numpy()
                v.load(pts)
                v.set(r=300)

                print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {fgt[0].item():.1f}/{fgt[0].item():.1f}")

        # save model and increment weight decay
        print("saving!")
        torch.save(model.state_dict(), modelout)
        decay.step()


####################################################################################3
if __name__ == '__main__':
    train()

