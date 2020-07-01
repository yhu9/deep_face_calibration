
import itertools
import argparse

import torch
import torch.optim
import numpy as np

from logger import Logger
#from model import Model1
from model import Model2
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
    #torch.manual_seed(2)
    #model = Model1(k=1,feature_transform=False)
    model = Model2(k=1,feature_transform=False)
    if modelin != "":
        model.load_state_dict(torch.load(modelin))
    model.apply(util.init_weights)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-1)

    # dataloader
    #data = dataloader.Data()
    #loader = data.batchloader
    loader = dataloader.SyntheticLoader()

    # mean shape and eigenvectors for 3dmm
    mu_lm = torch.from_numpy(loader.mu_lm).float()#.cuda()
    mu_lm[:,2] = mu_lm[:,2] * -1
    shape = mu_lm.detach()#.cuda()
    lm_eigenvec = torch.from_numpy(loader.lm_eigenvec).float()#.cuda()

    M = loader.M
    N = loader.N

    # main training loop
    for epoch in itertools.count():
        #for j,batch in enumerate(loader):

        #np.random.seed(0)
        for j, data in enumerate(loader):

            # load the data
            x_cam_gt = data['x_cam_gt']#.cuda()
            x_w_gt = data['x_w_gt']#.cuda()
            fgt = data['f_gt']#.cuda()
            x_img = data['x_img']#.cuda()
            x_img_gt = data['x_img_gt']#.cuda()

            x_img_pts = x_img.reshape((M,N,2)).permute(0,2,1)
            one = torch.ones(M*N,1)#.cuda()
            x_img_one = torch.cat([x_img,one],dim=1)
            x_cam_pt = x_cam_gt.permute(0,2,1).reshape(M*N,3)

            # create the input
            b = 10
            x = x_img_one.reshape(M,N,3).reshape(b,M//b,N,3).reshape(b,M//b*N,3)
            x = x.permute(0,2,1)
            ptsI = x_img.reshape((M,N,2)).permute(0,2,1)

            #km, c_w, scaled_betas, alphas = util.EPnP(ptsI,shape,K)
            #Xc,R,T, scaled_betas = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
            #loss = util.getReprojError2(ptsI,shape,R,T,K).mean()

            # optimize using EPNP+GN
            fvals = []
            errors = []
            for iter in itertools.count():
                optimizer.zero_grad()

                # model output
                f,_,_ = model(x)
                f = f + 1000
                K = torch.zeros((b,3,3)).float()
                K[:,0,0] = f.squeeze()
                K[:,1,1] = f.squeeze()
                K[:,2,2] = 1

                # differentiable pose estimation
                losses = []
                for i in range(b):
                    j = i+1
                    km, c_w, scaled_betas, alphas = util.EPnP(ptsI[i:j*b],shape,K[i])
                    Xc, R, T, _ = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI[i:j*b],K[i])
                    error2d = util.getReprojError2(ptsI[i:j*b],shape,R,T,K[i]).mean()
                    losses.append(error2d)
                loss = torch.stack(losses).mean()
                loss.backward()
                optimizer.step()
                print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.mean().item():.1f}/{fgt[0].item():.1f}")
                if iter == 50: break

                if f.mean() > 1000:
                    break

            data = {}
            data['ptsI'] = ptsI.detach().cpu().numpy()
            data['shape'] = shape.detach().cpu().numpy()
            data['R'] = R.detach().cpu().numpy()
            data['T'] = T.detach().cpu().numpy()
            data['Xc'] = Xc.detach().cpu().numpy()
            data['K'] = K.detach().cpu().numpy()
            data['fvals'] = np.array(fvals)
            data['loss'] = np.array(errors)
            scipy.io.savemat(f"visual/shape{iter:03d}.mat",data)


            quit()

####################################################################################3
if __name__ == '__main__':
    train()

