
import itertools
import argparse

import torch
import torch.optim
import numpy as np
import pptk
import scipy.io

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
    calib_model= Model1(k=1,feature_transform=False)
    sfm_model = Model1(k=199,feature_transform=False)
    #if modelin != "":
    #    model.load_state_dict(torch.load(modelin))
    sfm_model
    calib_model
    opt1 = torch.optim.Adam(sfm_model.parameters(),lr=1e-2)
    opt2 = torch.optim.Adam(calib_model.parameters(),lr=1e-1)

    # dataloader
    #data = dataloader.Data()
    #loader = data.batchloader
    loader = dataloader.BIWILoader()

    # mean shape and eigenvectors for 3dmm
    mu_lm = torch.from_numpy(loader.mu_lm).float()
    mu_lm[:,2] = mu_lm[:,2] * -1
    shape = mu_lm.detach()
    lm_eigenvec = torch.from_numpy(loader.lm_eigenvec).float()

    # main training loop
    for epoch in itertools.count():
        #for j,batch in enumerate(loader):

        np.random.seed(2)
        for j, data in enumerate(loader):

            M = loader.M
            N = loader.N

            # get the input and gt values
            x_cam_gt = data['x_cam_gt']
            x_w_gt = data['x_w_gt']
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

            '''
            K = torch.zeros((3,3)).float().cuda()
            K[0,0] = fgt[0]
            K[1,1] = fgt[0]
            K[2,2] = 1
            km, c_w, scaled_betas, alphas = util.EPnP(ptsI,shape,K)
            Xc,R,T, scaled_betas = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
            loss = util.getReprojError2(ptsI,shape,R,T,K,show=False).mean()
            '''

            fvals = []
            errors = []
            for iter in itertools.count():
                opt2.zero_grad()

                # model output
                #betas,_,_ = model(x.unsqueeze(0))
                #shape = torch.sum(betas * lm_eigenvec,1)
                #shape = shape.reshape(68,3) + mu_lm
                #shape[:,2] = shape[:,2] * -1

                # model output
                f,_,_ = calib_model(x.unsqueeze(0))
                f = torch.nn.functional.leaky_relu(f) + 300
                K = torch.zeros((3,3)).float()
                K[0,0] = f
                K[1,1] = f
                K[2,2] = 1

                # differentiable pose estimation
                km,c_w,scaled_betas, alphas = util.EPnP(ptsI,shape,K)
                Xc, R, T, _ = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
                loss = util.getReprojError2(ptsI,shape,R,T,K).mean()
                #loss = util.getError2(ptsI,Xc,K).mean()
                loss.backward()
                opt2.step()

                errors.append(loss.detach().cpu().item())
                fvals.append(f.detach().cpu().item())

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

                print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.item():.1f}/{fgt[0].item():.1f}")

            # optimize 3d shape
            v = pptk.viewer([0,0,0])
            v.set(point_size=1)
            v.set(lookat=[0,0,0])
            for iter in itertools.count():
                opt1.zero_grad()

                # model output
                betas, _, _ = sfm_model(x.unsqueeze(0))
                shape = torch.sum(betas * lm_eigenvec,1)
                shape = shape.reshape(68,3) + mu_lm
                shape[:,2] = shape[:,2] * -1
                K = torch.zeros((3,3)).float()
                K[0,0] = 1000
                K[1,1] = 1000
                K[2,2] = 1

                # EPnP
                km,c_w,scaled_betas,alphas = util.EPnP(ptsI,shape,K)
                Xc, R, T, _ = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
                loss = util.getReprojError2(ptsI,shape,R,T,K).mean()
                loss.backward()
                opt1.step()

                print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {1000}/{fgt[0].item():.1f}")

                data = {}
                data['shape'] = shape.detach().cpu().numpy()
                data['mu_lm'] = mu_lm.detach().cpu().numpy()
                scipy.io.savemat(f"visual/shape{iter:03d}.mat",data)

                #visualize shape
                v.clear()
                pts = shape.detach().cpu().numpy()
                v.load(pts)
                #if iter == 100: break
            shape = shape.detach()
        # save model and increment weight decay
        #print("saving!")
        #torch.save(model.state_dict(), modelout)
        #decay.step()


####################################################################################3
if __name__ == '__main__':
    train()

