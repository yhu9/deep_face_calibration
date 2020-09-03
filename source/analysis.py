
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
from model import CalibrationNet3
import dataloader
import util

####################################################

parser = argparse.ArgumentParser(description="training arguments")
parser.add_argument("--out",default="results/exp.mat")
parser.add_argument("--model", default="")
parser.add_argument("--device",default='cpu')
parser.add_argument("--opt",default=False, action="store_true")
args = parser.parse_args()

####################################################

def train(modelin=args.model, modelout=args.out,device=args.device,outfile=args.out):
    # define logger
    #torch.manual_seed(6)
    #if log:
    #    logger = Logger(logname)
    # define model, dataloader, 3dmm eigenvectors, optimization method
    torch.manual_seed(2)
    calib_net = CalibrationNet3(n=1)
    sfm_net = CalibrationNet3(n=199)
    if modelin != "":
        calib_path = os.path.join('model','calib_' + modelin)
        sfm_path = os.path.join('model','sfm_' + modelin)
        calib_net.load_state_dict(torch.load(calib_path))
        sfm_net.load_state_dict(torch.load(sfm_path))
    calib_net.to(device=device)
    sfm_net.to(device=device)
    opt1 = torch.optim.Adam(calib_net.parameters(),lr=1e-4)
    opt2 = torch.optim.Adam(sfm_net.parameters(),lr=1e-3)

    # dataloader
    #data = dataloader.Data()
    #loader = data.batchloader
    #loader = dataloader.BIWILoader()
    loader = dataloader.AnalysisLoader()

    # mean shape and eigenvectors for 3dmm
    mu_lm = torch.from_numpy(loader.mu_lm).float().to(device=device)
    mu_lm[:,2] = mu_lm[:,2] * -1
    shape = mu_lm
    lm_eigenvec = torch.from_numpy(loader.lm_eigenvec).float().to(device=device)
    sigma = torch.from_numpy(loader.sigma).float().to(device=device)
    sigma = torch.diag(sigma.squeeze())
    lm_eigenvec = torch.mm(lm_eigenvec,sigma)

    # main training loop
    mval = np.array([100 + i*100 for i in range(10)])
    deltaz = np.zeros(10)
    e2d = np.zeros((10,10))
    e3d = np.zeros((10,10))
    eshape = np.zeros((10,10))
    efocal = np.zeros((10,10))
    for a1 in range(10):
        for a2 in range(10):
            errors2d = []
            errors3d = []
            errorsshape = []
            errorsfocal = []
            for a3 in range(10):
                loader.f = 500 + a3*100
                loader.minz = 50
                loader.maxz = 50 + a2*50
                loader.M = 100 + a1*100
		deltaz[a2] = loader.maxz - loader.minz
                for j, data in enumerate(loader):

                    M = loader.M
                    N = loader.N

                    # get the input and gt values
                    x_cam_gt = data['x_cam_gt'].to(device=device)
                    shape_gt = data['x_w_gt'].to(device=device)
                    fgt = data['f_gt'].to(device=device)
                    x_img = data['x_img'].to(device=device)
                    x_img_gt = data['x_img_gt'].to(device=device)

                    ptsI = x_img.reshape((M,N,2)).permute(0,2,1)
                    x = ptsI.unsqueeze(0).permute(0,2,1,3)

                    # run the model for initial values
                    calib_net.load_state_dict(torch.load(calib_path))
                    sfm_net.load_state_dict(torch.load(sfm_path))
                    calib_net.train()
                    sfm_net.train()
                    opt1 = torch.optim.Adam(calib_net.parameters(),lr=1e-4)
                    opt2 = torch.optim.Adam(sfm_net.parameters(),lr=1e-2)
                    f = calib_net.forward2(x) + 300
                    betas = sfm_net.forward2(x)
                    betas = betas.squeeze(0).unsqueeze(-1)
                    shape = mu_lm + torch.mm(lm_eigenvec,betas).squeeze().view(N,3)
                    curloss = 100
                    for outerloop in itertools.count():
                        # calibration
                        shape = shape.detach()
                        for iter in itertools.count():
                            opt1.zero_grad()

                            # focal length prediction
                            f = calib_net.forward2(x)
                            f = f + 300
                            K = torch.zeros((3,3)).float()
                            K[0,0] = f
                            K[1,1] = f
                            K[2,2] = 1

                            f_error = torch.mean(torch.abs(f - fgt))
                            rmse = torch.norm(shape_gt - shape,dim=1).mean()

                            # differentiable PnP pose estimation
                            km,c_w,scaled_betas, alphas = util.EPnP(ptsI,shape,K)
                            Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
                            error2d = util.getReprojError2(ptsI,shape,R,T,K,show=False,loss='l1')
                            loss = error2d.mean()
                            if iter == 5: break
                            if iter > 10 and prev_loss < loss:
                                break
                            else:
                                prev_loss = loss
                            loss.backward()
                            opt1.step()
                            print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.item():.1f}/{fgt[0].item():.1f} | error2d: {error2d.mean().item():.3f} | rmse: {rmse.item():.3f} ")


                        # sfm
                        f = f.detach()
                        for iter in itertools.count():
                            opt2.zero_grad()

                            # shape prediction
                            betas = sfm_net.forward2(x)
                            shape = torch.sum(betas * lm_eigenvec,1)
                            shape = shape.reshape(68,3) + mu_lm
                            K = torch.zeros((3,3)).float()
                            K[0,0] = f
                            K[1,1] = f
                            K[2,2] = 1

                            #rmse = torch.norm(shape_gt - shape,dim=1).mean().detach()
                            rmse = torch.norm(shape_gt - shape,dim=1).mean().detach()

                            # differentiable PnP pose estimation
                            km,c_w,scaled_betas,alphas = util.EPnP(ptsI,shape,K)
                            Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
                            error2d = util.getReprojError2(ptsI,shape,R,T,K,show=False,loss='l1')

                            loss = error2d.mean()
                            if iter == 5: break
                            if iter > 10 and prev_loss < loss:
                                break
                            else:
                                prev_loss = loss
                            loss.backward()
                            opt2.step()
                            print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.item():.1f}/{fgt[0].item():.1f} | error2d: {error2d.mean().item():.3f} | rmse: {rmse.item():.3f} ")

                        # closing condition for outerloop on dual objective
                        if torch.abs(curloss - loss) < 0.01: break
                        curloss = loss

                    # get errors
                    reproj_errors2 = util.getReprojError2(ptsI,shape,R,T,K).mean()
                    reproj_errors3 = torch.norm(shape_gt - shape,dim=1).mean()
                    rel_errors =  util.getRelReprojError3(x_cam_gt,shape,R,T).mean()
                    f_error = torch.abs(fgt - f.squeeze()) / fgt

                    errors2d.append(reproj_errors2.detach().cpu().item())
                    errors3d.append(rel_errors.detach().cpu().item())
                    errorsshape.append(reproj_errors3.detach().cpu().item())
                    errorsfocal.append(f_error.detach().cpu().item())
                    break
                break

            e2d[a1,a2] = np.mean(errors2d)
            e3d[a1,a2] = np.mean(errors3d)
            eshape[a1,a2] = np.mean(errorsshape)
            efocal[a1,a2] = np.mean(errorsfocal)

            print(f"ferror: {np.mean(errorsfocal):.3f} | e3d: {np.mean(errors3d):.3f} | e2d: {np.mean(errors2d):.2f} | eshape: {np.mean(errorsshape):.2f}")
            #end for
            break

    matdata = {}
    matdata['Mvals'] = np.array(mval)
    matdata['zvals'] = np.array(deltaz)
    matdata['error_2d'] = e2d
    matdata['error_shape'] = eshape
    matdata['error_rel3d'] = e3d
    matdata['error_focal'] = efocal
    scipy.io.savemat(outfile,matdata)

####################################################################################3
if __name__ == '__main__':
    train()

