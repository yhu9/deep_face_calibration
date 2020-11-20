
import itertools
import argparse
import os

import scipy.io
import torch
import numpy as np

from model2 import PointNet
import dataloader
import util

####################################################

parser = argparse.ArgumentParser(description="training arguments")
parser.add_argument("--model", default="net.pt")
parser.add_argument("--out",default="results/exp.mat")
parser.add_argument("--opt", default=False, action='store_true')
parser.add_argument("--db", default="syn")
args = parser.parse_args()

np.random.seed(0)
#########################################################
data3dmm = dataloader.SyntheticLoader()
mu_lm = torch.from_numpy(data3dmm.mu_lm).float().detach()
mu_lm[:,2] = mu_lm[:,2]*-1
lm_eigenvec = torch.from_numpy(data3dmm.lm_eigenvec).float().detach()
sigma = torch.from_numpy(data3dmm.sigma).float().detach()
sigma = torch.diag(sigma.squeeze())
lm_eigenvec = torch.mm(lm_eigenvec, sigma)

#########################################################
def trainfc(model):
    for name, param in model.named_parameters():
        if 'fc' in name and 'feat' not in name:
            param.requires_grad = True

# dual optimization to optimize focal length and 3D shape
def dualoptimization(x,calib_net,sfm_net,lm_eigenvec,betas,mu_s,shape_gt=None,fgt=None,M=100,N=68,mode='still'):

    if mode == 'still':
        alpha = 0.1
    else:
        alpha = 0.001

    # define what weights gets optimized
    calib_net.eval()
    sfm_net.eval()
    trainfc(calib_net)
    trainfc(sfm_net)

    ptsI = x.squeeze().permute(1,0).reshape((M,N,2)).permute(0,2,1)

    # run the model
    f = calib_net(x) + 300
    betas = sfm_net(x)
    betas = betas.squeeze(0).unsqueeze(-1)
    shape = mu_s + torch.mm(lm_eigenvec,betas).squeeze().view(N,3)
    shape = shape - shape.mean(0).unsqueeze(0)

    opt1 = torch.optim.Adam(calib_net.parameters(),lr=1e-5)
    opt2 = torch.optim.Adam(sfm_net.parameters(),lr=1)
    curloss = 100
    for outerloop in itertools.count():
        shape = shape.detach()
        for iter in itertools.count():
            opt1.zero_grad()
            f = calib_net(x) + 300
            K = torch.zeros(3,3).float()
            K[0,0] = f
            K[1,1] = f
            K[2,2] = 1

            # pose estimation
            km,c_w,scaled_betas, alphas = util.EPnP(ptsI,shape,K)
            _, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
            Xc = torch.bmm(R,torch.stack(M*[shape.T])) + T.unsqueeze(2)
            #shape_error = util.getShapeError(ptsI,Xc,shape,f,R,T)
            error_time = util.getTimeConsistency(shape,R,T)
            error2d = util.getReprojError2(ptsI,shape,R,T,K,show=False,loss='l2')

            # apply loss
            loss = error2d.mean() + alpha*error_time
            if iter >= 5: break
            prv_loss = loss.item()
            loss.backward()
            opt1.step()

            # log results on console
            if not shape_gt is None:
                rmse = torch.norm(shape_gt - shape,dim=1).mean().item()
            else:
                rmse = -1
            if not fgt is None:
                ftrue = fgt.item()
            else:
                fgt = -1
            f_error = torch.mean(torch.abs(f-ftrue))
            print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.item():.1f}/{ftrue:.1f} | error2d: {error2d.mean().item():.3f} | rmse: {rmse:.2f}")

        f = f.detach()
        for iter in itertools.count():
            opt2.zero_grad()

            # shape prediction
            betas = sfm_net(x)
            betas = betas.squeeze(0).unsqueeze(-1)
            shape = mu_s + torch.mm(lm_eigenvec,betas).squeeze().view(N,3)
            shape = shape - shape.mean(0).unsqueeze(0)
            K = torch.zeros((3,3)).float()
            K[0,0] = f
            K[1,1] = f
            K[2,2] = 1

            # differentiable PnP pose estimation
            km,c_w,scaled_betas,alphas = util.EPnP(ptsI,shape,K)
            _, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
            error2d = util.getReprojError2(ptsI,shape,R,T,K,show=False,loss='l2')
            Xc = torch.bmm(R,torch.stack(M*[shape.T])) + T.unsqueeze(2)
            #shape_error = util.getShapeError(ptsI,Xc,shape,f,R,T)
            error_time = util.getTimeConsistency(shape,R,T)

            # apply loss
            #loss = error2d.mean()
            loss = error2d.mean() + alpha*error_time
            #if iter >= 5 and loss > prv_loss: break
            if iter >= 5: break
            loss.backward()
            opt2.step()
            prv_loss = loss.item()

            # log results on console
            if not shape_gt is None:
                rmse = torch.norm(shape_gt - shape,dim=1).mean().item()
            else:
                rmse = -1
            if not fgt is None:
                ftrue = fgt.item()
            else:
                fgt = -1
            f_error = torch.mean(torch.abs(f-ftrue))
            print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.item():.1f}/{ftrue:.1f} | error2d: {error2d.mean().item():.3f} | rmse: {rmse:.2f}")

        if torch.abs(curloss  - loss) <= 0.01 or curloss < loss: break
        curloss = loss
    return shape,K,R,T

def test(modelin=args.model,outfile=args.out,optimize=args.opt):

    # define model, dataloader, 3dmm eigenvectors, optimization method
    calib_net = PointNet(n=1)
    sfm_net = PointNet(n=199)
    if modelin != "":
        calib_path = os.path.join('model','calib_' + modelin)
        sfm_path = os.path.join('model','sfm_' + modelin)
        calib_net.load_state_dict(torch.load(calib_path))
        sfm_net.load_state_dict(torch.load(sfm_path))
    calib_net.eval()
    sfm_net.eval()

    Mvals = [i for i in range(1,100)]
    Nvals = [i for i in range(8,68)]
    f_vals = [i*200 for i in range(2,7)]

    fpred = np.zeros((100,61,5,5))
    factual = np.zeros((100,61,5,5))
    depth_error = np.zeros((100,61,5,5))

    for i,viewcount in enumerate(Mvals):
        for j,ptcount in enumerate(Nvals):
            for l,ftest in enumerate(f_vals):
                data3dmm = dataloader.MNLoader(M=viewcount,N=ptcount,f=ftest,seed=0)
                M = data3dmm.M
                N = data3dmm.N
                mu_s = torch.from_numpy(data3dmm.mu_s).float().detach()
                mu_s[:,2] = mu_s[:,2]*-1
                lm_eigenvec = torch.from_numpy(data3dmm.lm_eigenvec).float().detach()
                sigma = torch.from_numpy(data3dmm.sigma).float().detach()
                sigma = torch.diag(sigma.squeeze())
                lm_eigenvec = torch.mm(lm_eigenvec, sigma)

                for k in range(5):
                    data = data3dmm[k]

                    # load the data
                    x_cam_gt = data['x_cam_gt']
                    shape_gt = data['x_w_gt']
                    fgt = data['f_gt']
                    x_img = data['x_img']
                    x_img_gt = data['x_img_gt']

                    ptsI = x_img.reshape((M,N,2)).permute(0,2,1)
                    x = x_img.unsqueeze(0).permute(0,2,1)

                    # run the model
                    f = calib_net(x) + 300
                    betas = sfm_net(x)
                    betas = betas.squeeze(0).unsqueeze(-1)
                    shape = mu_s + torch.mm(lm_eigenvec,betas).squeeze().view(N,3)
                    shape = shape - shape.mean(0).unsqueeze(0)

                    # get motion type
                    K = torch.zeros(3,3).float()
                    K[0,0] = f
                    K[1,1] = f
                    K[2,2] = 1
                    km,c_w,scaled_betas,alphas = util.EPnP(ptsI,shape,K)
                    _, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI)
                    error_time = util.getTimeConsistency(shape,R,T)
                    if error_time > 20:
                        mode='walk'
                    else:
                        mode='still'

                    # apply dual optimization
                    if optimize:
                        calib_net.load_state_dict(torch.load(calib_path))
                        sfm_net.load_state_dict(torch.load(sfm_path))
                        shape,K,R,T = dualoptimization(x,calib_net,sfm_net,lm_eigenvec,betas,mu_s,shape_gt=shape_gt,fgt=fgt,M=M,N=N,mode=mode)
                        f = K[0,0].detach()
                    else:
                        K = torch.zeros(3,3).float()
                        K[0,0] = f
                        K[1,1] = f
                        K[2,2] = 1
                        km,c_w,scaled_betas,alphas = util.EPnP(ptsI,shape,K)
                        Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI)

                    # get errors
                    rel_errors =  util.getRelReprojError3(x_cam_gt,shape,R,T)

                    fpred[i,j,l,k] = f.detach().cpu().item()
                    factual[i,j,l,k] = fgt.detach().cpu().item()
                    depth_error[i,j,l,k] = rel_errors.cpu().mean().item()
                    print(f"M: {viewcount} | N: {ptcount} | f/fgt: {fpred[i,j,l,k]:.2f}/{factual[i,j,l,k]}")

                ferror = np.mean(np.abs(fpred[i,j,l] - factual[i,j,l]) / factual[i,j,l])
                derror = np.mean(depth_error[i,j,l])
                f = np.mean(fpred[i,j,l])
                print(f"M: {viewcount} | N: {ptcount} | f/fgt: {f:.2f}/{ftest:.2f} | ferror: {ferror:.2f} | derror: {derror:.2f}")

    matdata = {}
    matdata['fpred'] = fpred
    matdata['fgt'] = factual
    matdata['derror'] = depth_error
    scipy.io.savemat(outfile,matdata)

    print(f"saved output to {outfile}")

####################################################################################3
if __name__ == '__main__':

    if args.db == 'syn':
        test()
    elif args.db == 'biwi':
        testBIWI()
    elif args.db == 'biwiid':
        testBIWIID()
    else:
        test()


