
import itertools
import argparse
import os

import scipy.io
import torch
import numpy as np

from model import CalibrationNet3
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
def trainfc(model):
    for name, param in model.named_parameters():
        if 'fc' in name and 'feat' not in name:
            param.requires_grad = True

def test(modelin=args.model,outfile=args.out,optimize=args.opt):

    # define model, dataloader, 3dmm eigenvectors, optimization method
    calib_net = CalibrationNet3(n=1)
    sfm_net = CalibrationNet3(n=199)
    if modelin != "":
        calib_path = os.path.join('model','calib_' + modelin)
        sfm_path = os.path.join('model','sfm_' + modelin)
        calib_net.load_state_dict(torch.load(calib_path))
        sfm_net.load_state_dict(torch.load(sfm_path))
    calib_net.eval()
    sfm_net.eval()

    Mvals = [i * 100 for i in range(1,10)]
    Nvals = [i * 100 for i in range(1,10)]
    f_vals = [i*100 for i in range(4,14)]

    fpred = np.zeros((10,10,10,10))
    factual = np.zeros((10,10,10,10))
    depth_error = np.zeros((10,10,10,10))

    for i,viewcount in enumerate(Mvals):
        for j,ptcount in enumerate(Nvals):
            for l,ftest in enumerate(f_vals):
                data3dmm = dataloader.AnalysisLoader(M=viewcount,N=ptcount,f=ftest)
                M = data3dmm.M
                N = data3dmm.N
                mu_s = torch.from_numpy(data3dmm.mu_s).float().detach()
                mu_s[:,2] = mu_s[:,2]*-1
                lm_eigenvec = torch.from_numpy(data3dmm.lm_eigenvec).float().detach()
                sigma = torch.from_numpy(data3dmm.sigma).float().detach()
                sigma = torch.diag(sigma.squeeze())
                lm_eigenvec = torch.mm(lm_eigenvec, sigma)

                for k in range(10):
                    data = data3dmm[k]

                    # load the data
                    x_cam_gt = data['x_cam_gt']
                    shape_gt = data['x_w_gt']
                    fgt = data['f_gt']
                    x_img = data['x_img']
                    x_img_gt = data['x_img_gt']

                    ptsI = x_img.reshape((M,N,2)).permute(0,2,1)
                    x = ptsI.unsqueeze(0).permute(0,2,1,3)

                    # run the model
                    f = calib_net(x) + 300
                    betas = sfm_net(x)
                    betas = betas.squeeze(0).unsqueeze(-1)
                    shape = mu_s + torch.mm(lm_eigenvec,betas).squeeze().view(N,3)

                    # additional optimization on initial solution
                    if optimize:
                        calib_net.load_state_dict(torch.load(calib_path))
                        sfm_net.load_state_dict(torch.load(sfm_path))
                        calib_net.eval()
                        sfm_net.eval()
                        trainfc(calib_net)
                        trainfc(sfm_net)
                        opt1 = torch.optim.Adam(calib_net.parameters(),lr=1e-4)
                        opt2 = torch.optim.Adam(sfm_net.parameters(),lr=1e-2)
                        curloss = 100
                        for outerloop in itertools.count():

                            # camera calibration
                            shape = shape.detach()
                            for iter in itertools.count():
                                opt1.zero_grad()
                                f = calib_net.forward2(x) + 300
                                K = torch.zeros(3,3).float()
                                K[0,0] = f
                                K[1,1] = f
                                K[2,2] = 1

                                f_error = torch.mean(torch.abs(f - fgt))
                                rmse = torch.norm(shape_gt - shape,dim=1).mean()

                                # differentiable PnP pose estimation
                                km,c_w,scaled_betas, alphas = util.EPnP(ptsI,shape,K)
                                Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
                                error2d = util.getReprojError2(ptsI,shape,R,T,K,show=False,loss='l1')
                                error_time = util.getTimeConsistency(shape,R,T)

                                loss = error2d.mean() + 0.01*error_time
                                if iter == 5: break
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
                                shape = shape.reshape(-1,3) + mu_s
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
                                error_time = util.getTimeConsistency(shape,R,T)

                                loss = error2d.mean() + 0.01*error_time
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
                    else:
                        K = torch.zeros(3,3).float()
                        K[0,0] = f
                        K[1,1] = f
                        K[2,2] = 1
                        km,c_w,scaled_betas,alphas = util.EPnP(ptsI,shape,K)
                        Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)

                    # get errors
                    rel_errors =  util.getRelReprojError3(x_cam_gt,shape,R,T)

                    fpred[i,j,l,k] = f.detach().cpu().item()
                    factual[i,j,l,k] = fgt.detach().cpu().item()
                    depth_error[i,j,l,k] = rel_errors.cpu().mean().item()

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


