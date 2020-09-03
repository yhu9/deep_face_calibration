
import itertools
import argparse
import os

import torch
import torch.optim

from model import Model1
from model import CalibrationNet
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
    calib_net = CalibrationNet(n=1)
    sfm_net = CalibrationNet(n=199)
    if modelin != "":
        calib_path = os.path.join('model','calib_' + modelin)
        sfm_path = os.path.join('model','sfm_' + modelin)
        calib_net.load_state_dict(torch.load(calib_path))
        sfm_net.load_state_dict(torch.load(sfm_path))
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
    lm_eigenvec = torch.from_numpy(data.lm_eigenvec).float().to(device=device).detach()
    sigma = torch.from_numpy(data.sigma).float().to(device=device).detach()
    sigma = torch.diag(sigma.squeeze())
    lm_eigenvec = torch.mm(lm_eigenvec, sigma)
    lm_eigenvec = torch.stack(batch_size * [lm_eigenvec])

    M = data.M
    N = data.N

    # main training loop
    for epoch in itertools.count():
        for j,batch in enumerate(loader):

            # get the input and gt values
            x_cam_gt = batch['x_cam_gt'].to(device=device)
            shape_gt = batch['x_w_gt'].to(device=device)
            fgt = batch['f_gt'].to(device=device)
            x_img = batch['x_img'].to(device=device)
            #beta_gt = batch['beta_gt'].to(device=device)
            #x_img_norm = batch['x_img_norm']
            #x_img_gt = batch['x_img_gt'].to(device=device)
            batch_size = fgt.shape[0]

            one = torch.ones(batch_size,M*N,1).to(device=device)
            x_img_one = torch.cat([x_img,one],dim=2)
            x_cam_pt = x_cam_gt.permute(0,1,3,2).reshape(batch_size,6800,3)
            x = x_img.permute(0,2,1).reshape(batch_size,2,M,N)

            ptsI = x_img_one.reshape(batch_size,M,N,3).permute(0,1,3,2)[:,:,:2,:]

            # if just optimizing
            if not opt:
                # calibration
                f = calib_net(x)
                f = f + 300
                K = torch.zeros((batch_size,3,3)).float().to(device=device)
                K[:,0,0] = f.squeeze()
                K[:,1,1] = f.squeeze()
                K[:,2,2] = 1

                # ground truth l1 error
                opt1.zero_grad()
                f_error = torch.mean(torch.abs(f - fgt))
                f_error.backward()
                opt1.step()

                # sfm
                betas = sfm_net(x)
                betas = betas.unsqueeze(-1)
                shape = mu_lm + torch.bmm(lm_eigenvec,betas).squeeze().view(batch_size,N,3)

                # ground truth shape error
                opt2.zero_grad()
                error3d = torch.mean(torch.abs(shape - shape_gt))
                error3d.backward()
                opt2.step()
                print(f"f_error: {f_error.item():.3f} | error3d: {error3d.item():.3f} | f/fgt: {f[0].item():.1f}/{fgt[0].item():.1f} | f/fgt: {f[1].item():.1f}/{fgt[1].item():.1f} | f/fgt: {f[2].item():.1f}/{fgt[2].item():.1f} | f/fgt: {f[3].item():.1f}/{fgt[3].item():.1f} ")
                continue

            # dual optimization
            for outerloop in itertools.count():
                # calibration
                shape = shape.detach()
                for iter in itertools.count():
                    opt1.zero_grad()
                    f = calib_net(x)
                    f = f + 300
                    K = torch.zeros((batch_size,3,3)).float().to(device=device)
                    K[:,0,0] = f.squeeze()
                    K[:,1,1] = f.squeeze()
                    K[:,2,2] = 1

                    # ground truth l1 error
                    f_error = torch.mean(torch.abs(f - fgt))

                    # differentiable PnP pose estimation
                    error1 = []
                    for i in range(batch_size):
                        km, c_w, scaled_betas, alphas = util.EPnP(ptsI[i],shape[i],K[i])
                        Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape[i],ptsI[i],K[i])
                        error2d = util.getReprojError2(ptsI[i],shape[i],R,T,K[i],show=False,loss='l1')
                        error1.append(error2d.mean())

                    # batched loss
                    #loss1 = torch.stack(error1).mean() + f_error
                    loss1 = f_error

                    # stopping condition
                    if iter > 10 and prev_loss < loss1: break
                    else: prev_loss = loss1

                    # optimize network
                    loss1.backward()
                    opt1.step()
                    print(f"iter: {iter} | error: {loss1.item():.3f} | f/fgt: {f[0].item():.1f}/{fgt[0].item():.1f} | f/fgt: {f[1].item():.1f}/{fgt[1].item():.1f} | f/fgt: {f[2].item():.1f}/{fgt[2].item():.1f} | f/fgt: {f[3].item():.1f}/{fgt[3].item():.1f} ")

                # structure from motion
                f = f.detach()
                for iter in itertools.count():
                    opt2.zero_grad()

                    betas = sfm_net(x)
                    betas = betas.unsqueeze(-1)
                    shape = mu_lm + torch.bmm(lm_eigenvec,betas).squeeze().view(batch_size,N,3)

                    K = torch.zeros((batch_size,3,3)).float().to(device=device)
                    K[:,0,0] = f.squeeze()
                    K[:,1,1] = f.squeeze()
                    K[:,2,2] = 1

                    # ground truth shape error
                    error3d = torch.mean(torch.abs(shape - shape_gt))

                    # differentiable PnP pose estimation
                    error2 = []
                    for i in range(batch_size):
                        km, c_w, scaled_betas, alphas = util.EPnP(ptsI[i],shape[i],K[i])
                        Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape[i],ptsI[i],K[i])
                        error2d = util.getReprojError2(ptsI[i],shape[i],R,T,K[i],show=False,loss='l1')
                        error2.append(error2d.mean())

                    # batched loss
                    loss2 = torch.stack(error2).mean() + error3d

                    # stopping condition
                    if iter > 10 and prev_loss < loss2: break
                    else: prev_loss = loss2

                    # optimize network
                    loss2.backward()
                    opt2.step()
                    print(f"iter: {iter} | error: {loss2.item():.3f} | f/fgt: {f[0].item():.1f}/{fgt[0].item():.1f}")

                # outerloop stopping condition
                if outerloop == 1: break

            # get errors
            #rmse = torch.mean(torch.abs(shape - shape_gt))
            #f_error = torch.mean(torch.abs(fgt - f) / fgt)

            # get shape error from image projection
            print(f"f/fgt: {f[0].item():.3f}/{fgt[0].item():.3f} | rmse: {rmse:.3f} | f_rel: {f_error.item():.4f}  | loss1: {loss1.item():.3f} | loss2: {loss2.item():.3f}")

        # save model and increment weight decay
        print("saving!")
        torch.save(sfm_net.state_dict(), os.path.join('model','sfm_'+modelout))
        torch.save(calib_net.state_dict(), os.path.join('model','calib_'+modelout))
        #decay.step()


####################################################################################3
if __name__ == '__main__':
    train()

