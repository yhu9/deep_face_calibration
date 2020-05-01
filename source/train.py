
import itertools
import argparse

import torch
import torch.optim

from model import PointNet
import dataloader
import util

####################################################

def train():

    # define model, dataloader, 3dmm eigenvectors, optimization method
    model = PointNet(k=3+4+1+199)
    model.cuda()
    data = dataloader.SyntheticLoader()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

    # mean shape and eigenvectors for 3dmm
    mu_lm = torch.from_numpy(data.mu_lm).float().cuda()
    lm_eigenvec = torch.from_numpy(data.lm_eigenvec).float().cuda()
    #exp_eigenvec = torch.from_numpy(data.exp_eigenvec).float()

    M = data.M

    # main training loop
    for epoch in itertools.count():
        for i, batch in enumerate(data):

            # get input and gt values
            optimizer.zero_grad()
            x_w_gt = batch['x_w_gt'].cuda()
            x_cam_gt = batch['x_cam_gt'].cuda()
            x_img = batch['x_img'].cuda()
            x_img_gt = batch['x_img_gt'].cuda()
            f_gt = batch['f_gt'].cuda()
            K_gt = batch['K_gt']
            R_gt = batch['R_gt']
            Q_gt = batch['Q_gt'].cuda()
            one  = torch.ones(100,1,68).cuda()
            x_img = torch.cat([x_img,one],dim=1)

            # run the model
            out, trans, transfeat = model(x_img)
            alphas = out[:,:199].mean(0)
            f = out[:,199].mean(0)
            R = torch.tanh(out[:,200:204])*(3.14/4)
            tx = out[:,204]
            ty = out[:,205]
            tz = torch.relu(out[:,206])
            T = torch.stack([tx,ty,tz],1)

            # apply 3DMM model from predicted parameters
            alpha_matrix = torch.diag(alphas)
            shape_cov = torch.mm(lm_eigenvec,alpha_matrix)
            s = shape_cov.sum(1).view(68,3)
            shape = mu_lm + s

            # apply predicted projection onto camera space
            rx,ry,rz = util.quat2euler(R.T)
            rotm = util.R(rx,ry,rz)
            bshape = torch.stack(M*[shape.T])
            x_cam = torch.bmm(rotm,bshape) + T.unsqueeze(-1)

            # apply learned intrinsics to project onto image space
            K = torch.zeros(3,3).cuda()
            K[0,2] = 320
            K[1,2] = 240
            K[0,0] = f
            K[1,1] = f
            K[2,2] = 1
            bK = torch.stack(M*[K])
            proj = torch.bmm(bK,x_cam)
            img_proj = proj / (proj[:,2,:].unsqueeze(1) + 1)

            # 3dmm error
            torch.norm(x_w_gt - shape

            # 2d reprojection error
            reproj_error = torch.mean(torch.norm(img_proj[:,:2] - x_img_gt, p=2, dim=1))

            # 3d reconstruction error
            reconstruction_error = torch.mean(torch.norm(x_cam - x_cam_gt,p=2,dim=1)) * 0.01

            # weight update
            loss = reproj_error
            loss.backward()
            optimizer.step()

            #print(f"epoch/batch {epoch}/{i}  |   Loss: {loss:.4f} | reproj: {reproj_error:.4f} ")
            print(f"epoch/batch {epoch}/{i}  |   Loss: {loss:.4f} | reproj: {reproj_error:.4f}  | reconstruction: {reconstruction_error:.4f}")
        print("saving!")
        torch.save(model.state_dict(), f"model/chkpoint_{epoch:3d}.pt")


####################################################################################3
if __name__ == '__main__':
    train()

