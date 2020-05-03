
import itertools
import argparse

import torch
import torch.optim


from logger import Logger
from model import PointNet
import dataloader
import util

####################################################

def train():
    # define logger
    logger = Logger('./log')

    # define model, dataloader, 3dmm eigenvectors, optimization method
    model = PointNet(k=1+199)
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
        for i in range(len(data)):
            batch = data[i]

            # get input and gt values
            optimizer.zero_grad()
            x_w_gt = batch['x_w_gt'].cuda()
            f_gt = batch['f_gt'].cuda()
            K_gt = batch['K_gt']
            # x_img = batch['x_img'].cuda()
            x_img = batch['x_img_gt'].cuda()
            one  = torch.ones(100,1,68).cuda()
            x_img = torch.cat([x_img,one],dim=1)


            # run the model
            out, trans, transfeat = model(x_img)
            alphas = out[:,:199].mean(0)
            f = out[:,199].mean(0)

            # apply 3DMM model from predicted parameters
            alpha_matrix = torch.diag(alphas)
            shape_cov = torch.mm(lm_eigenvec,alpha_matrix)
            s = shape_cov.sum(1).view(68,3)
            shape = mu_lm + s

            # 3d reconstruction error
            reconstruction_error = torch.mean(torch.norm(shape - x_w_gt,p=2,dim=1))

            # focal length error
            f_error = torch.abs(f_gt - f)

            # weight update
            loss = reconstruction_error + f_error
            loss.backward()
            optimizer.step()

            #LOG THE SUMMARIES
            logger.scalar_summary({'rec_error': reconstruction_error.item(), 'f_error': f_error.item()},i)

            #print(f"epoch/batch {epoch}/{i}  |   Loss: {loss:.4f} | reproj: {reproj_error:.4f} ")
            print(f"epoch/batch {epoch}/{i}  |   Loss: {loss.item():.4f} | rec: {reconstruction_error.item():.4f}  | f: {f_error.item():.4f}")
        print("saving!")
        torch.save(model.state_dict(), f"model/chkpoint_{epoch:03d}.pt")


####################################################################################3
if __name__ == '__main__':
    train()

