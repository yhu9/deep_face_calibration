
import itertools
import argparse

import torch
import torch.optim

from logger import Logger
from model import CalibrationNet4
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
    if log:
        logger = Logger(logname)

    # define model, dataloader, 3dmm eigenvectors, optimization method
    model = CalibrationNet4()
    if modelin != "":
        model.load_state_dict(torch.load(modelin))
    model.cuda()

    data = dataloader.Data()
    loader = data.batchloader
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-1)

    # mean shape and eigenvectors for 3dmm
    mu_lm = torch.from_numpy(data.mu_lm).float()#.cuda()
    mu_lm[:,2] = mu_lm[:,2] * -1
    shape = mu_lm
    lm_eigenvec = torch.from_numpy(data.lm_eigenvec).float()#.cuda()

    #exp_eigenvec = torch.from_numpy(data.exp_eigenvec).float()
    M = data.M
    N = data.N

    # main training loop
    for epoch in itertools.count():
        for j,batch in enumerate(loader):
            # get input and gt values
            optimizer.zero_grad()
            x_cam_gt = batch['x_cam_gt']
            x_w_gt = batch['x_w_gt']
            f_gt = batch['f_gt']
            beta_gt = batch['beta_gt']

            x_img = batch['x_img']
            x_img_gt = batch['x_img']
            batch_size = f_gt.shape[0]
            x_img_pts = x_img.reshape((batch_size,M,N,2)).permute(0,1,3,2)

            one = torch.ones(batch_size,M*N,1)
            x_img_one = torch.cat([x_img,one],dim=2)

            # run the model
            out = model(x_img_one.permute(0,2,1))

            # evaluate the extrinsics
            f_error = []
            reconstruction_error = []
            reprojection_error = []
            for i in range(batch_size):
                betas = out[i,:199]
                f = torch.relu(out[i,199])
                K = torch.zeros((3,3)).float()
                K[0,0] = f;
                K[1,1] = f;
                K[2,2] = 1;
                K[0,2] = 320;
                K[1,2] = 240;

                # apply 3DMM model from predicted parameters
                alpha_matrix = torch.diag(betas)
                shape_cov = torch.mm(lm_eigenvec,alpha_matrix)
                s = shape_cov.sum(1).view(68,3)
                #shape = (mu_lm + s)
                #shape = mu_lm

                # run epnpp using predicted shape and intrinsics
                Xc,R,T = util.EPnP(x_img_pts[i],shape,K)

                # objective
                reproj_error2 = util.getReprojError2(x_img_pts[i],shape,R,T,K)
                reproj_error3 = util.getReprojError3(x_cam_gt[i],shape,R,T)
                rel_errors = util.getRelReprojError3(x_cam_gt[i],shape,R,T)

                reprojection_error.append(reproj_error2.mean())
                #reconstruction_error.append(reproj_error3.mean())
                reconstruction_error.append(rel_errors.mean())

                # beta error
                beta_error = torch.mean(torch.abs(betas - beta_gt))

                # focal length error
                #f_error.append(torch.abs(f_gt[i] - f))
                f_error.append(torch.abs(f_gt[i] - f) / f_gt[i])

            # get loss
            error_2d = torch.mean(torch.stack(reprojection_error))
            error_3d = torch.mean(torch.stack(reconstruction_error))
            error_f = torch.mean(torch.stack(f_error))
            loss = error_f

            # weight update
            #loss = f_error + reconstruction_error
            #loss = f_error*0.1  + beta_error + reproj_error3_n1.mean()*0.1
            #loss = f_error + reconstruction_error*0.1 + meanfeat_loss*0.1 + beta_error*0.1
            #loss = f_error + reconstruction_error + meanfeat_loss*0.1 + beta_error

            #loss = reconstruction_error + f_error + meanfeat_loss           # base2
            loss.backward()
            optimizer.step()

            #LOG THE SUMMARIES
            if log:
                logger.scalar_summary({'rec_error': reconstruction_error.item(), 'f_error': f_error.item(),'rep_error': reproj_error.item()})
                #logger.scalar_summary({'rec_error': reconstruction_error.item(),'rep_error':reproj_error.item(), 'f_error': f_error.item(), 'beta_error': beta_error.item()})
                logger.incStep()

            print(f"epoch/batch {epoch}/{j} | loss: {loss.item():.4f} | beta_error: {beta_error.item():.4f} | rec: {error_3d.item():.3f} | rep: {error_2d.item():.3f} | f_error: {error_f.item():.3f} | fgt/f: {f_gt[-1].item():.2f}/{f.item():.2f}")

            if j == 1000: break
        print("saving!")
        torch.save(model.state_dict(), modelout)


####################################################################################3
if __name__ == '__main__':
    train()

