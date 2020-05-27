
import itertools
import argparse

import torch
import torch.optim

from logger import Logger
from model import PointNet
from model import RNN
from model import feature_transform_regularizer
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
    model = PointNet(k=1+199,feature_transform=False)
    if modelin != "":
        model.load_state_dict(torch.load(modelin))
    model#.cuda()

    data = dataloader.SyntheticLoader()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)

    # mean shape and eigenvectors for 3dmm
    mu_lm = torch.from_numpy(data.mu_lm).float()#.cuda()
    lm_eigenvec = torch.from_numpy(data.lm_eigenvec).float()#.cuda()
    #exp_eigenvec = torch.from_numpy(data.exp_eigenvec).float()

    M = data.M

    # main training loop
    for epoch in itertools.count():
        for i in range(len(data)):
            sequence = data[i]

            # get input and gt values
            optimizer.zero_grad()
            x_cam_gt = sequence['x_cam_gt']#.cuda()
            x_w_gt = sequence['x_w_gt']#.cuda()
            f_gt = sequence['f_gt']#.cuda()
            beta_gt = sequence['beta_gt']#.cuda()

            x_img = sequence['x_img']#.cuda()
            #x_img = sequence['x_img_gt'].cuda()
            one  = torch.ones(100,1,68)#.cuda()
            x_img_one = torch.cat([x_img,one],dim=1)

            # run the model
            out, trans, transfeat = model(x_img_one)
            #feattransform_loss = feature_transform_regularizer(transfeat)*0.001
            betas = out[:,:199].mean(0)
            f = torch.mean(torch.relu(out[:,199]))

            # setup intrinsic matrix
            K = torch.zeros((3,3)).float()#.cuda()
            K[0,0] = f;
            K[1,1] = f;
            K[2,2] = 1;
            K[0,2] = 320;
            K[1,2] = 240;

            # create 3DMM model from predicted parameters
            alpha_matrix = torch.diag(betas)
            shape_cov = torch.mm(lm_eigenvec,alpha_matrix)
            s = shape_cov.sum(1).view(68,3)
            shape = mu_lm + s
            shape[:,-1] = shape[:,-1] * -1

            # run epnpp using predicted shape and intrinsics
            Xc,R,T = util.EPnP(x_img,shape,K)

            # objective
            reproj_error2 = util.getReprojError2(x_img,shape,R,T,K)
            reproj_error3 = util.getReprojError3(x_cam_gt,shape,R,T)
            rel_errors = util.getRelReprojError3(x_cam_gt,shape,R,T)

            reproj_error = reproj_error2.mean()
            reconstruction_error = rel_errors.mean()

            # beta error
            beta_error = torch.mean(torch.abs(betas - beta_gt))

            # focal length error
            #f_error = torch.abs(f_gt - f) / f_gt
            f_error = torch.abs(f_gt - f)

            # error in standard deviation of each frame
            meanfeat_loss = torch.mean(torch.abs(out[:,199] - f))

            # weight update
            #loss = f_error + reconstruction_error
            #loss = f_error*0.01 + reconstruction_error*0.01 + beta_error
            # loss = f_error*0.01 + beta_error + reconstruction_error*0.01
            # loss = f_error*0.01 + beta_error + reproj_error+ reconstruction_error*0.01
            # loss = f_error + reconstruction_error + feattransform_loss
            #loss = f_error*0.01 + reconstruction_error*0.01 + beta_error
            loss = reconstruction_error*10 + beta_error + meanfeat_loss * 0.001
            loss.backward()
            optimizer.step()

            #LOG THE SUMMARIES
            if log:
                logger.scalar_summary({'degeneracy': ratio.item()})
                logger.scalar_summary({'rec_error': reconstruction_error.item(),'rep_error':reproj_error.item(), 'f_error': f_error.item(), 'beta_error': beta_error.item(), 'meanfeat':meanfeat_loss.item()})
                logger.incStep()

            print(f"epoch/sequence {epoch}/{i}  |   Loss: {loss.item():.4f} | rec: {reconstruction_error.item():.4f}  | rep: {reproj_error.item():.4f} | f_error: {f_error.item():.4f} | fgt/f: {f_gt.item():.2f}/{f.item():.2f}   | beta_error: {beta_error.item():.4f} | meafeat: {meanfeat_loss.item():.4f}")

        print("saving!")
        torch.save(model.state_dict(), modelout)


####################################################################################3
if __name__ == '__main__':
    train()

