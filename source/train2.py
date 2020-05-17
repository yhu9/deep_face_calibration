
import itertools
import argparse

import torch
import torch.optim
from torch.utils.data import Dataset, DataLoader

from logger import Logger
from model import CalibrationNet
from model import RNN
import dataloader
import util

####################################################

parser = argparse.ArgumentParser(description="training arguments")
parser.add_argument("--out", default="model/model.pt")
parser.add_argument("--model", default="")
parser.add_argument("--log",type=int,default=1)
args = parser.parse_args()

####################################################

def train(modelin=args.model, modelout=args.out,log=args.log):
    # define logger
    if log:
        logger = Logger('./log')

    # define model, dataloader, 3dmm eigenvectors, optimization method
    model = RNN()
    if modelin != "":
        model.load_state_dict(torch.load(modelin))
    model.cuda()

    data = dataloader.SyntheticLoader()
    #batchsize=1
    #seq_len=100
    #batchloader = DataLoader(data,
    #        batch_size=batchsize,
    #        shuffle=True,
    #        num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)

    # mean shape and eigenvectors for 3dmm
    mu_lm = torch.from_numpy(data.mu_lm).float().cuda()
    lm_eigenvec = torch.from_numpy(data.lm_eigenvec).float().cuda()
    #exp_eigenvec = torch.from_numpy(data.exp_eigenvec).float()
    #batch_lmvec = torch.stack(batchsize*[lm_eigenvec])

    M = data.M

    # main training loop
    for epoch in itertools.count():
        for i,batch in enumerate(data):
            if i == 1000: break

            # get input and gt values
            optimizer.zero_grad()
            beta_gt = batch['beta_gt'].cuda()
            x_cam_gt = batch['x_cam_gt'].cuda()
            x_w_gt = batch['x_w_gt'].cuda()
            f_gt = batch['f_gt'].cuda()
            x_img = batch['x_img'].cuda()
            xin = x_img.reshape((100,136)).unsqueeze(0)

            #x_img = batch['x_img_gt'].cuda()
            #one  = torch.ones(100,1,68).cuda()
            #x_img_one = torch.cat([x_img,one],dim=1)

            # run the model
            out = model(xin).squeeze()

            betas = out[:199]
            f = torch.relu(out[199])
            K = torch.zeros((3,3)).float().cuda()
            K[0,0] = f;
            K[1,1] = f;
            K[2,2] = 1;
            K[0,2] = 320;
            K[1,2] = 240;

            # apply 3DMM model from predicted parameters
            alpha_matrix = torch.diag(betas)
            shape_cov = torch.mm(lm_eigenvec,alpha_matrix)
            s = shape_cov.sum(1).view((68,3))
            shape = (mu_lm + s)

            # run epnp algorithm"
            # get control points
            c_w = util.getControlPoints(shape)

            # solve alphas
            alphas = util.solveAlphas(shape,c_w)

            # setup M
            px = 320;
            py = 240;

            Matrix = util.setupM(alphas,x_img.permute(0,2,1),px,py,f)

            # get eigenvectors of M
            u,d,v = torch.svd(Matrix)

            #solve N=1
            c_c_n1 = v[:,:,-1].reshape((100,4,3)).permute(0,2,1)
            _ , x_c_n1, _ = util.scaleControlPoints(c_c_n1,c_w[:3,:],alphas,shape)
            Rn1,Tn1 = util.getExtrinsics(x_c_n1,shape)
            reproj_error2_n1 = util.getReprojError2(x_img,shape,Rn1,Tn1,K)
            reproj_error3_n1 = util.getReprojError3(x_cam_gt,shape,Rn1,Tn1)

            '''
            # solve N=2
            # get distance contraints
            d12,d13,d14,d23,d24,d34 = util.getDistances(c_w)
            distances = torch.stack([d12,d13,d14,d23,d24,d34])**2
            beta_n2 = util.getBetaN2(v[:,:,-2:],distances)
            c_c_n2 = util.getControlPointsN2(v[:,:,-2:],beta_n2)
            _,x_c_n2,_ = util.scaleControlPoints(c_c_n2,c_w[:3,:],alphas,shape)
            Rn2,Tn2 = util.getExtrinsics(x_c_n2,shape)
            reproj_error2_n2 = util.getReprojError2(x_img,shape,Rn2,Tn2,K)
            reproj_error3_n2 = util.getReprojError3(x_cam_gt,shape,Rn2,Tn2)

            # objective
            mask = reproj_error2_n1 < reproj_error2_n2
            reproj_error = torch.cat((reproj_error2_n1[mask],reproj_error2_n2[~mask])).mean()
            reconstruction_error = torch.cat((reproj_error3_n1[mask],reproj_error3_n2[~mask])).mean()
            '''

            # reproj error
            reproj_error2 = torch.mean(reproj_error2_n1)
            reconstruction_error = torch.mean(reproj_error3_n1)

            # beta error
            beta_error = torch.mean(torch.abs(betas - beta_gt))

            # focal length error
            f_error = torch.abs(f_gt - f)

            # weight update
            #loss = f_error + reconstruction_error
            loss = f_error*0.01 + beta_error
            loss.backward()
            optimizer.step()

            #LOG THE SUMMARIES
            if log:
                logger.scalar_summary({'rec_error': reconstruction_error.item(), 'f_error': f_error.item(),'rep_error': reproj_error2.item()})
                #logger.scalar_summary({'rec_error': reconstruction_error.item(), 'f_error': f_error.item()})
                logger.incStep()

            #print(f"epoch/batch {epoch}/{i}  |   Loss: {loss:.4f} | reproj: {reproj_error:.4f} ")
            #print(f"epoch/batch {epoch}/{i}  |   Loss: {loss.item():.4f} | rec: {reconstruction_error.item():.4f}  | f_error: {f_error.item():.4f} | fgt/f: {f_gt.item():.2f}/{f.item():.2f}")
            print(f"epoch/batch {epoch}/{i}  |   Loss: {loss.item():.4f} | beta_error: {beta_error.item():.4f} | f_error: {f_error.item():.4f} | fgt/f: {f_gt.item():.2f}/{f.item():.2f}")
        print("saving!")
        torch.save(model.state_dict(), modelout)


####################################################################################3
if __name__ == '__main__':
    train()
