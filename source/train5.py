
import itertools
import argparse

import torch
import torch.optim

from logger import Logger
from model import CalibrationNet4
from model import feature_transform_regularizer
from model import PointNet
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
    pointnet = PointNet(k=128,feature_transform=False)
    model = CalibrationNet4()
    if modelin != "":
        model.load_state_dict(torch.load(modelin))
    model.cuda()
    pointnet.cuda()
    opt2 = torch.optim.Adam(pointnet.parameters(),lr=1e-1)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-1)
    decay = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

    # dataloader
    data = dataloader.Data()
    loader = data.batchloader

    # mean shape and eigenvectors for 3dmm
    mu_lm = torch.from_numpy(data.mu_lm).float()#.cuda()
    mu_lm[:,2] = mu_lm[:,2] * -1
    shape = mu_lm
    lm_eigenvec = torch.from_numpy(data.lm_eigenvec).float()#.cuda()

    M = data.M
    N = data.N

    # main training loop
    for epoch in itertools.count():
        for j,batch in enumerate(loader):

            optimizer.zero_grad()

            # get the input and gt values
            x_cam_gt = batch['x_cam_gt'].cuda()
            x_w_gt = batch['x_w_gt'].cuda()
            fgt = batch['f_gt'].cuda()
            beta_gt = batch['beta_gt'].cuda()
            x_img = batch['x_img'].cuda()
            #x_img_norm = batch['x_img_norm']
            x_img_gt = batch['x_img_gt'].cuda()
            batch_size = fgt.shape[0]
            x_img_pts = x_img.reshape((batch_size,M,N,2)).permute(0,1,3,2)

            x = x_img.reshape((batch_size,M,N,2)).permute(0,1,3,2) / 640
            ones = torch.ones(batch_size,M,1,N).cuda()
            x_one = torch.cat([x,ones],dim=2)

            one = torch.ones(batch_size,M*N,1).cuda()
            x_img_one = torch.cat([x_img,one],dim=2)
            #x_cam_pt = x_cam_gt.permute(0,1,3,2).reshape(batch_size,6800,3)

            # run the model
            features = []
            for i in range(batch_size):
                feat,_,_ = pointnet(x_one[i])
                features.append(feat)
            tmp = torch.stack(features).unsqueeze(1)
            out = model(tmp)
            #print(tmp[0])
            betas = out[:,:199]
            fout = torch.relu(out[:,199])
            if torch.any(fout < 1): fout = fout+1
            #fout = fgt.flatten()
            #fout = fout + torch.rand(fout.shape).cuda()*300
            #print(fout)
            #print(fgt.flatten())

            # setup intrinsic matrix
            K = torch.zeros((batch_size,3,3)).float().cuda()
            K[:,0,0] = fout
            K[:,1,1] = fout
            K[:,2,2] = 1

            # setup inverse of intrinsic matrix
            kinv = torch.zeros((batch_size,3,3)).float().cuda()
            kinv[:,0,0] = 1/fout
            kinv[:,1,1] = 1/fout
            kinv[:,2,2] = 1

            # get reconstruction error
            x_cam_gt = x_cam_gt.permute(0,1,3,2).reshape((batch_size,M*N,3))
            x_cam_gt = x_cam_gt.permute(0,2,1)
            error_3d = util.getPCError(x_cam_gt,x_img_one,kinv,mode='l1')

            # get reprojection error
            proj = torch.bmm(K,x_cam_gt)
            proj = proj / proj[:,2,:].unsqueeze(1)
            error_2d = torch.mean(torch.abs(proj - x_img_one.permute(0,2,1)))

            # get beta error
            beta_error = torch.mean(torch.abs(betas - beta_gt))

            #error_f = torch.mean(torch.abs(fout - f_gt.squeeze()) / f_gt.squeeze())
            #error_f = torch.mean((fout - f_gt.squeeze())**2 / f_gt.squeeze()**2)
            error_f = torch.mean(torch.abs(fout - fgt.squeeze()))
            #error_f = torch.nn.functional.mse_loss(fout.unsqueeze(1),f_gt)

            # get loss
            #error_2d = torch.mean(torch.stack(reprojection_error))
            #error_3d = torch.mean(torch.stack(reconstruction_error))
            #loss = error_f + error_3d
            loss = error_3d
            #loss = error_3d + error_f
            #loss = error_3d*100 + error_f*0.01 # screen 2
            #loss = error_3d + error_f
            #loss = error_3d*100 + error_f*0.01 + error_2d #screen 1

            loss.backward()
            optimizer.step()
            opt2.step()
            #fout.retain_grad()
            #print(fout)
            #print(fgt)
            #print(fout.grad)
            #quit()

            #LOG THE SUMMARIES
            if log:
                logger.scalar_summary({'degeneracy': ratio.item()})
                logger.scalar_summary({'rec_error': reconstruction_error.item(),'rep_error':reproj_error.item(), 'f_error': f_error.item(), 'beta_error': beta_error.item(), 'meanfeat':meanfeat_loss.item()})
                logger.incStep()

            print(f"epoch/batch {epoch}/{j} | loss: {loss.item():.4f} | beta_error: {beta_error.item():.4f} | rec: {error_3d.item():.3f} | rep: {error_2d.item():.3f} | f_error: {error_f.item():.3f} | fgt/f: {fgt[-1].item():.2f}/{fout[-1].item():.2f}")

        # save model and increment weight decay
        print("saving!")
        torch.save(model.state_dict(), modelout)
        decay.step()


####################################################################################3
if __name__ == '__main__':
    train()

