
import itertools
import argparse
import os

import torch
import torch.optim

from model2 import PointNet
from test6 import test
import dataloader
import util

####################################################

parser = argparse.ArgumentParser(description="training arguments")
parser.add_argument("--out", default="model.pt")
parser.add_argument("--model", default="")
parser.add_argument("--device",default='cpu')
parser.add_argument("--opt",default=False, action="store_true")
parser.add_argument("--ft",default=False, action="store_true")
args = parser.parse_args()

####################################################

# stack list of videos
# input:
#   1. list of dictionaries output by loader
# output:
#   2. single dictionary for all videos
def stackVideos(vids,M,N,device):

    video = {}
    x_cam_gt = []
    x_w_gt = []
    f_gt = []
    x_img = []
    alpha_gt = []
    for v in vids:
        # get the input and gt values
        alpha_gt.append(v['alpha_gt'].to(device=device).squeeze())
        x_cam_gt.append(v['x_cam_gt'].to(device=device))
        x_w_gt.append(v['x_w_gt'].to(device=device))
        f_gt.append(v['f_gt'].to(device=device))
        x_img.append(v['x_img'].reshape(M,N,2).permute(0,2,1).to(device=device))

    video['alpha'] = torch.stack(alpha_gt)
    video['x_cam_gt'] = torch.cat(x_cam_gt)
    video['x_w_gt'] = torch.stack(x_w_gt)
    video['f_gt'] = torch.stack(f_gt)
    video['x_img'] = torch.cat(x_img)

    return video

def train(modelin=args.model, modelout=args.out,device=args.device,opt=args.opt,ft=args.ft):

    # define model, dataloader, 3dmm eigenvectors, optimization method
    calib_net = PointNet(n=1,feature_transform=ft)
    sfm_net = PointNet(n=199,feature_transform=ft)
    if modelin != "":
        calib_path = os.path.join('model','calib_' + modelin)
        sfm_path = os.path.join('model','sfm_' + modelin)
        pretrained1 = torch.load(calib_path)
        pretrained2 = torch.load(sfm_path)
        calib_dict = calib_net.state_dict()
        sfm_dict = sfm_net.state_dict()

        pretrained1 = {k: v for k,v in pretrained1.items() if k in calib_dict}
        pretrained2 = {k: v for k,v in pretrained2.items() if k in sfm_dict}
        calib_dict.update(pretrained1)
        sfm_dict.update(pretrained2)

        calib_net.load_state_dict(pretrained1)
        sfm_net.load_state_dict(pretrained2)

    calib_net.to(device=device)
    sfm_net.to(device=device)
    opt1 = torch.optim.Adam(calib_net.parameters(),lr=1e-3)
    opt2 = torch.optim.Adam(sfm_net.parameters(),lr=1e-3)

    # dataloader
    loader = dataloader.SyntheticLoader()
    batch_size = 100
    M = loader.M
    N = loader.N

    # mean shape and eigenvectors for 3dmm
    mu_lm = torch.from_numpy(loader.mu_lm).float()#.to(device=device)
    mu_lm[:,2] = mu_lm[:,2] * -1
    mu_lm = torch.stack(300 * [mu_lm.to(device=device)])
    shape = mu_lm
    lm_eigenvec = torch.from_numpy(loader.lm_eigenvec).float().to(device=device)
    sigma = torch.from_numpy(loader.sigma).float().detach().to(device=device)
    sigma = torch.diag(sigma.squeeze())
    lm_eigenvec = torch.mm(lm_eigenvec, sigma)
    lm_eigenvec = torch.stack(300 * [lm_eigenvec])

    # main training loop
    best = 10000
    for epoch in itertools.count():
        for i in range(len(loader)):
            if i < 3: continue
            v1 = loader[i]
            v2 = loader[i-1]
            v3 = loader[i-2]
            batch = stackVideos([v1,v2,v3],100,68,device=device)

            # get the input and gt values
            alpha_gt = batch['alpha']
            x_cam_gt = batch['x_cam_gt']
            shape_gt = batch['x_w_gt']
            fgt = batch['f_gt']
            x = batch['x_img']

            M = x.shape[0]
            N = x.shape[-1]

            # calibration
            f = torch.squeeze(calib_net(x) + 300)
            K = torch.zeros((M,3,3)).float().to(device=device)
            K[:,0,0] = f
            K[:,1,1] = f
            K[:,2,2] = 1

            # sfm
            alpha = sfm_net(x)
            alpha = alpha.unsqueeze(-1)
            shape = mu_lm + torch.bmm(lm_eigenvec,alpha).squeeze().view(M,N,3)
            shape[0:100] = shape[0:100] - shape[0:100].mean(1).unsqueeze(1)
            shape[100:200] = shape[100:200] - shape[100:200].mean(1).unsqueeze(1)
            shape[200:300] = shape[200:300] - shape[200:300].mean(1).unsqueeze(1)

            opt1.zero_grad()
            opt2.zero_grad()

            f1_error = torch.mean(torch.abs(f[0:100] - fgt[0]))
            f2_error = torch.mean(torch.abs(f[100:200] - fgt[1]))
            f3_error = torch.mean(torch.abs(f[200:300] - fgt[2]))

            #a1_error = torch.mean(torch.abs(alpha[0:100] - alpha_gt[0]))
            #a2_error = torch.mean(torch.abs(alpha[100:200] - alpha_gt[1]))
            #a3_error = torch.mean(torch.abs(alpha[200:300] - alpha_gt[2]))

            s1_error = torch.mean(torch.abs(shape[0:100] - shape_gt[0].unsqueeze(0)))
            s2_error = torch.mean(torch.abs(shape[100:200] - shape_gt[1].unsqueeze(0)))
            s3_error = torch.mean(torch.abs(shape[200:300] - shape_gt[2].unsqueeze(0)))

            ferror = f1_error + f2_error + f3_error
            #aerror = a1_error + a2_error + a3_error
            serror = s1_error + s2_error + s3_error

            #f_error = torch.mean(torch.abs(f - fgt))
            #error3d = torch.mean(torch.norm(shape - shape_gt,dim=2))
            #error = ferror + aerror
            error = ferror + serror
            error.backward()
            opt1.step()
            opt2.step()

            print(f"iter: {i} | best: {best:.2f} | f_error: {ferror.item():.3f} | serror: {serror.item():.3f} ")
            if i == 1000: break

        # save model and increment weight decay
        torch.save(sfm_net.state_dict(), os.path.join('model','sfm_model.pt'))
        torch.save(calib_net.state_dict(), os.path.join('model','calib_model.pt'))
        ferror = test(modelin='model.pt',outfile=args.out,optimize=False)
        if ferror < best:
            best = ferror
            print("saving!")
            torch.save(sfm_net.state_dict(), os.path.join('model','sfm_'+modelout))
            torch.save(calib_net.state_dict(), os.path.join('model','calib_'+modelout))

        sfm_net.train()
        calib_net.train()
        #decay.step()

####################################################################################3
if __name__ == '__main__':
    train()

