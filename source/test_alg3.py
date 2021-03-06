
import itertools
import argparse

import scipy.io
import torch
import numpy as np

from logger import Logger
from model import Model1
import dataloader
import util

####################################################

parser = argparse.ArgumentParser(description="training arguments")
parser.add_argument("--model", default="model/model_ximgnoisy.pt")
parser.add_argument("--out",default="results/exp.mat")
parser.add_argument("--feat_trans", default=False, action='store_true')
args = parser.parse_args()

####################################################

def testBIWI(model,modelin=args.model,outfile=args.out,feature_transform=args.feat_trans):
    if modelin != "":
        model.load_state_dict(torch.load(modelin))
    model.eval()

    # load 3dmm data
    data3dmm = dataloader.SyntheticLoader()
    mu_lm = torch.from_numpy(data3dmm.mu_lm).float()
    lm_eigenvec = torch.from_numpy(data3dmm.lm_eigenvec).float()
    shape = mu_lm
    shape[:,2] = shape[:,2] * -1

    loader = dataloader.BIWILoader()
    seterror_3d = []
    seterror_rel3d = []
    seterror_relf = []
    seterror_2d = []
    for sub in range(len(loader)):
        batch = loader[sub]

        x_cam_gt = batch['x_cam_gt']
        x_w_gt = batch['x_w_gt']
        f_gt = batch['f_gt']
        x_img = batch['x_img']
        x_img_gt = batch['x_img_gt']
        M = x_img_gt.shape[0]

        one  = torch.ones(M,1,68)
        x_img_one = torch.cat([x_img,one],dim=1)

        # run the model
        out, trans, transfeat = model(x_img_one)
        alphas = out[:,:199].mean(0)
        f = torch.relu(out[:,199]).mean()
        K = torch.zeros((3,3)).float()
        K[0,0] = f;
        K[1,1] = f;
        K[2,2] = 1;
        K[0,2] = 320;
        K[1,2] = 240;
        Xc,R,T = util.EPnP(x_img,shape,K)

        # apply 3DMM model from predicted parameters
        reproj_errors2 = util.getReprojError2(x_img,shape,R,T,K)
        reproj_errors3 = util.getReprojError3(x_cam_gt,shape,R,T)
        rel_errors = util.getRelReprojError3(x_cam_gt,shape,R,T)

        reproj_error = reproj_errors2.mean()
        reconstruction_error = reproj_errors3.mean()
        rel_error = rel_errors.mean()
        f_error = torch.abs(f_gt - f) / f_gt

        seterror_2d.append(reproj_error.cpu().data.item())
        seterror_3d.append(reconstruction_error.cpu().data.item())
        seterror_rel3d.append(rel_error.cpu().data.item())
        seterror_relf.append(f_error.cpu().data.item())

        print(f"fgt: {f_gt.mean().item():.3f}  | f_error_rel: {f_error.item():.4f}  | rmse: {reconstruction_error.item():.4f}  | rel rmse: {rel_error.item():.4f}    | 2d error: {reproj_error.item():.4f}")
        #end for

    matdata = {}
    matdata['seterror_2d'] = np.array(seterror_2d)
    matdata['seterror_3d'] = np.array(seterror_3d)
    matdata['seterror_rel3d'] = np.array(seterror_rel3d)
    matdata['seterror_relf'] = np.array(seterror_relf)
    scipy.io.savemat(outfile,matdata)

    print(f"MEAN seterror_2d: {np.mean(seterror_2d)}")
    print(f"MEAN seterror_3d: {np.mean(seterror_3d)}")
    print(f"MEAN seterror_rel3d: {np.mean(seterror_rel3d)}")
    print(f"MEAN seterror_relf: {np.mean(seterror_relf)}")

def test(modelin=args.model,outfile=args.out,feature_transform=args.feat_trans):

    # define model, dataloader, 3dmm eigenvectors, optimization method
    #if modelin != "":
    #    model.load_state_dict(torch.load(modelin))
    #model.eval()
    #model.cuda()

    # mean shape and eigenvectors for 3dmm
    M = 100
    N = 68
    data3dmm = dataloader.SyntheticLoader()
    mu_lm = torch.from_numpy(data3dmm.mu_lm).float()
    mu_lm[:,2] = mu_lm[:,2]*-1
    shape = mu_lm.detach()
    lm_eigenvec = torch.from_numpy(data3dmm.lm_eigenvec).float()
    #optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)

    # sample from f testing set
    allerror_2d = []
    allerror_3d = []
    allerror_rel3d = []
    allerror_relf = []
    all_f = []
    all_depth = []

    seterror_3d = []
    seterror_rel3d = []
    seterror_relf = []
    seterror_2d = []
    f_vals = [i*100 for i in range(4,21)]
    np.random.seed(0)
    for f_test in f_vals:
        f_test = 1200
        # create dataloader
        loader = dataloader.TestLoader(f_test)

        error_2d = []
        error_3d = []
        error_rel3d = []
        error_relf = []
        M = 100;
        N = 68;
        batch_size = 1;

        for j, data in enumerate(loader):
            # create a model and optimizer for it
            #model2 = Model1(k=199,feature_transform=False)
            #model2.apply(util.init_weights)
            model = Model1(k=1, feature_transform=False)
            model.apply(util.init_weights)
            optimizer = torch.optim.Adam(model.parameters(),lr=2e-1)

            #data = loader[67]
            x_cam_gt = data['x_cam_gt']
            shape = data['x_w_gt']
            fgt = data['f_gt']
            x_img = data['x_img']
            x_img_gt = data['x_img_gt']
            T_gt = data['T_gt']
            all_depth.append(np.mean(T_gt[:,2]))
            all_f.append(fgt.numpy()[0])

            x_img_pts = x_img.reshape((M,N,2)).permute(0,2,1)
            one = torch.ones(M*N,1)
            x_img_one = torch.cat([x_img,one],dim=1)
            x_cam_pt = x_cam_gt.permute(0,2,1).reshape(M*N,3)
            x = x_img_one.permute(1,0)

            ptsI = x_img.reshape((M,N,2)).permute(0,2,1)

            for iter in itertools.count():
                optimizer.zero_grad()

                #betas,_,_ = model2(x.unsqueeze(0))
                #shape = torch.sum(betas * lm_eigenvec,1)
                #shape = shape.reshape(68,3) + mu_lm

                f,_,_ = model(x.unsqueeze(0))
                #f = f + 300
                #f = (torch.nn.functional.tanh(f)+1)*850 + 300
                f = f + 300
                #f = torch.nn.functional.sigmoid(f)
                K = torch.zeros((3,3)).float()
                K[0,0] = f
                K[1,1] = f
                K[2,2] = 1

                # differentiable pose estimation
                km,c_w,scaled_betas, alphas = util.EPnP(ptsI,shape,K)
                Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
                error2d = util.getReprojError2(ptsI,shape,R,T,K,show=False,loss='l1')
                loss = error2d.mean()
                loss.backward()
                if torch.any(model.fc2.weight.grad != model.fc2.weight.grad):
                    print("oh oh something broke")
                    break
                optimizer.step()
                print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.item():.1f}/{fgt[0].item():.1f}")
                if iter == 200: break

            # get errors
            reproj_errors2 = util.getReprojError2(ptsI,shape,R,T,K)
            reproj_errors3 = util.getReprojError3(x_cam_gt,shape,R,T)
            rel_errors =  util.getRelReprojError3(x_cam_gt,shape,R,T)

            reproj_error = reproj_errors2.mean()
            reconstruction_error = reproj_errors3.mean()
            rel_error = rel_errors.mean()
            f_error = torch.abs(fgt - f) / fgt

            allerror_3d.append(reproj_error.data.numpy())
            allerror_2d.append(reconstruction_error.data.numpy())
            allerror_rel3d.append(rel_error.data.numpy())

            error_2d.append(reproj_error.cpu().data.item())
            error_3d.append(reconstruction_error.cpu().data.item())
            error_rel3d.append(rel_error.cpu().data.item())
            error_relf.append(f_error.cpu().data.item())

            print(f"f/sequence: {f_test}/{j}  | f/fgt: {f[0].item():.3f}/{fgt.item():.3f} |  f_error_rel: {f_error.item():.4f}  | rmse: {reconstruction_error.item():.4f}  | rel rmse: {rel_error.item():.4f}    | 2d error: {reproj_error.item():.4f}")
            #end for

        avg_2d = np.mean(error_2d)
        avg_rel3d = np.mean(error_rel3d)
        avg_3d = np.mean(error_3d)
        avg_relf = np.mean(error_relf)

        seterror_2d.append(avg_2d)
        seterror_3d.append(avg_3d)
        seterror_rel3d.append(avg_rel3d)
        seterror_relf.append(avg_relf)
        #end for
        break

    all_f = np.stack(all_f).flatten()
    all_d = np.stack(all_depth).flatten()
    allerror_2d = np.stack(allerror_2d).flatten()
    allerror_3d = np.stack(allerror_3d).flatten()
    allerror_rel3d = np.stack(allerror_rel3d).flatten()

    matdata = {}
    matdata['fvals'] = np.array(f_vals)
    matdata['all_f'] = np.array(all_f)
    matdata['all_d'] = np.array(all_depth)
    matdata['error_2d'] = allerror_2d
    matdata['error_3d'] = allerror_3d
    matdata['error_rel3d'] = allerror_rel3d
    matdata['seterror_2d'] = np.array(seterror_2d)
    matdata['seterror_3d'] = np.array(seterror_3d)
    matdata['seterror_rel3d'] = np.array(seterror_rel3d)
    matdata['seterror_relf'] = np.array(seterror_relf)
    scipy.io.savemat(outfile,matdata)

    print(f"MEAN seterror_2d: {np.mean(seterror_2d)}")
    print(f"MEAN seterror_3d: {np.mean(seterror_3d)}")
    print(f"MEAN seterror_rel3d: {np.mean(seterror_rel3d)}")
    print(f"MEAN seterror_relf: {np.mean(seterror_relf)}")
    #end function

####################################################################################3
if __name__ == '__main__':

    #model = Model1(k=1, feature_transform=args.feat_trans)

    test()
    #testBIWI(model)


