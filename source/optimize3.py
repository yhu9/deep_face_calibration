
import itertools
import argparse

import scipy.io
import torch
import numpy as np
from torch.autograd import Variable

from logger import Logger
from model import CalibrationNet3
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

# optimization for the focal length
def optimizef(xI,xW,f):

    # create optimization parameters
    varf = Variable(f,requires_grad=True)
    optimizer = torch.optim.Adam([varf],lr=1)

    # intitialize intial values
    minerror = 10000
    convergence = 0
    for iter in itertools.count():

        # create K and inverse of K
        K = torch.zeros((3,3))
        K[0,0] = varf
        K[1,1] = varf
        K[2,2] = 1

        kinv =  torch.zeros(3,3).float()
        kinv[0,0] = 1/varf
        kinv[1,1] = 1/varf
        kinv[2,2] = 1

        # get error
        #reproj_errors2 = util.getReprojError2(sequence,shape,R,T,K)
        #reproj_errors3 = util.getReprojError3(x_cam_gt,shape,varR,varT)
        #error_3d = util.getRelReprojError3(x_cam_gt,shape,R,T).mean()
        #error_Rconsistency = util.getRConsistency(R)
        #error_Tconsistency = util.getTConsistency(T)*0.001
        #error_3dconsistency = util.get3DConsistency(sequence,shape,kinv,R,T)

        Xc,R,T = util.EPnP(xI,xW,K)
        optimizer.zero_grad()

        error_3dconsistency = util.get3DConsistency(xI,xW,kinv,R,T)
        loss = error_3dconsistency

        # determine optimization convergence
        if loss < minerror:
            minerror = loss
            optf = varf.item()
            optR = R
            optT = T
            convergence = 0
        else:
            convergence += 1

        loss.backward()
        optimizer.step()
        delta = K[0,0] - varf
        direction = torch.sign(delta)

        print(f"iter: {iter} | loss: {loss.item():.3f} | f: {f.item():.3f} |  error 3d: {error_3dconsistency.item():.3f} | delta: {delta.item():.3f}")

    return optf,optR,optT

# optimization for the extrinsics
def optimizeExtrinsics(model):

    return optR,optT

# optimization for the rotation translation

def test(model, modelin=args.model,outfile=args.out,feature_transform=args.feat_trans):

    # define model, dataloader, 3dmm eigenvectors, optimization method
    if modelin != "":
        model.load_state_dict(torch.load(modelin))
    model.eval()

    # mean shape and eigenvectors for 3dmm
    M = 100
    data3dmm = dataloader.SyntheticLoader()
    mu_lm = torch.from_numpy(data3dmm.mu_lm).float()
    lm_eigenvec = torch.from_numpy(data3dmm.lm_eigenvec).float()
    shape = mu_lm

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
    for f_test in f_vals:
        # create dataloader
        data = dataloader.TestLoader(f_test)

        error_2d = []
        error_3d = []
        error_rel3d = []
        error_relf = []
        M = 100;
        N = 68;
        batch_size = 1;

        for k in range(len(data)):
            batch = data[4]
            x_cam_gt = batch['x_cam_gt']
            x_w_gt = batch['x_w_gt']
            f_gt = batch['f_gt']
            x_img = batch['x_img'].unsqueeze(0)
            x_img_gt = batch['x_img_gt']
            T_gt = batch['T_gt']
            sequence = batch['x_img'].reshape((M,N,2)).permute(0,2,1)

            all_depth.append(np.mean(T_gt[:,2]))
            all_f.append(f_gt.numpy()[0])

            x = x_img.reshape((batch_size,M,N,2)).permute(0,3,2,1) / 640
            x_one = torch.cat([x.squeeze().permute(2,0,1)*640,torch.ones(M,1,N)],dim=1)

            # run the model
            out = model(x)
            betas = out[:,:199]
            fout = torch.relu(out[:,199])
            if torch.any(fout < 1): fout = fout+1

            # apply 3DMM model from predicted parameters
            alpha_matrix = torch.diag(betas.squeeze())
            shape_cov = torch.mm(lm_eigenvec,alpha_matrix)
            s = shape_cov.sum(1).view(68,3)
            #shape = (mu_lm + s)
            #shape = mu_lm
            #shape[:,2] = shape[:,2]*-1

            # create variables and optimizer for variables as SGD
            # run epnp using predicted shape and intrinsics
            varf = Variable(fout,requires_grad=True)
            K = torch.zeros((3,3))
            K[0,0] = varf;
            K[1,1] = varf;
            K[2,2] = 1;
            K[0,2] = 0;
            K[1,2] = 0;
            Xc,R,T = util.EPnP(sequence,shape,K)
            tmpT = T.detach()
            tmpR = R.detach()
            varR = Variable(R,requires_grad=True)
            varT = Variable(T,requires_grad=True)
            optimizer = torch.optim.Adam([varR,varT],lr=1e-1)

            # optimize results for image consistency
            ferror = []
            losses = []
            minerror = 10000
            for iter in itertools.count():
                K = torch.zeros((3,3))
                K[0,0] = varf;
                K[1,1] = varf;
                K[2,2] = 1;
                K[0,2] = 0;
                K[1,2] = 0;

                R = varR
                T = varT
                Xc,_,_ = util.EPnP(sequence,shape,K)
                #Xc,R,T = util.EPnP(sequence,shape,K)
                optimizer.zero_grad()

                # k inverse
                kinv = torch.zeros(3,3).float()
                kinv[0,0] = 1/varf
                kinv[1,1] = 1/varf
                kinv[2,2] = 1

                # get errors
                reproj_errors2 = util.getReprojError2(sequence,shape,R,T,K)
                #reproj_errors3 = util.getReprojError3(x_cam_gt,shape,varR,varT)
                error_3d = util.getRelReprojError3(x_cam_gt,shape,R,T).mean()
                #error_3d = util.getPCError(x_cam_gt,x_one.permute(0,2,1),torch.stack(100*[kinv]),mode='l2')

                error_Rconsistency = util.getRConsistency(R)
                error_Tconsistency = util.getTConsistency(T)*0.001
                error_3dconsistency = util.get3DConsistency(sequence,shape,kinv,R,T)
                reproj_error = torch.mean(reproj_errors2)

                # determine convergence
                loss = error_3dconsistency
                if loss < minerror:
                    minerror = loss
                    minf = varf.item()
                    minR = R
                    minT = T
                    convergence = 0
                else:
                    convergence += 1

                loss.backward()
                optimizer.step()

                f = util.solvef(sequence,Xc.detach())
                print(f)
                #if varf < 0: varf = varf*-1
                delta = K[0,0] - varf
                direction = torch.sign(delta)
                error_f = torch.abs(varf - f_gt) / f_gt
                ferror.append(error_f.item())
                losses.append(loss.item())

                print(f"iter: {iter} | loss: {loss.item():.3f} | f/fgt: {varf.item():.3f}/{f_gt.item():.3f} | 2d error: {reproj_error.item():.3f} | error R: {error_Rconsistency.item():.3f} | error T: {error_Tconsistency.item():.3f} | error 3d: {error_3dconsistency.item():.3f} | GT RMSE: {error_3d.item():.3f} | delta: {delta.item():.3f}")
                if convergence == 100: break

            data = {'ferror': np.array(ferror), 'loss': np.array(losses)}
            scipy.io.savemat("optimizationlr1.mat",data)
            quit()

            reconstruction_error = reproj_errors3.mean()
            rel_error = rel_errors.mean()
            f_error = torch.abs(f_gt - fout) / f_gt

            allerror_3d.append(reproj_error.data.numpy())
            allerror_2d.append(reconstruction_error.data.numpy())
            allerror_rel3d.append(rel_error.data.numpy())

            error_2d.append(reproj_error.cpu().data.item())
            error_3d.append(reconstruction_error.cpu().data.item())
            error_rel3d.append(rel_error.cpu().data.item())
            error_relf.append(f_error.cpu().data.item())

            print(f"f/sequence: {f_test}/{k}  | f/fgt: {fout[0].item():.3f}/{f_gt.item():.3f} |  f_error_rel: {f_error.item():.4f}  | rmse: {reconstruction_error.item():.4f}  | rel rmse: {rel_error.item():.4f}    | 2d error: {reproj_error.item():.4f}")
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

    model = CalibrationNet3()
    test(model)
    #testBIWI(model)


