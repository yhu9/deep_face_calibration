
import itertools
import argparse

import scipy.io
import torch
import numpy as np

from logger import Logger
from model import PointNet
import dataloader
import util

####################################################

parser = argparse.ArgumentParser(description="training arguments")
parser.add_argument("--model", default="")
parser.add_argument("--out",default="results/exp.mat")
args = parser.parse_args()

####################################################

def test(modelin=args.model,outfile=args.out):

    # define model, dataloader, 3dmm eigenvectors, optimization method
    model = CalibrationNet2()
    if modelin != "":
        model.load_state_dict(torch.load(modelin))
    model.cuda()

    # mean shape and eigenvectors for 3dmm
    M = 100
    data3dmm = dataloader.SyntheticLoader()
    mu_lm = torch.from_numpy(data3dmm.mu_lm).float().cuda()
    lm_eigenvec = torch.from_numpy(data3dmm.lm_eigenvec).float().cuda()

    # sample from f testing set
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

        for k in range(len(data)):
            batch = data[k]
            x_cam_gt = batch['x_cam_gt'].cuda()
            x_w_gt = batch['x_w_gt'].cuda()
            f_gt = batch['f_gt'].cuda()
            x_img = batch['x_img'].cuda()
            x_img_gt = batch['x_img_gt'].cuda()

            one  = torch.ones(M,1,68).cuda()
            x_img_one = torch.cat([x_img,one],dim=1)

            # run the model
            out, trans, transfeat = model(x_img_one)
            alphas = out[:,:199].mean(0)
            f = torch.relu(out[:,199]).mean()
            K = torch.zeros((3,3)).float().cuda()
            K[0,0] = f;
            K[1,1] = f;
            K[2,2] = 1;
            K[0,2] = 320;
            K[1,2] = 240;

            # apply 3DMM model from predicted parameters
            alpha_matrix = torch.diag(alphas)
            shape_cov = torch.mm(lm_eigenvec,alpha_matrix)
            s = shape_cov.sum(1).view(68,3)
            shape = (mu_lm + s)

            # run epnp algorithm
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
            rel_error_n1 = util.getRelReprojError3(x_cam_gt,shape,Rn1,Tn1)

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
            rel_error_n2 = util.getRelReprojError3(x_cam_gt,shape,Rn1,Tn1)

            mask = reproj_error2_n1 < reproj_error2_n2


            reproj_errors = torch.cat((reproj_error2_n1[mask],reproj_error2_n2[~mask]))
            rmse_errors = torch.cat((reproj_error3_n1[mask],reproj_error3_n2[~mask]))
            rel_errors = torch.cat((rel_error_n2[~mask],rel_error_n1[mask]))

            # errors
            reproj_error = torch.mean(reproj_errors)
            reconstruction_error = torch.mean(rmse_errors)
            rel_error = torch.mean(rel_errors)
            f_error = torch.abs(f_gt - f) / f_gt

            error_2d.append(reproj_error.cpu().data.item())
            error_3d.append(reconstruction_error.cpu().data.item())
            error_rel3d.append(rel_error.cpu().data.item())
            error_relf.append(f_error.cpu().data.item())

            print(f"f/sequence: {f_test}/{k}  | f_error_rel: {f_error.item():.4f}  | rmse: {reconstruction_error.item():.4f}  | rel rmse: {rel_error.item():.4f}    | 2d error: {reproj_error.item():.4f}")

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

    matdata = {}
    matdata['fvals'] = np.array(f_vals)
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
    test()


