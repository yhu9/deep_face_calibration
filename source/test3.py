
import itertools
import argparse
import os

import scipy.io
import torch
import numpy as np

from model import CalibrationNet3
import dataloader
import util

####################################################

parser = argparse.ArgumentParser(description="training arguments")
parser.add_argument("--model", default="net.pt")
parser.add_argument("--out",default="results/exp.mat")
parser.add_argument("--opt", default=False, action='store_true')
parser.add_argument("--db", default="syn")
args = parser.parse_args()

np.random.seed(0)
#########################################################
def trainfc(model):
    for name, param in model.named_parameters():
        if 'fc' in name and 'feat' not in name:
            param.requires_grad = True


# apply dual optimization on two networks
def dualoptimization(calib_net,sfm_net):
    f=1
    shape=1
    return f, shape

def testBIWIID(modelin=args.model,outfile=args.out,optimize=args.opt):
    # define model, dataloader, 3dmm eigenvectors, optimization method
    calib_net = CalibrationNet3(n=1)
    sfm_net = CalibrationNet3(n=199)
    if modelin != "":
        calib_path = os.path.join('model','calib_' + modelin)
        sfm_path = os.path.join('model','sfm_' + modelin)
        calib_net.load_state_dict(torch.load(calib_path))
        sfm_net.load_state_dict(torch.load(sfm_path))
    calib_net.eval()
    sfm_net.eval()

    # mean shape and eigenvectors for 3dmm
    data3dmm = dataloader.SyntheticLoader()
    mu_lm = torch.from_numpy(data3dmm.mu_lm).float().detach()
    mu_lm[:,2] = mu_lm[:,2]*-1
    lm_eigenvec = torch.from_numpy(data3dmm.lm_eigenvec).float().detach()
    sigma = torch.from_numpy(data3dmm.sigma).float().detach()
    sigma = torch.diag(sigma.squeeze())
    lm_eigenvec = torch.mm(lm_eigenvec, sigma)

    # define loader
    loader = dataloader.BIWIIDLoader()
    f_pred = []
    shape_pred = []
    error_2d = []
    error_relf = []
    error_rel3d = []
    for idx in range(len(loader)):
        batch = loader[idx]
        x_cam_gt = batch['x_cam_gt']
        fgt = batch['f_gt']
        x_img = batch['x_img']
        x_img_gt = batch['x_img_gt']
        M = x_img_gt.shape[0]
        N = 68

        ptsI = x_img.reshape((M,N,2)).permute(0,2,1)
        x = ptsI.unsqueeze(0).permute(0,2,1,3)

        # run the model
        f = calib_net(x) + 300
        betas = sfm_net(x)
        betas = betas.squeeze(0).unsqueeze(-1)
        shape = mu_lm + torch.mm(lm_eigenvec,betas).squeeze().view(N,3)

        # additional optimization on initial solution
        if optimize:
            calib_net.load_state_dict(torch.load(calib_path))
            sfm_net.load_state_dict(torch.load(sfm_path))
            calib_net.eval()
            sfm_net.eval()
            trainfc(calib_net)
            trainfc(sfm_net)

            opt1 = torch.optim.Adam(calib_net.parameters(),lr=1e-4)
            opt2 = torch.optim.Adam(sfm_net.parameters(),lr=1e-5)
            curloss = 100

            for outerloop in itertools.count():

                # camera calibration
                shape = shape.detach()
                for iter in itertools.count():
                    opt1.zero_grad()
                    f = calib_net.forward2(x) + 300
                    K = torch.zeros(3,3).float()
                    K[0,0] = f
                    K[1,1] = f
                    K[2,2] = 1

                    f_error = torch.mean(torch.abs(f - fgt))
                    #rmse = torch.norm(shape_gt - shape,dim=1).mean()

                    # differentiable PnP pose estimation
                    km,c_w,scaled_betas, alphas = util.EPnP(ptsI,shape,K)
                    Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
                    error2d = util.getReprojError2(ptsI,shape,R,T,K,show=False,loss='l2')
                    error_time = util.getTimeConsistency(shape,R,T)

                    loss = error2d.mean() + 0.01*error_time

                    if iter == 5: break
                    #if iter > 10 and prev_loss < loss:
                    #    break
                    #else:
                    #    prev_loss = loss
                    loss.backward()
                    opt1.step()
                    print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.item():.1f}/{fgt[0].item():.1f} | error2d: {error2d.mean().item():.3f} ")

                # sfm
                f = f.detach()
                for iter in itertools.count():
                    opt2.zero_grad()

                    # shape prediction
                    betas = sfm_net.forward2(x)
                    shape = torch.sum(betas * lm_eigenvec,1)
                    shape = shape.reshape(68,3) + mu_lm
                    shape = shape - shape.mean(0).unsqueeze(0)

                    K = torch.zeros((3,3)).float()
                    K[0,0] = f
                    K[1,1] = f
                    K[2,2] = 1

                    #rmse = torch.norm(shape_gt - shape,dim=1).mean().detach()
                    #rmse = torch.norm(shape_gt - shape,dim=1).mean().detach()

                    # differentiable PnP pose estimation
                    km,c_w,scaled_betas,alphas = util.EPnP(ptsI,shape,K)
                    Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
                    error2d = util.getReprojError2(ptsI,shape,R,T,K,show=False,loss='l2')
                    error_time = util.getTimeConsistency(shape,R,T)

                    loss = error2d.mean() + 0.01*error_time
                    if iter == 5: break
                    if iter > 10 and prev_loss < loss:
                        break
                    else:
                        prev_loss = loss
                    loss.backward()
                    opt2.step()
                    print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.item():.1f}/{fgt[0].item():.1f} | error2d: {error2d.mean().item():.3f} ")

                # closing condition for outerloop on dual objective
                if torch.abs(curloss - loss) < 0.01: break
                curloss = loss
        else:
            K = torch.zeros(3,3).float()
            K[0,0] = f
            K[1,1] = f
            K[2,2] = 1
            km,c_w,scaled_betas,alphas = util.EPnP(ptsI,shape,K)
            Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)

        # get errors
        reproj_errors2 = util.getReprojError2(ptsI,shape,R,T,K)
        rel_errors = util.getRelReprojError3(x_cam_gt,shape,R,T)

        reproj_error = reproj_errors2.mean()
        rel_error = rel_errors.mean()
        f_error = torch.abs(fgt - f) / fgt

        # save final prediction
        f_pred.append(f.detach().cpu().item())
        shape_pred.append(shape.detach().cpu().numpy())

        error_2d.append(reproj_error.cpu().data.item())
        error_rel3d.append(rel_error.cpu().data.item())
        error_relf.append(f_error.cpu().data.item())

        print(f" f/fgt: {f[0].item():.3f}/{fgt.item():.3f} |  f_error_rel: {f_error.item():.4f}  | rel rmse: {rel_error.item():.4f}    | 2d error: {reproj_error.item():.4f}")
        #end for

    # prepare output file
    out_shape = np.stack(shape_pred)
    out_f = np.stack(f_pred)

    matdata = {}
    matdata['shape'] = np.stack(out_shape)
    matdata['f'] = np.stack(out_f)
    matdata['error_2d'] = np.array(error_2d)
    matdata['error_rel3d'] = np.array(error_rel3d)
    matdata['error_relf'] = np.array(error_relf)
    scipy.io.savemat(outfile,matdata)

    print(f"MEAN seterror_2d: {np.mean(error_2d)}")
    print(f"MEAN seterror_rel3d: {np.mean(error_rel3d)}")
    print(f"MEAN seterror_relf: {np.mean(error_relf)}")

def testBIWI(modelin=args.model,outfile=args.out,optimize=args.opt):
    # define model, dataloader, 3dmm eigenvectors, optimization method
    calib_net = CalibrationNet3(n=1)
    sfm_net = CalibrationNet3(n=199)
    if modelin != "":
        calib_path = os.path.join('model','calib_' + modelin)
        sfm_path = os.path.join('model','sfm_' + modelin)
        calib_net.load_state_dict(torch.load(calib_path))
        sfm_net.load_state_dict(torch.load(sfm_path))
    calib_net.eval()
    sfm_net.eval()

    # mean shape and eigenvectors for 3dmm
    data3dmm = dataloader.SyntheticLoader()
    mu_lm = torch.from_numpy(data3dmm.mu_lm).float().detach()
    mu_lm[:,2] = mu_lm[:,2]*-1
    lm_eigenvec = torch.from_numpy(data3dmm.lm_eigenvec).float().detach()
    sigma = torch.from_numpy(data3dmm.sigma).float().detach()
    sigma = torch.diag(sigma.squeeze())
    lm_eigenvec = torch.mm(lm_eigenvec, sigma)

    # define loader
    loader = dataloader.BIWILoader()
    f_pred = []
    shape_pred = []
    error_2d = []
    error_relf = []
    error_rel3d = []
    for sub in range(len(loader)):
        batch = loader[sub]
        x_cam_gt = batch['x_cam_gt']
        fgt = batch['f_gt']
        x_img = batch['x_img']
        x_img_gt = batch['x_img_gt']
        M = x_img_gt.shape[0]
        N = 68


        ptsI = x_img.reshape((M,N,2)).permute(0,2,1)
        x = ptsI.unsqueeze(0).permute(0,2,1,3)

        # run the model
        f = calib_net(x) + 300
        betas = sfm_net(x)
        betas = betas.squeeze(0).unsqueeze(-1)
        shape = mu_lm + torch.mm(lm_eigenvec,betas).squeeze().view(N,3)

        # additional optimization on initial solution
        if optimize:
            calib_net.load_state_dict(torch.load(calib_path))
            sfm_net.load_state_dict(torch.load(sfm_path))
            calib_net.eval()
            sfm_net.eval()
            trainfc(calib_net)
            trainfc(sfm_net)

            opt1 = torch.optim.Adam(calib_net.parameters(),lr=1e-4)
            opt2 = torch.optim.Adam(sfm_net.parameters(),lr=1e-5)
            curloss = 100

            for outerloop in itertools.count():

                # camera calibration
                shape = shape.detach()
                for iter in itertools.count():
                    opt1.zero_grad()
                    f = calib_net.forward2(x) + 300
                    K = torch.zeros(3,3).float()
                    K[0,0] = f
                    K[1,1] = f
                    K[2,2] = 1

                    f_error = torch.mean(torch.abs(f - fgt))
                    #rmse = torch.norm(shape_gt - shape,dim=1).mean()

                    # differentiable PnP pose estimation
                    km,c_w,scaled_betas, alphas = util.EPnP(ptsI,shape,K)
                    Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
                    error2d = util.getReprojError2(ptsI,shape,R,T,K,show=False,loss='l2')
                    error_time = util.getTimeConsistency(shape,R,T)

                    loss = error2d.mean() + 0.01*error_time
                    if iter == 5: break
                    if iter > 10 and prev_loss < loss:
                        break
                    else:
                        prev_loss = loss
                    loss.backward()
                    opt1.step()
                    print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.item():.1f}/{fgt[0].item():.1f} | error2d: {error2d.mean().item():.3f} ")

                # sfm
                f = f.detach()
                for iter in itertools.count():
                    opt2.zero_grad()

                    # shape prediction
                    betas = sfm_net.forward2(x)
                    shape = torch.sum(betas * lm_eigenvec,1)
                    shape = shape.reshape(68,3) + mu_lm
                    shape = shape - shape.mean(0).unsqueeze(0)
                    K = torch.zeros((3,3)).float()
                    K[0,0] = f
                    K[1,1] = f
                    K[2,2] = 1

                    #rmse = torch.norm(shape_gt - shape,dim=1).mean().detach()
                    #rmse = torch.norm(shape_gt - shape,dim=1).mean().detach()

                    # differentiable PnP pose estimation
                    km,c_w,scaled_betas,alphas = util.EPnP(ptsI,shape,K)
                    Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
                    error2d = util.getReprojError2(ptsI,shape,R,T,K,show=False,loss='l2')
                    error_time = util.getTimeConsistency(shape,R,T)

                    loss = error2d.mean() + 0.01*error_time
                    if iter == 5: break
                    if iter > 10 and prev_loss < loss:
                        break
                    else:
                        prev_loss = loss
                    loss.backward()
                    opt2.step()
                    print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.item():.1f}/{fgt[0].item():.1f} | error2d: {error2d.mean().item():.3f} ")

                # closing condition for outerloop on dual objective
                if torch.abs(curloss - loss) < 0.01: break
                curloss = loss
        else:
            K = torch.zeros(3,3).float()
            K[0,0] = f
            K[1,1] = f
            K[2,2] = 1
            km,c_w,scaled_betas,alphas = util.EPnP(ptsI,shape,K)
            Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)

        # get errors
        reproj_errors2 = util.getReprojError2(ptsI,shape,R,T,K)
        rel_errors = util.getRelReprojError3(x_cam_gt,shape,R,T)

        reproj_error = reproj_errors2.mean()
        rel_error = rel_errors.mean()
        f_error = torch.abs(fgt - f) / fgt

        # save final prediction
        f_pred.append(f.detach().cpu().item())
        shape_pred.append(shape.detach().cpu().numpy())

        error_2d.append(reproj_error.cpu().data.item())
        error_rel3d.append(rel_error.cpu().data.item())
        error_relf.append(f_error.cpu().data.item())

        print(f" f/fgt: {f[0].item():.3f}/{fgt.item():.3f} |  f_error_rel: {f_error.item():.4f}  | rel rmse: {rel_error.item():.4f}    | 2d error: {reproj_error.item():.4f}")
        #end for

    # prepare output file
    out_shape = np.stack(shape_pred)
    out_f = np.stack(f_pred)

    matdata = {}
    matdata['shape'] = np.stack(out_shape)
    matdata['f'] = np.stack(out_f)
    matdata['error_2d'] = np.array(error_2d)
    matdata['error_rel3d'] = np.array(error_rel3d)
    matdata['error_relf'] = np.array(error_relf)
    scipy.io.savemat(outfile,matdata)

    print(f"MEAN seterror_2d: {np.mean(error_2d)}")
    print(f"MEAN seterror_rel3d: {np.mean(error_rel3d)}")
    print(f"MEAN seterror_relf: {np.mean(error_relf)}")

def test(modelin=args.model,outfile=args.out,optimize=args.opt):

    # define model, dataloader, 3dmm eigenvectors, optimization method
    calib_net = CalibrationNet3(n=1)
    sfm_net = CalibrationNet3(n=199)
    if modelin != "":
        calib_path = os.path.join('model','calib_' + modelin)
        sfm_path = os.path.join('model','sfm_' + modelin)
        calib_net.load_state_dict(torch.load(calib_path))
        sfm_net.load_state_dict(torch.load(sfm_path))
    calib_net.eval()
    sfm_net.eval()

    # mean shape and eigenvectors for 3dmm
    M = 100
    data3dmm = dataloader.SyntheticLoader()
    mu_lm = torch.from_numpy(data3dmm.mu_lm).float().detach()
    mu_lm[:,2] = mu_lm[:,2]*-1
    lm_eigenvec = torch.from_numpy(data3dmm.lm_eigenvec).float().detach()
    sigma = torch.from_numpy(data3dmm.sigma).float().detach()
    sigma = torch.diag(sigma.squeeze())
    lm_eigenvec = torch.mm(lm_eigenvec, sigma)

    # sample from f testing set
    allerror_2d = []
    allerror_3d = []
    allerror_rel3d = []
    allerror_relf = []
    all_f = []
    all_fpred = []
    all_depth = []
    out_shape = []
    out_f = []

    seterror_3d = []
    seterror_rel3d = []
    seterror_relf = []
    seterror_2d = []
    f_vals = [i*100 for i in range(4,15)]
    for f_test in f_vals:
        # create dataloader
        #f_test = 1000
        loader = dataloader.TestLoader(f_test)

        f_pred = []
        shape_pred = []
        error_2d = []
        error_3d = []
        error_rel3d = []
        error_relf = []
        M = 100;
        N = 68;
        batch_size = 1;

        for j,data in enumerate(loader):
            if j == 10: break
            # load the data
            x_cam_gt = data['x_cam_gt']
            shape_gt = data['x_w_gt']
            fgt = data['f_gt']
            x_img = data['x_img']
            x_img_gt = data['x_img_gt']
            T_gt = data['T_gt']

            all_depth.append(np.mean(T_gt[:,2]))
            all_f.append(fgt.numpy()[0])

            ptsI = x_img.reshape((M,N,2)).permute(0,2,1)
            x = ptsI.unsqueeze(0).permute(0,2,1,3)

            # run the model
            f = calib_net(x) + 300
            betas = sfm_net(x)
            betas = betas.squeeze(0).unsqueeze(-1)
            shape = mu_lm + torch.mm(lm_eigenvec,betas).squeeze().view(N,3)

            # additional optimization on initial solution
            if optimize:
                calib_net.load_state_dict(torch.load(calib_path))
                sfm_net.load_state_dict(torch.load(sfm_path))
                calib_net.eval()
                sfm_net.eval()
                trainfc(calib_net)
                trainfc(sfm_net)
                opt1 = torch.optim.Adam(calib_net.parameters(),lr=1e-4)
                opt2 = torch.optim.Adam(sfm_net.parameters(),lr=1e-2)
                curloss = 100
                for outerloop in itertools.count():

                    # camera calibration
                    shape = shape.detach()
                    for iter in itertools.count():
                        opt1.zero_grad()
                        f = calib_net.forward2(x) + 300
                        K = torch.zeros(3,3).float()
                        K[0,0] = f
                        K[1,1] = f
                        K[2,2] = 1

                        f_error = torch.mean(torch.abs(f - fgt))
                        rmse = torch.norm(shape_gt - shape,dim=1).mean()

                        # differentiable PnP pose estimation
                        km,c_w,scaled_betas, alphas = util.EPnP(ptsI,shape,K)
                        Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
                        error2d = util.getReprojError2(ptsI,shape,R,T,K,show=False,loss='l2')
                        #error2d = util.getReprojError2_(ptsI,Xc,K,show=True,loss='l2')
                        error_time = util.getTimeConsistency(shape,R,T)

                        loss = error2d.mean() + 0.01*error_time
                        if iter == 5: break
                        loss.backward()
                        opt1.step()
                        print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.item():.1f}/{fgt[0].item():.1f} | error2d: {error2d.mean().item():.3f} | rmse: {rmse.item():.3f} ")

                    # sfm
                    f = f.detach()
                    for iter in itertools.count():
                        opt2.zero_grad()

                        # shape prediction
                        betas = sfm_net.forward2(x)
                        shape = torch.sum(betas * lm_eigenvec,1)
                        shape = shape.reshape(68,3) + mu_lm
                        shape = shape - shape.mean(0).unsqueeze(0)
                        K = torch.zeros((3,3)).float()
                        K[0,0] = f
                        K[1,1] = f
                        K[2,2] = 1

                        #rmse = torch.norm(shape_gt - shape,dim=1).mean().detach()
                        rmse = torch.norm(shape_gt - shape,dim=1).mean().detach()

                        # differentiable PnP pose estimation
                        km,c_w,scaled_betas,alphas = util.EPnP(ptsI,shape,K)
                        Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)
                        error2d = util.getReprojError2(ptsI,shape,R,T,K,show=False,loss='l2')
                        error_time = util.getTimeConsistency(shape,R,T)

                        loss = error2d.mean() + 0.01*error_time
                        if iter == 5: break
                        if iter > 10 and prev_loss < loss:
                            break
                        else:
                            prev_loss = loss
                        loss.backward()
                        opt2.step()
                        print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.item():.1f}/{fgt[0].item():.1f} | error2d: {error2d.mean().item():.3f} | rmse: {rmse.item():.3f} ")

                    # closing condition for outerloop on dual objective
                    if torch.abs(curloss - loss) < 0.01: break
                    curloss = loss
            else:
                K = torch.zeros(3,3).float()
                K[0,0] = f
                K[1,1] = f
                K[2,2] = 1
                km,c_w,scaled_betas,alphas = util.EPnP(ptsI,shape,K)
                Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)

            all_fpred.append(f.detach().numpy()[0])

            # get errors
            reproj_errors2 = util.getReprojError2(ptsI,shape,R,T,K,show=False)
            reproj_errors3 = torch.norm(shape_gt - shape,dim=1).mean()
            rel_errors =  util.getRelReprojError3(x_cam_gt,shape,R,T)

            reproj_error = reproj_errors2.mean()
            reconstruction_error = reproj_errors3.mean()
            rel_error = rel_errors.mean()
            f_error = torch.abs(fgt - f) / fgt

            # save final prediction
            f_pred.append(f.detach().cpu().item())
            shape_pred.append(shape.detach().cpu().numpy())

            allerror_3d.append(reproj_error.data.numpy())
            allerror_2d.append(reconstruction_error.data.numpy())
            allerror_rel3d.append(rel_error.data.numpy())
            error_2d.append(reproj_error.cpu().data.item())
            error_3d.append(reconstruction_error.cpu().data.item())
            error_rel3d.append(rel_error.cpu().data.item())
            error_relf.append(f_error.cpu().data.item())

            print(f"f/sequence: {f_test}/{j}  | f/fgt: {f[0].item():.3f}/{fgt.item():.3f} |  f_error_rel: {f_error.item():.4f}  | rmse: {reconstruction_error.item():.4f}  | rel rmse: {rel_error.item():.4f}    | 2d error: {reproj_error.item():.4f}")

        avg_2d = np.mean(error_2d)
        avg_rel3d = np.mean(error_rel3d)
        avg_3d = np.mean(error_3d)
        avg_relf = np.mean(error_relf)

        seterror_2d.append(avg_2d)
        seterror_3d.append(avg_3d)
        seterror_rel3d.append(avg_rel3d)
        seterror_relf.append(avg_relf)
        out_f.append(np.stack(f_pred))
        out_shape.append(np.stack(shape_pred,axis=0))
        print(f"f_error_rel: {avg_relf:.4f}  | rel rmse: {avg_rel3d:.4f}    | 2d error: {reproj_error.item():.4f} |  rmse: {avg_3d:.4f}  |")

    out_shape = np.stack(out_shape)
    out_f = np.stack(out_f)
    all_f = np.stack(all_f).flatten()
    all_fpred = np.stack(all_fpred).flatten()
    all_d = np.stack(all_depth).flatten()
    allerror_2d = np.stack(allerror_2d).flatten()
    allerror_3d = np.stack(allerror_3d).flatten()
    allerror_rel3d = np.stack(allerror_rel3d).flatten()

    matdata = {}
    matdata['fvals'] = np.array(f_vals)
    matdata['all_f'] = np.array(all_f)
    matdata['all_fpred'] = np.array(all_fpred)
    matdata['all_d'] = np.array(all_depth)
    matdata['error_2d'] = allerror_2d
    matdata['error_3d'] = allerror_3d
    matdata['error_rel3d'] = allerror_rel3d
    matdata['seterror_2d'] = np.array(seterror_2d)
    matdata['seterror_3d'] = np.array(seterror_3d)
    matdata['seterror_rel3d'] = np.array(seterror_rel3d)
    matdata['seterror_relf'] = np.array(seterror_relf)
    matdata['shape'] = np.stack(out_shape)
    matdata['f'] = np.stack(out_f)
    scipy.io.savemat(outfile,matdata)

    print(f"MEAN seterror_2d: {np.mean(seterror_2d)}")
    print(f"MEAN seterror_3d: {np.mean(seterror_3d)}")
    print(f"MEAN seterror_rel3d: {np.mean(seterror_rel3d)}")
    print(f"MEAN seterror_relf: {np.mean(seterror_relf)}")
    #end function

####################################################################################3
if __name__ == '__main__':

    if args.db == 'syn':
        test()
    elif args.db == 'biwi':
        testBIWI()
    elif args.db == 'biwiid':
        testBIWIID()
    else:
        test()


