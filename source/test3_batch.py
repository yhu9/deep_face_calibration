
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
parser.add_argument("--device", default='cpu')
args = parser.parse_args()

#########################################################

def testBIWI(model,modelin=args.model,outfile=args.out):
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

def test(modelin=args.model,outfile=args.out,optimize=args.opt):

    # define model, dataloader, 3dmm eigenvectors, optimization method
    calib_net = CalibrationNet3(n=1)
    sfm_net = CalibrationNet3(n=199)
    if modelin != "":
        calib_path = os.path.join('model','calib_' + modelin)
        sfm_path = os.path.join('model','sfm_' + modelin)
        calib_net.load_state_dict(torch.load(calib_path,map_location='cpu'))
        sfm_net.load_state_dict(torch.load(sfm_path,map_location='cpu'))
    calib_net.to(args.device)
    sfm_net.to(args.device)
    calib_net.eval()
    sfm_net.eval()

    # mean shape and eigenvectors for 3dmm
    M = 100
    data3dmm = dataloader.SyntheticLoader()
    mu_lm = torch.from_numpy(data3dmm.mu_lm).float().to(args.device).detach()
    mu_lm[:,2] = mu_lm[:,2]*-1
    lm_eigenvec = torch.from_numpy(data3dmm.lm_eigenvec).float().to(args.device).detach()
    sigma = torch.from_numpy(data3dmm.sigma).float().to(args.device).detach()
    sigma = torch.diag(sigma.squeeze())
    lm_eigenvec = torch.mm(lm_eigenvec, sigma)

    batch_size = 10
    lm_eigenvec = torch.stack(batch_size*[lm_eigenvec])

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
        f_test = 1000

        # create dataloader
        data = dataloader.TestData()
        data.batchsize = batch_size
        loader = data.createLoader(f_test)

        # containers
        f_pred = []
        shape_pred = []
        error_2d = []
        error_3d = []
        error_rel3d = []
        error_relf = []
        M = 100;
        N = 68;
        batch_size = data.batchsize;

        for j,data in enumerate(loader):
            # load the data
            x_cam_gt = data['x_cam_gt'].to(args.device)
            shape_gt = data['x_w_gt'].to(args.device)
            fgt = data['f_gt'].to(args.device)
            x_img = data['x_img'].to(args.device)
            x_img_gt = data['x_img_gt'].to(args.device)
            T_gt = data['T_gt'].to(args.device)

            # reshape and form data
            one = torch.ones(batch_size,M*N,1).to(device=args.device)
            x_img_one = torch.cat([x_img,one],dim=2)
            x_cam_pt = x_cam_gt.permute(0,1,3,2).reshape(batch_size,6800,3)
            x = x_img.permute(0,2,1).reshape(batch_size,2,M,N)
            ptsI = x_img_one.reshape(batch_size,M,N,3).permute(0,1,3,2)[:,:,:2,:]

            # run the model
            f = calib_net(x) + 300
            betas = sfm_net(x)
            betas = betas.squeeze(0).unsqueeze(-1)
            shape = mu_lm + torch.bmm(lm_eigenvec,betas).squeeze().view(batch_size,N,3)

            # additional optimization on initial solution
            if optimize:
                calib_net.load_state_dict(torch.load(calib_path,map_location=args.device))
                sfm_net.load_state_dict(torch.load(sfm_path,map_location=args.device))
                calib_net.train()
                sfm_net.train()
                opt1 = torch.optim.Adam(calib_net.parameters(),lr=1e-4)
                opt2 = torch.optim.Adam(sfm_net.parameters(),lr=1e-2)
                curloss = 100
                for outerloop in itertools.count():

                    # camera calibration
                    shape = shape.detach()
                    for iter in itertools.count():
                        opt1.zero_grad()
                        f = torch.mean(calib_net.forward2(x) + 300)
                        K = torch.zeros(3,3).float().to(device=args.device)
                        K[0,0] = f
                        K[1,1] = f
                        K[2,2] = 1

                        # ground truth l1 error
                        f_error = torch.mean(torch.abs(f - fgt))

                        # rmse
                        rmse = torch.norm(shape_gt - shape,dim=2).mean()

                        # differentiable PnP pose estimation
                        error1 = []
                        for i in range(batch_size):
                            km, c_w, scaled_betas, alphas = util.EPnP(ptsI[i],shape[i],K)
                            Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape[i],ptsI[i],K)
                            error2d = util.getReprojError2(ptsI[i],shape[i],R,T,K,show=False,loss='l1')
                            error1.append(error2d.mean())

                        # loss
                        loss = torch.stack(error1).mean()

                        # stopping condition
                        if iter == 5: break
                        if iter > 5 and prev_loss < loss:
                            break
                        else:
                            prev_loss = loss

                        # update
                        loss.backward()
                        opt1.step()
                        print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.mean().item():.1f}/{fgt.mean().item():.1f} | error2d: {loss.item():.3f} | rmse: {rmse.item():.3f} ")

                    # sfm
                    f = f.detach()
                    for iter in itertools.count():
                        opt2.zero_grad()

                        # shape prediction
                        betas = sfm_net.forward2(x)
                        betas = betas.unsqueeze(-1)
                        shape = mu_lm + torch.bmm(lm_eigenvec,betas).squeeze().view(batch_size,N,3)
                        K = torch.zeros((3,3)).float()
                        K[0,0] = f
                        K[1,1] = f
                        K[2,2] = 1

                        #rmse = torch.norm(shape_gt - shape,dim=1).mean().detach()
                        rmse = torch.norm(shape_gt - shape,dim=2).mean()

                        # differentiable PnP pose estimation
                        error1 = []
                        for i in range(batch_size):
                            km, c_w, scaled_betas, alphas = util.EPnP(ptsI[i],shape[i],K)
                            Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape[i],ptsI[i],K)
                            error2d = util.getReprojError2(ptsI[i],shape[i],R,T,K,show=False,loss='l1')
                            errorTime = util.getTimeConsistency(shape[i],R,T)
                            error1.append(error2d.mean())

                        #loss = torch.stack(error1).mean() + 0.01*torch.stack(error2).mean()
                        loss = torch.stack(error1).mean()

                        if iter == 5: break
                        if iter > 5 and prev_loss < loss:
                            break
                        else:
                            prev_loss = loss
                        loss.backward()
                        opt2.step()
                        print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.mean().item():.1f}/{fgt.mean().item():.1f} | error2d: {loss.item():.3f} | rmse: {rmse.item():.3f} ")
                    # closing condition for outerloop on dual objective
                    if torch.abs(curloss - loss) < 0.01: break
                    curloss = loss
            else:
                K = torch.zeros((batch_size,3,3)).float().to(device=args.device)
                K[:,0,0] = f.squeeze()
                K[:,1,1] = f.squeeze()
                K[:,2,2] = 1
                km,c_w,scaled_betas,alphas = util.EPnP(ptsI,shape,K)
                Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)

            #all_fpred.append(batch_size*[f.detach().item()])
            e2d,e3d,eshape,e2d_all,e3d_all,d_all = util.getBatchError(ptsI.detach(),shape.detach(),K.detach(),x_cam_gt,shape_gt)
            f_error = torch.squeeze(torch.abs(fgt - f)/fgt)

            e2d = e2d.cpu().numpy()
            e3d = e3d.cpu().numpy()
            eshape = eshape.cpu().numpy()
            f_error = f_error.cpu().squeeze().numpy()
            e2d_all = e2d_all.cpu().numpy()
            e3d_all = e3d_all.cpu().numpy()
            d_all = d_all.cpu().numpy()

            f_pred.append(f.detach().cpu().item())
            shape_pred.append(shape.detach().cpu().numpy())
            all_depth.append(d_all.flatten())
            all_f.append(np.array([fgt.mean()] * d_all.flatten().shape[0]))
            all_fpred.append(np.array([f.mean()]*d_all.flatten().shape[0]))

            print(f"f/sequence: {f_test}/{j}  | f/fgt: {f.mean().item():.3f}/{fgt.mean().item():.3f} |  f_error_rel: {f_error.mean().item():.4f}  | rmse: {eshape.mean().item():.4f}  | rel rmse: {np.mean(e3d):.4f}    | 2d error: {np.mean(e2d):.4f}")

        avg_2d = np.mean(error_2d)
        avg_rel3d = np.mean(error_rel3d)
        avg_3d = np.mean(error_3d)
        avg_relf = np.mean(error_relf)

        seterror_2d.append(avg_2d)
        seterror_3d.append(avg_3d)
        seterror_rel3d.append(avg_rel3d)
        seterror_relf.append(avg_relf)
        out_f.append(np.array(f_pred))
        out_shape.append(np.concatenate(shape_pred,axis=0))
        print(f"f_error_rel: {avg_relf:.4f}  | rel rmse: {avg_rel3d:.4f}    | 2d error: {avg_2d:.4f} |  rmse: {avg_3d:.4f}  |")

    out_shape = np.stack(out_shape)
    out_f = np.stack(out_f)
    all_f = np.stack(all_f).flatten()
    all_fpred = np.stack(all_fpred).flatten()
    all_depth = np.stack(all_depth).flatten()
    allerror_2d = np.stack(allerror_2d).flatten()
    allerror_rel3d = np.stack(allerror_rel3d).flatten()

    matdata = {}
    matdata['fvals'] = np.array(f_vals)
    matdata['all_f'] = np.array(all_f)
    matdata['all_fpred'] = np.array(all_fpred)
    matdata['all_d'] = np.array(all_depth)
    matdata['error_2d'] = allerror_2d
    matdata['error_rel3d'] = allerror_rel3d
    matdata['seterror_2d'] = np.array(seterror_2d)
    matdata['seterror_3d'] = np.array(seterror_3d)
    matdata['seterror_rel3d'] = np.array(seterror_rel3d)
    matdata['seterror_relf'] = np.array(seterror_relf)
    matdata['out_shape'] = out_shape
    matdata['out_f'] = out_f
    scipy.io.savemat(outfile,matdata)

    print(f"MEAN seterror_2d: {np.mean(seterror_2d)}")
    print(f"MEAN seterror_3d: {np.mean(seterror_3d)}")
    print(f"MEAN seterror_rel3d: {np.mean(seterror_rel3d)}")
    print(f"MEAN seterror_relf: {np.mean(seterror_relf)}")
    #end function

####################################################################################3
if __name__ == '__main__':

    test()
    #testBIWI(model)


