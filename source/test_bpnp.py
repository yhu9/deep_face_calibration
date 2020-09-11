
import itertools
import argparse

import scipy.io
import torch
import numpy as np
import torchvision

#from logger import Logger
from model import Model1
import dataloader
import util
import BPnP

bpnp = BPnP.BPnP.apply

##############################################################################################
##############################################################################################
##############################################################################################
parser = argparse.ArgumentParser(description="training arguments")
parser.add_argument("--model", default="net.pt")
parser.add_argument("--out",default="results/exp.mat")
parser.add_argument("--device",default='cpu')
parser.add_argument("--opt", default=False, action='store_true')
parser.add_argument("--db", default="syn")
parser.add_argument("--ft",default=False, action="store_true")
args = parser.parse_args()

np.random.seed(0)
torch.manual_seed(0)
##############################################################################################
##############################################################################################
##############################################################################################
data3dmm = dataloader.SyntheticLoader()
mu_lm = torch.from_numpy(data3dmm.mu_lm).float()
mu_lm[:,2] = mu_lm[:,2] * -1
le = torch.mean(mu_lm[36:42,:],axis=0)
re = torch.mean(mu_lm[42:48,:],axis=0)
ipd = torch.norm(le - re)

# HELPER FUNCTIONS

# dual optimization to optimize focal length and 3D shape
def dualoptimization(x,ptsI,x2d,ini_pose,calib_net,sfm_net,shape_gt=None,fgt=None,M=100,N=68):

    # run the model
    f = torch.sigmoid(calib_net)*2000
    shape = mu_lm

    opt1 = torch.optim.Adam({calib_net},lr=1e-1)
    opt2 = torch.optim.Adam(sfm_net.parameters(),lr=1e-4)
    curloss = 100
    for outerloop in itertools.count():
        f = f.detach()
        for iter in itertools.count():
            opt2.zero_grad()

            # shape prediction
            shape = sfm_net(torch.ones(1,3,32,32)).view(N,3)
            le = torch.mean(shape[36:42,:],axis=0)
            re = torch.mean(shape[42:48,:],axis=0)
            pred_ipd = torch.norm(le - re).detach()
            shape = (ipd/pred_ipd) * shape
            shape = shape - shape.mean(0).unsqueeze(0)


            # focal length
            K = torch.zeros((3,3)).float()
            K[0,0] = f
            K[1,1] = f
            K[2,2] = 1

            # differentiable PnP pose estimation
            pose = bpnp(x2d,shape,K,ini_pose)
            pred = BPnP.batch_project(pose,shape,K)

            # apply loss
            loss = (torch.norm(pred - x2d,dim=-1)).mean()
            if iter >= 5 and loss > prv_loss: break
            loss.backward()
            opt2.step()
            prv_loss = loss.item()

            # log results on console
            if not shape_gt is None:
                d,Z,tform = util.procrustes(shape.detach().numpy(),shape_gt.detach().numpy())
                rmse = torch.norm(shape_gt - Z,dim=1).mean().detach()
            else:
                rmse = -1
            if not fgt is None:
                ftrue = fgt.item()
            else:
                fgt = -1
            f_error = torch.mean(torch.abs(f-ftrue))
            print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.item():.1f}/{ftrue:.1f} | error2d: {loss.item():.3f} | rmse: {rmse:.2f}")

        shape = shape.detach()
        for iter in itertools.count():
            opt1.zero_grad()
            f = torch.sigmoid(calib_net)*2000
            K = torch.zeros(3,3).float()
            K[0,0] = f
            K[1,1] = f
            K[2,2] = 1

            # differentiable PnP pose estimation
            pose = bpnp(x2d,shape,K,ini_pose)
            pred = BPnP.batch_project(pose,shape,K)

            # apply loss
            loss = (torch.norm(pred - x2d,dim=-1)).mean()
            if iter >= 5 and loss > prv_loss: break
            prv_loss = loss.item()
            loss.backward()
            opt1.step()

            # log results on console
            if not shape_gt is None:
                d,Z,tform = util.procrustes(shape.detach().numpy(),shape_gt.detach().numpy())
                rmse = torch.norm(shape_gt - Z,dim=1).mean().detach()
            else:
                rmse = -1
            if not fgt is None:
                ftrue = fgt.item()
            else:
                fgt = -1
            f_error = torch.mean(torch.abs(f-ftrue))
            print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.item():.1f}/{ftrue:.1f} | error2d: {loss.item():.3f} | rmse: {rmse:.2f}")

        if torch.abs(curloss  - loss) <= 0.01 or curloss < loss: break
        curloss = loss

    km,c_w,scaled_betas,alphas = util.EPnP(ptsI,shape,K)
    Xc,R,T,mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)

    return shape,K,R,T

def getLoader(db):
    if db == 'syn':
        loader = dataloader.TestLoader(f_test)
    elif db == 'human36':
        loader = dataloader.Human36Loader()
    elif db == 'cad120':
        loader = dataloader.Cad120Loader()
    elif db == 'biwi':
        loader = dataloader.BIWILoader()
    elif db == 'biwiid':
        loader = dataloader.BIWIIDLoader()
    return loader

def testReal(modelin=args.model,outfile=args.out,optimize=args.opt,db=args.db):
    # mean shape and eigenvectors for 3dmm
    data3dmm = dataloader.SyntheticLoader()
    mu_lm = torch.from_numpy(data3dmm.mu_lm).float().detach()
    mu_lm[:,2] = mu_lm[:,2]*-1
    lm_eigenvec = torch.from_numpy(data3dmm.lm_eigenvec).float().detach()
    sigma = torch.from_numpy(data3dmm.sigma).float().detach()
    sigma = torch.diag(sigma.squeeze())
    lm_eigenvec = torch.mm(lm_eigenvec, sigma)

    # define loader and initialize logging variables
    loader = getLoader(db)
    f_pred = []
    d_pred = []
    shape_pred = []
    error_2d = []
    error_relf = []
    error_rel3d = []
    for idx in range(len(loader)):

        # load the data
        data = loader[idx]
        x_cam_gt = data['x_cam_gt']
        fgt = data['f_gt']
        x_img = data['x_img']
        x_img_gt = data['x_img_gt']
        M = x_img_gt.shape[0]
        N = 68

        # create bpnp camera calibration model
        calib_net= (1.1*torch.randn(1)).requires_grad_()

        # create bpnp sfm model
        sfm_net  = torchvision.models.vgg11()
        sfm_net.classifier = torch.nn.Linear(25088,N*3)

        ptsI = x_img.reshape((M,N,2)).permute(0,2,1)
        x2d = x_img.view((M,N,2))
        x_img_pts = x_img.reshape((M,N,2)).permute(0,2,1)
        one = torch.ones(M*N,1)
        x_img_one = torch.cat([x_img,one],dim=1)
        x = x_img_one.permute(1,0)

        # run the model
        f = torch.sigmoid(calib_net)*2000
        shape = mu_lm
        ini_pose = torch.zeros((M,6))
        ini_pose[:,5] = 99
        curloss = 100

        # apply dual optimization
        shape,K,R,T = dualoptimization(x,ptsI,x2d,ini_pose,calib_net,sfm_net,fgt=fgt)
        f = K[0,0].detach()

        # get final result to save
        depth = torch.norm(x_cam_gt.mean(2),dim=1)

        # get errors
        reproj_errors2 = util.getReprojError2(ptsI,shape,R,T,K)
        rel_errors = util.getRelReprojError3(x_cam_gt,shape,R,T)

        reproj_error = reproj_errors2.mean()
        rel_error = rel_errors.mean()
        f_error = torch.abs(fgt - f) / fgt

        # save final prediction
        f_pred.append(f.detach().cpu().item())
        d_pred.append(depth.detach().cpu().numpy())
        shape_pred.append(shape.detach().cpu().numpy())

        error_2d.append(reproj_error.cpu().data.item())
        error_rel3d.append(rel_error.cpu().data.item())
        error_relf.append(f_error.cpu().data.item())

        print(f" f/fgt: {f.item():.3f}/{fgt.item():.3f} |  f_error_rel: {f_error.item():.4f}  | rel rmse: {rel_error.item():.4f}    | 2d error: {reproj_error.item():.4f}")

    # prepare output file
    matdata = {}
    matdata['f_pred'] = np.stack(f_pred)
    matdata['d_pred'] = np.concatenate(d_pred)
    matdata['shape_pred'] = np.stack(shape_pred)
    matdata['error_2d'] = np.array(error_2d)
    matdata['error_relf'] = np.array(error_relf)
    matdata['error_rel3d'] = np.stack(error_rel3d)
    scipy.io.savemat(outfile,matdata)

    print(f"MEAN seterror_2d: {np.mean(error_2d)}")
    print(f"MEAN seterror_rel3d: {np.mean(error_rel3d)}")
    print(f"MEAN seterror_relf: {np.mean(error_relf)}")

def testBIWIID(modelin=args.model,outfile=args.out,optimize=args.opt):
    # mean shape and eigenvectors for 3dmm
    data3dmm = dataloader.SyntheticLoader()
    mu_lm = torch.from_numpy(data3dmm.mu_lm).float().detach()
    mu_lm[:,2] = mu_lm[:,2]*-1
    lm_eigenvec = torch.from_numpy(data3dmm.lm_eigenvec).float().detach()
    sigma = torch.from_numpy(data3dmm.sigma).float().detach()
    sigma = torch.diag(sigma.squeeze())
    lm_eigenvec = torch.mm(lm_eigenvec, sigma)

    # define loader and initialize logging variables
    loader = dataloader.BIWIIDLoader()
    f_pred = []
    d_pred = []
    shape_pred = []
    error_2d = []
    error_relf = []
    error_rel3d = []
    for idx in range(len(loader)):

        # load the data
        data = loader[idx]
        x_cam_gt = data['x_cam_gt']
        fgt = data['f_gt']
        x_img = data['x_img']
        x_img_gt = data['x_img_gt']
        M = x_img_gt.shape[0]
        N = 68

        # create bpnp camera calibration model
        calib_net= (1.1*torch.randn(1)).requires_grad_()

        # create bpnp sfm model
        sfm_net  = torchvision.models.vgg11()
        sfm_net.classifier = torch.nn.Linear(25088,N*3)

        ptsI = x_img.reshape((M,N,2)).permute(0,2,1)
        x2d = x_img.view((M,N,2))
        x_img_pts = x_img.reshape((M,N,2)).permute(0,2,1)
        one = torch.ones(M*N,1)
        x_img_one = torch.cat([x_img,one],dim=1)
        x = x_img_one.permute(1,0)

        # run the model
        f = torch.sigmoid(calib_net)*2000
        shape = mu_lm
        ini_pose = torch.zeros((M,6))
        ini_pose[:,5] = 99
        curloss = 100

        # apply dual optimization
        shape,K,R,T = dualoptimization(x,ptsI,x2d,ini_pose,calib_net,sfm_net,fgt=fgt)
        f = K[0,0].detach()

        # get final result to save
        depth = torch.norm(x_cam_gt.mean(2),dim=1)

        # get errors
        reproj_errors2 = util.getReprojError2(ptsI,shape,R,T,K)
        rel_errors = util.getRelReprojError3(x_cam_gt,shape,R,T)

        reproj_error = reproj_errors2.mean()
        rel_error = rel_errors.mean()
        f_error = torch.abs(fgt - f) / fgt

        # save final prediction
        f_pred.append(f.detach().cpu().item())
        d_pred.append(depth.detach().cpu().numpy())
        shape_pred.append(shape.detach().cpu().numpy())

        error_2d.append(reproj_error.cpu().data.item())
        error_rel3d.append(rel_error.cpu().data.item())
        error_relf.append(f_error.cpu().data.item())

        print(f" f/fgt: {f.item():.3f}/{fgt.item():.3f} |  f_error_rel: {f_error.item():.4f}  | rel rmse: {rel_error.item():.4f}    | 2d error: {reproj_error.item():.4f}")

    # prepare output file
    matdata = {}
    matdata['f_pred'] = np.stack(f_pred)
    matdata['d_pred'] = np.concatenate(d_pred)
    matdata['shape_pred'] = np.stack(shape_pred)
    matdata['error_2d'] = np.array(error_2d)
    matdata['error_relf'] = np.array(error_relf)
    matdata['error_rel3d'] = np.stack(error_rel3d)
    scipy.io.savemat(outfile,matdata)

    print(f"MEAN seterror_2d: {np.mean(error_2d)}")
    print(f"MEAN seterror_rel3d: {np.mean(error_rel3d)}")
    print(f"MEAN seterror_relf: {np.mean(error_relf)}")

def test(modelin=args.model,outfile=args.out,feature_transform=args.ft):

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
    le = torch.mean(mu_lm[36:42,:],axis=0)
    re = torch.mean(mu_lm[42:48,:],axis=0)
    ipd = torch.norm(le - re)
    lm_eigenvec = torch.from_numpy(data3dmm.lm_eigenvec).float()
    #optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)

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

    # set random seed for reproducibility of test set
    for f_test in f_vals:
        # create dataloader
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

        for j, data in enumerate(loader):
            if j == 10: break
            # create bpnp camera calibration model
            calib_net= (1.1*torch.randn(1)).requires_grad_()

            # create bpnp sfm model
            sfm_net  = torchvision.models.vgg11()
            sfm_net.classifier = torch.nn.Linear(25088,N*3)

            # load the data
            x_cam_gt = data['x_cam_gt']
            shape_gt = data['x_w_gt']
            fgt = data['f_gt']
            x_img = data['x_img']
            x_img_gt = data['x_img_gt']

            depth = torch.norm(x_cam_gt.mean(2),dim=1)
            all_depth.append(depth.numpy())
            all_f.append(fgt.numpy()[0])

            ptsI = x_img.reshape((M,N,2)).permute(0,2,1)
            x2d = x_img.view((M,N,2))
            x_img_pts = x_img.reshape((M,N,2)).permute(0,2,1)
            one = torch.ones(M*N,1)
            x_img_one = torch.cat([x_img,one],dim=1)
            x = x_img_one.permute(1,0)

            # run the model
            f = torch.sigmoid(calib_net)*2000
            shape = mu_lm
            ini_pose = torch.zeros((M,6))
            ini_pose[:,5] = 99
            curloss = 100

            # apply dual optimization
            shape,K,R,T = dualoptimization(x,ptsI,x2d,ini_pose,calib_net,sfm_net,shape_gt=shape_gt,fgt=fgt)
            f = K[0,0].detach()
            all_fpred.append(f.item())

            # get errors
            km,c_w,scaled_betas, alphas = util.EPnP(ptsI,shape,K)
            Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI,K)

            # get errors
            reproj_errors2 = util.getReprojError2(ptsI,shape,R,T,K)
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

            print(f"f/sequence: {f_test}/{j}  | f/fgt: {f.item():.3f}/{fgt.item():.3f} |  f_error_rel: {f_error.item():.4f}  | rmse: {reconstruction_error.item():.4f}  | rel rmse: {rel_error.item():.4f}    | 2d error: {reproj_error.item():.4f}")
            #end for

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
        #end for

    all_f = np.stack(all_f).flatten()
    all_fpred = np.stack(all_fpred).flatten()
    all_d = np.stack(all_depth).flatten()
    allerror_2d = np.stack(allerror_2d).flatten()
    allerror_3d = np.stack(allerror_3d).flatten()
    allerror_rel3d = np.stack(allerror_rel3d).flatten()

    matdata = {}
    #matdata['shape'] = shape.detach().cpu().numpy()
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
    else:
        testReal()
