
import itertools
import argparse
import os

from numpy.core.records import fromarrays
import scipy.io
import torch
import numpy as np

from model2 import PointNet
import dataloader
import util
import time

####################################################

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
mu_lm = torch.from_numpy(data3dmm.mu_lm).float().detach()
mu_lm[:,2] = mu_lm[:,2]*-1
lm_eigenvec = torch.from_numpy(data3dmm.lm_eigenvec).float().detach()
sigma = torch.from_numpy(data3dmm.sigma).float().detach()
sigma = torch.diag(sigma.squeeze())
lm_eigenvec = torch.mm(lm_eigenvec, sigma)

# HELPER FUNCTIONS
def trainfc(model):
    for name, param in model.named_parameters():
        if 'fc' in name and 'feat' not in name:
            param.requires_grad = True

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

##############################################################################################
##############################################################################################
##############################################################################################

# dual optimization to optimize focal length and 3D shape
def dualoptimization(ptsI,calib_net,sfm_net,shape_gt=None,fgt=None,M=100,N=68,mode='still',ptstart=0,db='real'):

    if mode == 'still':
        alpha = 1
    else:
        alpha = 0.001
    alpha = 0.001

    # define what weights gets optimized
    calib_net.eval()
    sfm_net.eval()
    #calib_net.train()
    #sfm_net.train()
    trainfc(calib_net)
    trainfc(sfm_net)
    M = ptsI.shape[0]
    xmin,_ = torch.min(ptsI[:,0,:],dim=1)
    xmax,_ = torch.max(ptsI[:,0,:],dim=1)
    ymin,_ = torch.min(ptsI[:,1,:],dim=1)
    ymax,_ = torch.max(ptsI[:,1,:],dim=1)
    width = torch.abs(xmin - xmax)
    height = torch.abs(ymin - ymax)
    area = width*height

    # run the model
    #ptsI = x.squeeze().permute(1,0).reshape((M,N,2)).permute(0,2,1)
    f = torch.squeeze(calib_net(ptsI) + 300)
    betas = sfm_net(ptsI)
    betas = betas.unsqueeze(-1)
    eigenvec = torch.stack(M * [lm_eigenvec])
    shape = torch.stack(M*[mu_lm]) + torch.bmm(eigenvec,betas).squeeze().view(M,N,3)
    shape = shape - shape.mean(1).unsqueeze(1)
    shape = shape.mean(0)

    opt1 = torch.optim.Adam(calib_net.parameters(),lr=1e-4)
    opt2 = torch.optim.Adam(sfm_net.parameters(),lr=1e-1)
    curloss = 100000
    for outerloop in itertools.count():
        shape = shape.detach()
        for iter in itertools.count():
            opt1.zero_grad()
            f = torch.squeeze(calib_net(ptsI) + 300)
            #f = f.mean()
            K = torch.zeros(M,3,3).float()
            K[:,0,0] = f
            K[:,1,1] = f
            K[:,2,2] = 1

            # pose estimation
            km,c_w,scaled_betas, alphas = util.EPnP_single(ptsI,shape,K)
            _, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI)
            Xc = torch.bmm(R,torch.stack(M*[shape.T])) + T.unsqueeze(2)

            # get time error
            error_time = util.getTimeConsistency(shape,R,T)

            # get shape consistency
            error_s = util.getShapeError(ptsI,Xc,shape,f,R,T)

            # get 2D reprojection error
            error2d = util.getError(ptsI,shape,R,T,K,show=False,loss='l2')

            # get relative depth error
            error3d = util.getDepthError(area,shape,R,T)

            # apply loss
            #loss = error2d.mean()
            #loss = error2d.mean() + error3d
            loss = error2d.mean() + error_time*alpha
            #loss = error_s + error2d.mean()
            #loss = error_s
            #loss = error_s + error_time*alpha
            #loss = error3d

            if iter >= 5: break
            prv_loss = loss.item()
            loss.backward()
            opt1.step()

            # log results on console
            if not shape_gt is None:
                rmse = torch.norm(shape_gt - shape,dim=1).mean().item()
            else:
                rmse = -1
            if not fgt is None:
                ftrue = fgt.item()
            else:
                fgt = -1
            f_error = torch.mean(torch.abs(f-ftrue))
            print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.mean().item():.3f}/{f.median().item():.3f}/{f.std().item():.3f}/{fgt.item():.3f} | f_error: {f_error.item():.1f} | error2d: {error2d.mean().item():.3f} | rmse: {rmse:.2f}")

        f = f.detach()
        for iter in itertools.count():
            opt2.zero_grad()

            # shape prediction
            betas = sfm_net(ptsI)
            betas = betas.unsqueeze(-1)
            eigenvec = torch.stack(M * [lm_eigenvec])
            shape = torch.stack(M*[mu_lm]) + torch.bmm(eigenvec,betas).squeeze().view(M,N,3)
            shape = shape - shape.mean(1).unsqueeze(1)
            shape = shape.mean(0)
            K = torch.zeros(M,3,3).float()
            K[:,0,0] = f
            K[:,1,1] = f
            K[:,2,2] = 1

            # differentiable PnP pose estimation
            km,c_w,scaled_betas,alphas = util.EPnP_single(ptsI,shape,K)
            _, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI)
            Xc = torch.bmm(R,torch.stack(M*[shape.T])) + T.unsqueeze(2)

            # get time error
            error_time = util.getTimeConsistency(shape,R,T)

            # get shape consistency
            error_s = util.getShapeError(ptsI,Xc,shape,f,R,T)

            # error2d
            error2d = util.getError(ptsI,shape,R,T,K,show=False,loss='l2')
            #Xc = torch.bmm(R,torch.stack(M*[shape.T])) + T.unsqueeze(2)

            # get relative depth error
            error3d = util.getDepthError(area,shape,R,T)

            # apply loss
            #loss = error2d.mean()
            #loss = error2d.mean()+ error3d
            loss = error2d.mean()+ error_time*alpha
            #loss = error_s + error2d.mean()
            #loss = error_s
            #loss = error_s + error_time*alpha
            #loss = error3d

            if iter >= 5: break
            loss.backward()
            opt2.step()
            prv_loss = loss.item()

            # log results on console
            if not shape_gt is None:
                rmse = torch.norm(shape_gt - shape,dim=1).mean().item()
            else:
                rmse = -1
            if not fgt is None:
                ftrue = fgt.item()
            else:
                fgt = -1
            f_error = torch.mean(torch.abs(f-ftrue))
            print(f"iter: {iter} | error: {loss.item():.3f} | f/fgt: {f.mean().item():.3f}/{f.median().item():.3f}/{f.std().item():.3f}/{fgt.item():.3f} | f_error: {f_error.item():.1f} | error2d: {error2d.mean().item():.3f} | rmse: {rmse:.2f}")

        if torch.abs(curloss  - loss) <= 0.01 or curloss < loss: break
        curloss = loss

    return shape,K,R,T,outerloop

##############################################################################################
##############################################################################################
##############################################################################################
# testing on different datasets
def testReal(modelin=args.model,outfile=args.out,optimize=args.opt,db=args.db):
    # define model, dataloader, 3dmm eigenvectors, optimization method
    calib_net = PointNet(n=1)
    sfm_net = PointNet(n=199)
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
    loader = getLoader(db)
    out_fpred = []
    out_fgt = []
    out_dpred = []
    out_dgt = []
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
        N = x_img_gt.shape[-1]

        ptsI = x_img.reshape((M,N,2)).permute(0,2,1)
        x = x_img.unsqueeze(0).permute(0,2,1)

        # run the model
        f = torch.squeeze(calib_net(ptsI) + 300)
        betas = sfm_net(ptsI)
        betas = betas.unsqueeze(-1)
        eigenvec = torch.stack(M * [lm_eigenvec])
        shape = torch.stack(M*[mu_lm]) + torch.bmm(eigenvec,betas).squeeze().view(M,N,3)
        shape = shape - shape.mean(1).unsqueeze(1)
        shape = shape.mean(0)

        # get motion measurement guess
        K = torch.zeros((M,3,3)).float()
        K[:,0,0] = f
        K[:,1,1] = f
        K[:,2,2] = 1
        km,c_w,scaled_betas,alphas = util.EPnP_single(ptsI,shape,K)
        _, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI)
        error_time = util.getTimeConsistency(shape,R,T)
        if error_time > 30:
            mode='walk'
        else:
            mode='still'
        print(mode,error_time)

        # additional optimization on initial solution
        shape_gt = batch['x_w_gt'] if db == 'biwi' else None
        if optimize:
            calib_net.load_state_dict(torch.load(calib_path))
            sfm_net.load_state_dict(torch.load(sfm_path))
            print(mode)
            if db == 'biwi':
                shape,K,R,T,iter = dualoptimization(ptsI,calib_net,sfm_net,shape_gt=shape_gt,fgt=fgt,db='biwi',mode=mode)
            else:
                shape,K,R,T,iter = dualoptimization(ptsI,calib_net,sfm_net,fgt=fgt,mode=mode)
            f = K[:,0,0].detach()

        # get pose with single intrinsic
        fmu = f.mean()
        fmed = f.flatten().median()
        K = torch.zeros(M,3,3).float()
        K[:,0,0] = fmu
        K[:,1,1] = fmu
        K[:,2,2] = 1
        km,c_w,scaled_betas,alphas = util.EPnP_single(ptsI,shape,K)
        Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI)

        # get errors
        #reproj_errors2 = util.getError(ptsI,shape,R,T,K)
        reproj_errors2 = util.getReprojError2(ptsI,shape,R,T,K.mean(0),show=False,loss='l2')
        rel_errors = util.getRelReprojError3(x_cam_gt,shape,R,T)
        d = torch.norm(T,dim=1)
        dgt = torch.norm(torch.mean(x_cam_gt,dim=2),dim=1)

        reproj_error = reproj_errors2.mean()
        rel_error = rel_errors.mean()
        f_error = torch.mean(torch.abs(fgt - fmu) / fgt)

        # save final prediction
        out_fpred.append(f.detach().cpu().numpy())
        out_fgt.append(fgt.numpy())
        out_dpred.append(d.detach().cpu().numpy())
        out_dgt.append(dgt.cpu().numpy())
        f_x = torch.mean(fmu.detach()).cpu().item()
        shape_pred.append(shape.detach().cpu().numpy())

        error_2d.append(reproj_error.cpu().data.item())
        error_rel3d.append(rel_error.cpu().data.item())
        error_relf.append(f_error.cpu().data.item())

        print(f" f/fgt: {f_x:.3f}/{fgt.item():.3f} |  f_error_rel: {f_error.item():.4f}  | rel rmse: {rel_error.item():.4f}    | 2d error: {reproj_error.item():.4f}")
        #end for

    # prepare output file
    out_shape = np.stack(shape_pred)
    out_fpred = np.array(out_fpred,dtype=np.object)
    out_fgt = np.array(out_fgt,dtype=np.object).T

    matdata = {}
    matdata['fpred'] = out_fpred
    matdata['fgt'] = out_fgt
    matdata['dpred'] = out_dpred
    matdata['dgt'] = out_dgt
    matdata['shape'] = np.stack(out_shape)
    matdata['error_2d'] = np.array(error_2d)
    matdata['error_rel3d'] = np.array(error_rel3d)
    matdata['error_relf'] = np.array(error_relf)
    scipy.io.savemat(outfile,matdata)

    print(f"MEAN seterror_2d: {np.mean(error_2d)}")
    print(f"MEAN seterror_rel3d: {np.mean(error_rel3d)}")
    print(f"MEAN seterror_relf: {np.mean(error_relf)}")


def test(modelin=args.model,outfile=args.out,optimize=args.opt,ft=args.ft):
    # define model, dataloader, 3dmm eigenvectors, optimization method
    calib_net = PointNet(n=1,feature_transform=ft)
    sfm_net = PointNet(n=199,feature_transform=ft)
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
    all_depth = []
    out_shape = []
    out_fpred = []
    out_fgt = []
    out_dpred = []
    out_dgt = []
    seterror_3d = []
    seterror_rel3d = []
    seterror_relf = []
    seterror_2d = []
    iterations = []
    exec_time = []
    f_vals = [i*100 for i in range(4,15)]

    # set random seed for reproducibility of test set
    np.random.seed(0)
    torch.manual_seed(0)
    for f_test in f_vals:
        # create dataloader
        loader = dataloader.TestLoader(f_test)

        shape_pred = []
        error_2d = []
        error_3d = []
        error_rel3d = []
        error_relf = []
        #M = 100;
        #N = 68;
        batch_size = 1;
        loader.M = 50
        loader.N = 68

        for j,data in enumerate(loader):
            if j >= 3: break
            # load the data
            x_cam_gt = data['x_cam_gt']
            shape_gt = data['x_w_gt']
            fgt = data['f_gt']
            x_img = data['x_img']
            x_img_gt = data['x_img_gt']
            M = loader.M
            N = loader.N

            depth = torch.norm(x_cam_gt.mean(2),dim=1)
            all_depth.append(depth.numpy())
            all_f.append(fgt.numpy()[0])

            ptsI = x_img.reshape((M,N,2)).permute(0,2,1)
            x = x_img.unsqueeze(0).permute(0,2,1)

            # run the model
            f = torch.squeeze(calib_net(ptsI) + 300)
            betas = sfm_net(ptsI)
            betas = betas.unsqueeze(-1)
            eigenvec = torch.stack(M * [lm_eigenvec])
            shape = torch.stack(M*[mu_lm]) + torch.bmm(eigenvec,betas).squeeze().view(M,N,3)
            shape = shape - shape.mean(1).unsqueeze(1)
            shape = shape.mean(0)

            # get motion measurement guess
            K = torch.zeros((M,3,3)).float()
            K[:,0,0] = f
            K[:,1,1] = f
            K[:,2,2] = 1
            km,c_w,scaled_betas,alphas = util.EPnP_single(ptsI,shape,K)
            _, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI)
            error_time = util.getTimeConsistency(shape,R,T)
            if error_time > 30:
                mode='walk'
            else:
                mode='still'
            print(mode,error_time)

            # apply dual optimization
            if optimize:
                calib_net.load_state_dict(torch.load(calib_path))
                sfm_net.load_state_dict(torch.load(sfm_path))
                st_time = time.time()
                shape,K,R,T,iter = dualoptimization(ptsI,calib_net,sfm_net,shape_gt=shape_gt,fgt=fgt,mode=mode)
                end_time = time.time()
                f = K[:,0,0].detach()
                t = end_time - st_time
                iterations.append(iter)
                exec_time.append(t)
                print(iterations, t)
            else:
                K = torch.zeros(M,3,3).float()
                K[:,0,0] = f
                K[:,1,1] = f
                K[:,2,2] = 1
                km,c_w,scaled_betas,alphas = util.EPnP_single(ptsI,shape,K)
                Xc, R, T, mask = util.optimizeGN(km,c_w,scaled_betas,alphas,shape,ptsI)

            # get errors
            reproj_errors2 = util.getError(ptsI,shape,R,T,K,show=False)
            reproj_errors3 = torch.norm(shape_gt - shape,dim=1).mean()
            rel_errors =  util.getRelReprojError3(x_cam_gt,shape,R,T)
            d = torch.norm(T,dim=1)
            dgt = torch.norm(torch.mean(x_cam_gt,dim=2),dim=1)

            reproj_error = reproj_errors2.mean()
            reconstruction_error = reproj_errors3.mean()
            rel_error = rel_errors.mean()
            f_error = torch.mean(torch.abs(fgt - f) / fgt)

            # save final prediction
            out_fpred.append(f.detach().cpu().numpy())
            out_fgt.append(fgt.numpy())
            out_dpred.append(d.detach().cpu().numpy())
            out_dgt.append(dgt.detach().cpu().numpy())
            shape_pred.append(shape.detach().cpu().numpy())
            f = torch.mean(f)

            allerror_3d.append(reproj_error.data.numpy())
            allerror_2d.append(reconstruction_error.data.numpy())
            allerror_rel3d.append(rel_error.data.numpy())
            error_2d.append(reproj_error.cpu().data.item())
            error_3d.append(reconstruction_error.cpu().data.item())
            error_rel3d.append(rel_error.cpu().data.item())
            error_relf.append(f_error.cpu().data.item())

            print(f"f/sequence: {f_test}/{j}  | f/fgt: {f.item():.3f}/{fgt.item():.3f} |  f_error_rel: {f_error.item():.4f}  | rmse: {reconstruction_error.item():.4f}  | rel rmse: {rel_error.item():.4f}    | 2d error: {reproj_error.item():.4f}")

        avg_2d = np.mean(error_2d)
        avg_rel3d = np.mean(error_rel3d)
        avg_3d = np.mean(error_3d)
        avg_relf = np.mean(error_relf)

        seterror_2d.append(avg_2d)
        seterror_3d.append(avg_3d)
        seterror_rel3d.append(avg_rel3d)
        seterror_relf.append(avg_relf)
        out_shape.append(np.concatenate(shape_pred,axis=0))
        print(f"f_error_rel: {avg_relf:.4f}  | rel rmse: {avg_rel3d:.4f}    | 2d error: {avg_2d:.4f} |  rmse: {avg_3d:.4f}  |")

    # save output
    out_shape = np.stack(out_shape)
    out_fpred = np.array(out_fpred,dtype=np.object)
    out_fgt = np.array(out_fgt,dtype=np.object)
    out_dpred = np.array(out_dpred,dtype=np.object)
    out_dgt = np.array(out_dgt,dtype=np.object)
    all_f = np.stack(all_f).flatten()
    allerror_2d = np.stack(allerror_2d).flatten()
    allerror_rel3d = np.stack(allerror_rel3d).flatten()
    print(np.mean(iterations))

    matdata = {}
    matdata['exec_time'] = np.array(exec_time)
    matdata['iterations'] = np.array(iterations)
    matdata['fvals'] = np.array(f_vals)
    matdata['all_f'] = np.array(all_f)
    matdata['fpred'] = out_fpred
    matdata['fgt'] = out_fgt
    matdata['dpred'] = out_dpred
    matdata['dgt'] = out_dgt
    matdata['error_2d'] = allerror_2d
    matdata['error_3d'] = allerror_3d
    matdata['error_rel3d'] = allerror_rel3d
    matdata['seterror_2d'] = np.array(seterror_2d)
    matdata['seterror_3d'] = np.array(seterror_3d)
    matdata['seterror_rel3d'] = np.array(seterror_rel3d)
    matdata['seterror_relf'] = np.array(seterror_relf)
    matdata['shape'] = np.stack(out_shape)
    scipy.io.savemat(outfile,matdata)

    print(f"MEAN seterror_2d: {np.mean(seterror_2d)}")
    print(f"MEAN seterror_3d: {np.mean(seterror_3d)}")
    print(f"MEAN seterror_rel3d: {np.mean(seterror_rel3d)}")
    print(f"MEAN seterror_relf: {np.mean(seterror_relf)}")

    return np.mean(seterror_relf)


####################################################################################3
if __name__ == '__main__':

    if args.db == 'syn':
        test()
    else:
        testReal()

