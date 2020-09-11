
import math

import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import pptk

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        #m.weight.data.fill_(0.00)
        m.bias.data.fill_(0.00)
    if type(m) == torch.nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        #m.weight.data.fill_(0.00)
        m.bias.data.fill_(0.00)

# control points set as arbitrary basis vector in R4
#Input:
#   Pw              (N,3)
#Output:
#   control_w
def getControlPoints(Pw):
    if Pw.is_cuda:
        control_w = torch.zeros((4,4)).float().cuda()
    else:
        control_w = torch.zeros((4,4)).float()

    control_w[0,0] = 1
    control_w[1,1] = 1
    control_w[2,2] = 1
    control_w[-1,:] = 1

    return control_w

# control points set as arbitrary basis vector in R4
#Input:
#   Pw              (b,N,3)
#Output:
#   control_w       (bx4x4)
def getBatchControlPoints(Pw):
    b = Pw.shape[0]
    if Pw.is_cuda:
        control_w = torch.zeros((b,4,4)).float().cuda()
    else:
        control_w = torch.zeros((b,4,4)).float()

    control_w[:,0,0] = 1
    control_w[:,1,1] = 1
    control_w[:,2,2] = 1
    control_w[:,-1,:] = 1

    return control_w


'''     control points using svd and finding principal components
#Input:
# Pw                (N,3)
#
#Output:
# control_w         (4,4)
def getControlPoints(Pw):
    if Pw.is_cuda:
        ones = torch.ones((1,4)).cuda()
    else:
        ones = torch.ones((1,4))
    c = torch.mean(Pw,dim=0)
    centered = Pw-c
    u,d,v = torch.svd(centered)
    control_w = v + c.unsqueeze(1)
    control_w = torch.cat([control_w,c.unsqueeze(1)],dim=1)
    control_w = torch.cat([control_w,ones],dim=0)

    return control_w
'''

#Input:
# Pw                (N,3)
# control_w         (4,4)
#
#Output:
# alphas            (3,N)
def solveAlphas(Pw,control_w):
    N = Pw.shape[0]
    if Pw.is_cuda:
        ones = torch.ones((1,N)).cuda()
    else:
        ones = torch.ones((1,N))
    ph_w = torch.cat([Pw.T,ones],dim=0)
    alphas, LU = torch.solve(ph_w,control_w)

    return alphas


#Input:
# Pw                (b,N,3)
# control_w         (b,4,4)
#
#Output:
# alphas            (b,3,N)
def batchSolveAlphas(Pw,control_w):
    b = Pw.shape[0]
    if Pw.is_cuda:
        ones = torch.ones((b,1,68)).cuda()
    else:
        ones = torch.ones((b,1,68))

    ph_w = torch.cat([Pw.permute(0,2,1),ones],dim=1)
    alphas, LU = torch.solve(ph_w,control_w)

    return alphas


#Input:
#   alphas                  (3,N)
#   p_img                   (M,N,2)
#   px                      scalar
#   py                      scalar
#   f                       scalar
#Output:
#
#   M                       (M,2*N,12)
def setupM(alphas,p_img,px,py,f):
    views = p_img.shape[0]
    N = p_img.shape[1]
    if p_img.is_cuda:
        M = torch.zeros((views,2*N,12)).float().cuda()
    else:
        M = torch.zeros((views,2*N,12))

    M[:,0::2,0] = alphas[0,:]*f
    M[:,0::2,1] = 0
    M[:,0::2,2] = alphas[0,:].unsqueeze(0) * (px - p_img[:,:,0])
    M[:,0::2,3] = alphas[1,:]*f
    M[:,0::2,4] = 0
    M[:,0::2,5] = alphas[1,:].unsqueeze(0) * (px-p_img[:,:,0])
    M[:,0::2,6] = alphas[2,:]*f
    M[:,0::2,7] = 0
    M[:,0::2,8] = alphas[2,:].unsqueeze(0) * (px-p_img[:,:,0])
    M[:,0::2,9] = alphas[3,:]*f
    M[:,0::2,10] = 0
    M[:,0::2,11] = alphas[3,:].unsqueeze(0) * (px-p_img[:,:,0])
    M[:,1::2,0] = 0
    M[:,1::2,1] = alphas[0,:]*f
    M[:,1::2,2] = alphas[0,:].unsqueeze(0) * (py-p_img[:,:,1])
    M[:,1::2,3] = 0
    M[:,1::2,4] = alphas[1,:]*f
    M[:,1::2,5] = alphas[1,:].unsqueeze(0) * (py-p_img[:,:,1])
    M[:,1::2,6] = 0
    M[:,1::2,7] = alphas[2,:]*f
    M[:,1::2,8] = alphas[2,:].unsqueeze(0) * (py-p_img[:,:,1])
    M[:,1::2,9] = 0
    M[:,1::2,10] = alphas[3,:]*f
    M[:,1::2,11] = alphas[3,:].unsqueeze(0) * (py-p_img[:,:,1])

    return M

#Input:
# control_w             (4x4)
#
#Output:
# d12                   scalar
# d13                   scalar
# d14                   scalar
# d23                   scalar
# d24                   scalar
# d34                   scalar
#
def getDistances(control_w):
    p1 = control_w[:3,0]
    p2 = control_w[:3,1]
    p3 = control_w[:3,2]
    p4 = control_w[:3,3]

    d12 = torch.norm(p1-p2,p=2)
    d13 = torch.norm(p1-p3,p=2)
    d14 = torch.norm(p1-p4,p=2)
    d23 = torch.norm(p2-p3,p=2)
    d24 = torch.norm(p2-p4,p=2)
    d34 = torch.norm(p3-p4,p=2)

    return d12,d13,d14,d23,d24,d34

# scale control points and get camera coordinates via method similar to procrustes
#Input:
#   c_c                 (Mx3x4)
#   c_w                 (3x4)
#   alphas              (4xN)
#   x_w                 (Nx3)
#
#Output:
#   sc_c                (Mx3x4)
#   sx_c                (MxNx3)
#   s                   (M)
def scaleControlPoints(c_c,c_w,alphas,x_w):
    views = c_c.shape[0]
    rep_alpha = torch.stack(views*[alphas])
    x_c = torch.bmm(c_c,rep_alpha)

    # center x_w and x_c
    centered_xw = x_w - torch.mean(x_w,dim=0).unsqueeze(0)
    d_w = torch.norm(centered_xw,p=2,dim=1)
    centered_xc = x_c - torch.mean(x_c,dim=2).unsqueeze(2)
    d_c = torch.norm(centered_xc,p=2,dim=1)

    #s = print(d_w.shape)
    #s = torch.mean(d_c / d_w.unsqueeze(0),axis=1)

    # least square solution to scale
    s = 1.0 /((1.0/(torch.sum(d_c*d_c,dim=1)) * torch.sum(d_c*d_w.unsqueeze(0),dim=1)))

    # apply scale onto c_c and recompute the camera coordinates
    sc_c = c_c / s.unsqueeze(1).unsqueeze(1)
    sx_c = torch.bmm(sc_c,rep_alpha).permute(0,2,1)

    # fix the sign so negative depth is not possible
    negdepth_mask = sx_c[:,:,-1] < 0
    sx_c = sx_c * (negdepth_mask.float().unsqueeze(2) * -2 +1)

    return sc_c, sx_c, s

# Method for getting extrinsics after finding world coordinates and the camera coord
# solved using idea similar to procrustes and closest Rotation matrix
#Input:
#   x_c                 (MxNx3)
#   p_w                 (Nx3)
#Output:
#   R                   (Mx3x3)
#   T                   (Mx3x1)
def getExtrinsics(x_c,p_w):
    c_center = torch.mean(x_c,dim=1)
    w_center = torch.mean(p_w,dim=0)

    # center the  3d shapes
    x_c_centered = x_c - c_center.unsqueeze(1)
    p_w_centered = p_w - w_center.unsqueeze(0)

    # create martrix to solve
    M = x_c.shape[0]
    if p_w.is_cuda:
        Matrix = torch.zeros((M,3,3)).cuda()
    else:
        Matrix = torch.zeros((M,3,3))
    for i in range(M):
        m1 = x_c_centered[i].unsqueeze(2)
        m2 = p_w_centered.unsqueeze(1)
        tmpM = torch.bmm(m1,m2)
        Matrix[i] = torch.sum(tmpM,dim=0)

    u,d,v = torch.svd(Matrix)

    '''
    try:
        u,d,v = torch.svd(Matrix)
        mask = torch.ones(M,dtype=torch.bool)
    except:
        mask = Matrix == Matrix
        mask = torch.all(mask.reshape(M,-1),axis=-1)
        u,d,v = torch.svd(Matrix[mask])
        M = torch.sum(mask.long()).item()
        print(M)
    '''

    R = torch.bmm(u,v.permute(0,2,1))

    #rdet = torch.det(r)
    #redetmask = (rdet < 0).float() * -2 + 1
    #redetmask = redetmask.detach()
    #R = r * redetmask.unsqueeze(1).unsqueeze(1)

    # solve T using centers
    rep_w_center = torch.stack(M * [w_center]).unsqueeze(2)
    rot_w_center = torch.bmm(R,rep_w_center)
    T = c_center - rot_w_center.squeeze()

    return R,T

# prot of matlab's procrustes analysis function in numpy
# https://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

# pts       (b,M,2,N)
# shape     (b,N,3)
# K         (3,3)
# xc_gt     (b,M,3,N)
# xw_gt     (b,N,3)
def getBatchError(pts,shape,K,xc_gt,xw_gt):
    e2d = []
    e3d = []
    e2d_all = []
    e3d_all = []
    d_all = []
    b = pts.shape[0]
    for i in range(b):
        pW = shape[i]
        pI = pts[i]
        pC = xc_gt[i]
        M = pI.shape[0]
        km, c_w, scaled_betas, alphas = EPnP(pI,pW,K)
        Xc, R, T, mask = optimizeGN(km,c_w,scaled_betas,alphas,pW,pI,K)

        error2d = getReprojError2(pI,pW,R,T,K,loss='l2')
        error3d = getRelReprojError3(pC,pW,R,T)
        d = torch.norm(pC,p=2,dim=1).mean(1)

        e2d.append(error2d.mean())
        e3d.append(error3d.mean())
        e2d_all.append(error2d)
        e3d_all.append(error3d)
        d_all.append(d)

    errorShape = torch.mean(torch.norm(xw_gt - shape,dim=2),dim=1)

    return torch.stack(e2d),torch.stack(e3d),errorShape,torch.stack(e2d_all),torch.stack(e3d_all),torch.stack(d_all)



def getError2(pimg,pcam,A,show=False):
    M = pimg.shape[0]
    N = pimg.shape[1]

    proj = torch.bmm(torch.stack(M*[A]),pcam)
    proj = proj_img = proj / proj[:,-1,:].unsqueeze(1)
    pimg_pred = proj_img[:,:2,:]
    diff = pimg - pimg_pred
    error = torch.mean(torch.norm(diff,p=2,dim=1),dim=1)

    return error

# batched reprojection error using intrinsics and extrinsics on world coordinates
#
#INPUT:
#   xcam                (M,3,N)
#   pw                  (N,3)
#   R                   (M,3,3)
#   T                   (M,3)
#   A                   (3,3)
#
#OUTPUT:
#   error               (M)
def getReprojError3(xcam,pw,R,T):

    M = xcam.shape[0]
    pc = torch.bmm(R,torch.stack(M*[pw]).permute(0,2,1))
    pct = pc + T.unsqueeze(2)
    diff = pct - xcam
    error  = torch.mean(torch.norm(diff,p=2,dim=1),dim=1)

    return error

def getReprojError(xcam,pw,R,T):
    M = xcam.shape[0]
    pc = torch.bmm(R,torch.stack(M*[pw]).permute(0,2,1))
    pct = pc + T.unsqueeze(2)

    error = torch.mean(torch.abs(pct - xcam))

    return error

#xcam = torch.Size([16, 3, 6800])
#ximg = torch.Size([16, 6800, 3])
#kinv = torch.Size([16, 3, 3])
def getPCError(xcam,ximg,kinv,mode='l1'):

    #torch.set_printoptions(profile='full')
    M = xcam.shape[0]
    proj = torch.bmm(kinv,ximg.permute(0,2,1))

    xcam_pred = proj*xcam[:,2,:].unsqueeze(1)
    if mode == 'l1':
        return torch.nn.functional.l1_loss(xcam_pred[:,:2,:],xcam[:,:2,:])
    else:
        return torch.nn.functional.mse_loss(xcam_pred[:,:2,:],xcam[:,:2,:])

#INPUT
# pI        (b,M*N,3)
# pW        (b,N,3)
# pC        (b,3,M*N)
# kinv      (b,3,3)
#OUTPUT
# scalar
def getShapeError(pI,pW,pC,kinv):

    b = pI.shape[0]
    N = pW.shape[1]
    M = pI.shape[1] // N

    pI_proj = torch.bmm(kinv,pI.permute(0,2,1))
    z = pC[:,2,:].unsqueeze(1)
    pC_proj = pI_proj * z

    le_gt = torch.mean(pW[:,36:42,:],dim=1)
    re_gt = torch.mean(pW[:,42:48,:],dim=1)
    #d_gt = torch.sum(torch.pow(le_gt - re_gt,2),dim=1)
    d_gt = torch.norm(le_gt - re_gt,dim=1)
    proj = pC_proj.permute(0,2,1).reshape(b,M,N,3)

    le = torch.mean(proj[:,:,36:42,:],dim=-2)
    re = torch.mean(proj[:,:,42:48,:],dim=-2)
    #d = torch.sum(torch.pow(le - re,2),dim=2)
    d = torch.norm(le - re,dim=2)

    error = torch.mean(torch.abs(d - d_gt.unsqueeze(1)))

    return error

#INPUT
# pI        (M,2,N)
# pC        (M,3,N)
def solvef(pI,pC):
    M = pI.shape[0]
    N = pI.shape[2]
    fx = (pC[:,2,:] / pC[:,0,:]) * (pI[:,0,:] - 1)
    fy = (pC[:,2,:] / pC[:,1,:])* (pI[:,1,:] - 1)
    f = torch.cat([fx,fy]).mean()
    return f

#xcam = torch.Size([16, 3, 6800])
#ximg = torch.Size([16, 6800, 3])
#kinv = torch.Size([16, 3, 3])
def getRelPCError(xcam,ximg,kinv,mode='l1'):

    #torch.set_printoptions(profile='full')
    M = xcam.shape[0]
    proj = torch.bmm(kinv,ximg.permute(0,2,1))

    xcam_pred = proj*xcam[:,2,:].unsqueeze(1)
    diff = xcam_pred[:,:2,:] - xcam[:,:2,:]
    if mode == 'l1':
        l1_reldiff = torch.mean(torch.abs(diff) / xcam[:,2,:])
        return l1_reldiff
    else:
        l2_reldiff = torch.mean(torch.log(torch.sum(torch.pow(diff,2),1) / torch.pow(xcam[:,2,:],2)))
        return l2_reldiff

# getTimeConsistency(pW,R,T)
def getTimeConsistency(pW,R,T):
    M = R.shape[0]
    pc = torch.bmm(R,torch.stack(M*[pW]).permute(0,2,1))
    pct = pc + T.unsqueeze(2)

    error = torch.mean(torch.norm(pct[:M-1,:,:] - pct[1:,:,:],dim=1))
    return error

#INPUT:
# T     (M,3)
#OUTPUT:
# SCALAR
def getTConsistency(T):

    t1 = T[1:]
    t2 = T[:-1]

    error = torch.mean(torch.sum(torch.pow(t1 - t2,2),1))

    return error

#INPUT:
# R     (M,3,3)
#OUTPUT:
# SCALAR
def getRConsistency(R):

    r1 = R[1:]
    r2 = R[:-1]

    error = torch.mean(torch.pow(r1 - r2,2))
    return error

#INPUT:
# pI        (M,2,N)
# pW        (N,3)
# kinv      (3,3)
# R         (M,3,3)
# T         (M,3)
#OUTPUT:
# SCALAR
def get3DConsistency(pI,pW,kinv,R,T):
    kinv[0,0] = 1/100
    kinv[1,1] = 1/100
    M = pI.shape[0]
    N = pI.shape[2]

    pC = torch.bmm(R,torch.stack(M*[pW.T])) + T.unsqueeze(2)
    z = pC[:,2,:]
    ones = torch.ones(M,1,N).to(pI.device)

    pI_h = torch.cat((pI,ones),dim=1)
    pI_proj = torch.bmm(torch.stack(M*[kinv]),pI_h)
    pC_proj = pI_proj * z.unsqueeze(1)

    le_gt = torch.mean(pW[36:42,:],dim=0)
    re_gt = torch.mean(pW[42:48,:],dim=0)
    d_gt = torch.norm(le_gt - re_gt)

    le = torch.mean(pC_proj[:,:,36:42],dim=2)
    re = torch.mean(pC_proj[:,:,42:48],dim=2)
    d = torch.norm(le - re,dim=1)

    return torch.mean(torch.abs(d - d_gt))

# batched reprojection error using intrinsics and extrinsics on world coordinates
#
#INPUT:
#   xcam                (M,3,N)
#   pw                  (N,3)
#   R                   (M,3,3)
#   T                   (M,3)
#   A                   (3,3)
#
#OUTPUT:
#   error               (M)
def getRelReprojError3(xcam,pw,R,T):

    M = xcam.shape[0]
    pc = torch.bmm(R,torch.stack(M*[pw]).permute(0,2,1))
    pct = pc + T.unsqueeze(2)

    #import pptk
    #x1 = xcam[1].T
    #x2 = pct[1].T
    #x1 = x1.cpu().data.numpy()
    #x2 = x2.cpu().data.numpy()
    #pts = np.concatenate((x1,x2),axis=0)

    diff = pct - xcam
    d = torch.norm(xcam,p=2,dim=1)
    error = torch.norm(diff,p=2,dim=1)

    #print(torch.mean(error/d,dim=1).mean())
    #v = pptk.viewer(pts)
    #v.set(point_size=10)
    #quit()
    return torch.mean(error / d,dim=1)

# batched reprojection error using intrinsics and extrinsics on world coordinates
#
#INPUT:
#   pimg                (M,2,N)
#   pw                  (N,3)
#   R                   (M,3,3)
#   T                   (M,3)
#   A                   (3,3)
#
#OUTPUT:
#   error               (M)
def getReprojError2_(pimg,pct,A,show=False,loss='l2'):

    M = pimg.shape[0]
    N = pimg.shape[2]
    proj = torch.bmm(torch.stack(M*[A]),pct)
    proj_img = proj / proj[:,-1,:].unsqueeze(1)

    pimg_pred = proj_img[:,:2,:]
    diff = pimg - pimg_pred
    if loss == 'l2':
        error  = torch.norm(diff,p=2,dim=1).mean(1)
    elif loss == 'l1':
        error = torch.abs(diff)

    if show:
        #import pptk
        #x = pct[-1].T.detach().cpu().numpy()
        #v = pptk.viewer(x)
        #v.set(point_size=1.1)
        #for i in range(M):
        #    pt1 = pimg[i].T.cpu().numpy()
        #    scatter(pt1)
        pta = pimg[0].T.cpu().numpy()
        ptb = pimg[-1].T.cpu().numpy()
        ptc = pimg_pred[0].detach().T.cpu().numpy()
        ptd = pimg_pred[-1].detach().T.cpu().numpy()
        plt.scatter(pta[:,0],pta[:,1],s=15,facecolors='none',edgecolors='green')
        #plt.scatter(ptb[:,0],pta[:,1],s=15,facecolors='none',edgecolors='green')
        plt.scatter(ptb[:,0],ptb[:,1],s=10,marker='.',edgecolors='red')
        #plt.scatter(ptb[:,0],pta[:,1],s=10,marker='.',edgecolors='red')

        #plt.xlim((-320,320))
        #plt.ylim((-240,240))
        plt.show()

        quit()

    return error
# batched reprojection error using intrinsics and extrinsics on world coordinates
#
#INPUT:
#   pimg                (M,2,N)
#   pw                  (N,3)
#   R                   (M,3,3)
#   T                   (M,3)
#   A                   (3,3)
#
#OUTPUT:
#   error               (M)
def getReprojError2(pimg,pw,R,T,A,show=False,loss='l2'):

    M = pimg.shape[0]
    N = pimg.shape[2]
    pc = torch.bmm(R,torch.stack(M*[pw]).permute(0,2,1))
    pct = pc + T.unsqueeze(2)
    proj = torch.bmm(torch.stack(M*[A]),pct)
    proj_img = proj / proj[:,-1,:].unsqueeze(1)

    pimg_pred = proj_img[:,:2,:]
    diff = pimg - pimg_pred
    if loss == 'l2':
        error  = torch.norm(diff,p=2,dim=1).mean(1)
    elif loss == 'l1':
        error = torch.abs(diff)

    if show:
        #import pptk
        #x = pct[-1].T.detach().cpu().numpy()
        #v = pptk.viewer(x)
        #v.set(point_size=1.1)
        #for i in range(M):
        #    pt1 = pimg[i].T.cpu().numpy()
        #    scatter(pt1)

        pta = pimg[0].T.cpu().numpy()
        ptb = pimg[-1].T.cpu().numpy()
        ptc = pimg_pred[0].detach().T.cpu().numpy()
        ptd = pimg_pred[-1].detach().T.cpu().numpy()
        plt.scatter(pta[:,0],pta[:,1],s=15,facecolors='none',edgecolors='green')
        #plt.scatter(ptb[:,0],pta[:,1],s=15,facecolors='none',edgecolors='green')
        plt.scatter(ptc[:,0],ptc[:,1],s=10,marker='.',edgecolors='red')
        #plt.scatter(ptb[:,0],pta[:,1],s=10,marker='.',edgecolors='red')

        #plt.xlim((-320,320))
        #plt.ylim((-240,240))
        plt.show()
        quit()

    return error
# computes the beta values as b11,b12,b22 in that order using distance constraints
#Input:
#   v                   (Mx12x2)
#   d                   (6)
#
#Output:
#   betas               (Mx3x1)
def getBetaN2(v,d):
    views = v.shape[0]
    v1 = v[:,:3,:]
    v2 = v[:,3:6,:]
    v3 = v[:,6:9,:]
    v4 = v[:,9:12,:]

    if v.is_cuda:
        M = torch.zeros((views,6,3)).cuda()
    else:
        M = torch.zeros((views,6,3))

    M00 = torch.sum(v1[:,:,0]**2 - 2*v1[:,:,0]*v2[:,:,0] + v2[:,:,0]**2,dim=1)
    M01 = torch.sum(2*v1[:,:,0]*v1[:,:,1] - 2*v1[:,:,0]*v2[:,:,1] - 2*v1[:,:,1]*v2[:,:,0] + 2*v2[:,:,0]*v2[:,:,1],dim=1)
    M02 = torch.sum(v1[:,:,1]**2 - 2*v1[:,:,1]*v2[:,:,1] + v2[:,:,1]**2,dim=1)
    M10 = torch.sum(v1[:,:,0]**2 - 2*v1[:,:,0]*v3[:,:,0] + v3[:,:,0]**2,dim=1)
    M11 = torch.sum(2*v1[:,:,0]*v1[:,:,1] - 2*v1[:,:,0]*v3[:,:,1] - 2*v1[:,:,1]*v3[:,:,0] + 2*v3[:,:,0]*v3[:,:,1],dim=1)
    M12 = torch.sum(v1[:,:,1]**2 - 2*v1[:,:,1]*v3[:,:,1] + v3[:,:,1]**2,dim=1)
    M20 = torch.sum(v1[:,:,0]**2 - 2*v1[:,:,0]*v4[:,:,0] + v4[:,:,0]**2,dim=1)
    M21 = torch.sum(2*v1[:,:,0]*v1[:,:,1] - 2*v1[:,:,0]*v4[:,:,1] - 2*v1[:,:,1]*v4[:,:,0] + 2*v4[:,:,0]*v4[:,:,1],dim=1)
    M22 = torch.sum(v1[:,:,1]**2 - 2*v1[:,:,1]*v4[:,:,1] + v4[:,:,1]**2,dim=1)
    M30 = torch.sum(v2[:,:,0]**2 - 2*v2[:,:,0]*v3[:,:,0] + v3[:,:,0]**2,dim=1)
    M31 = torch.sum(2*v2[:,:,0]*v2[:,:,1] - 2*v2[:,:,0]*v3[:,:,1] - 2*v2[:,:,1]*v3[:,:,0] + 2*v3[:,:,0]*v3[:,:,1],dim=1)
    M32 = torch.sum(v2[:,:,1]**2 - 2*v2[:,:,1]*v3[:,:,1] + v3[:,:,1]**2,dim=1)
    M40 = torch.sum(v2[:,:,0]**2 - 2*v2[:,:,0]*v4[:,:,0] + v4[:,:,0]**2,dim=1)
    M41 = torch.sum(2*v2[:,:,0]*v2[:,:,1] - 2*v2[:,:,0]*v4[:,:,1] - 2*v2[:,:,1]*v4[:,:,0] + 2*v4[:,:,0]*v4[:,:,1],dim=1)
    M42 = torch.sum(v2[:,:,1]**2 - 2*v2[:,:,1]*v4[:,:,1] + v4[:,:,1]**2,dim=1)
    M50 = torch.sum(v3[:,:,0]**2 - 2*v3[:,:,0]*v4[:,:,0] + v4[:,:,0]**2,dim=1)
    M51 = torch.sum(2*v3[:,:,0]*v3[:,:,1] - 2*v3[:,:,0]*v4[:,:,1] - 2*v3[:,:,1]*v4[:,:,0] + 2*v4[:,:,0]*v4[:,:,1],dim=1)
    M52 = torch.sum(v3[:,:,1]**2 - 2*v3[:,:,1]*v4[:,:,1] + v4[:,:,1]**2,dim=1)

    M[:,0,0] = M00
    M[:,0,1] = M01
    M[:,0,2] = M02
    M[:,1,0] = M10
    M[:,1,1] = M11
    M[:,1,2] = M12
    M[:,2,0] = M20
    M[:,2,1] = M21
    M[:,2,2] = M22
    M[:,3,0] = M30
    M[:,3,1] = M31
    M[:,3,2] = M32
    M[:,4,0] = M40
    M[:,4,1] = M41
    M[:,4,2] = M42
    M[:,5,0] = M50
    M[:,5,1] = M51
    M[:,5,2] = M52

    #mtm_inv_mt = torch.bmm(torch.inverse(torch.bmm(M.permute(0,2,1),M)),M.permute(0,2,1))
    #betas = torch.bmm(mtm_inv_mt,torch.stack(views*[d]).unsqueeze(-1))
    #mtb[0]

    # not sure which would be more stable solution
    # solve for betas using inverse of MtM
    #MtM = torch.bmm(M.permute(0,2,1),M)
    #b = torch.stack(views*[d])
    #MtM_inv = torch.inverse(MtM)
    #Mtb = torch.bmm(M.permute(0,2,1),b.unsqueeze(-1))
    #beta = torch.bmm(MtM_inv,Mtb)
    #beta = torch.bmm(torch.bmm(torch.inverse(MtM), M.permute(0,2,1)),b.unsqueeze(-1))

    # solve lstsq using qr decomposition
    q,r = torch.qr(M)
    b = torch.stack(views*[d])
    qtb = torch.bmm(q.permute(0,2,1),b.unsqueeze(-1))
    betas = torch.bmm(torch.inverse(r),qtb)
    #try:
    #betas = torch.bmm(torch.inverse(r),qtb)
    #except:
    #    noise = 1e-4 * r.mean().detach() * torch.rand(r.shape).to(r.device)
    #    betas = torch.bmm(torch.inverse(r + noise),qtb)

    # solve lstsq using svd on batch
    # https://gist.github.com/gngdb/611d8f180ef0f0baddaa539e29a4200e
    '''
    U_double,D_double,V_double = torch.svd(M.double())
    U = U_double.float()
    D = D_double.float()
    V = V_double.float()
    b = torch.stack(views*[d])
    Utb = torch.bmm(U.permute(0,2,1),b.unsqueeze(-1))
    D_inv = torch.diag_embed(1.0/D)
    VS = torch.bmm(V,D_inv)
    betas = torch.bmm(VS, Utb)
    '''

    return betas

# assumes betas are in order b11,b12,b22
#
#INPUT:
#   v               (Mx12x2)
#   beta            (Mx4)
#OUTPUT:
#   c_c             (Mx3x4)
def getControlPointsN2(v,beta):
    M = v.shape[0]
    b1 = beta[:,2]
    b2 = beta[:,3]
    #b1 = torch.sqrt(torch.abs(beta[:,0]))
    #b2 = torch.sqrt(torch.abs(beta[:,2])) * torch.sign(beta[:,1]) * torch.sign(beta[:,0])
    p = v[:,:,0]*b1.unsqueeze(1) + v[:,:,1]*b2.unsqueeze(1)

    c_c = p.reshape((M,4,3)).permute(0,2,1)
    return c_c

# optimize via gauss newton the betas
# get the optimzed rotation, translation, and camera coord
def optimize_betas_gauss_newton(km, cw, betas, alphas, x_w, x_img, A):

    M = km.shape[0]
    beta_opt, err = gauss_newton(km,cw,betas)

    # compute control points using optimized betas
    kmsum = beta_opt[:,0].unsqueeze(1)*km[:,:,0] + beta_opt[:,1].unsqueeze(1)*km[:,:,1] + beta_opt[:,2].unsqueeze(1)*km[:,:,2] + beta_opt[:,3].unsqueeze(1)*km[:,:,3]
    c_c = kmsum.reshape((M,4,3)).permute(0,2,1)

    # check sign of the determinent to keep orientation
    sign1 = torch.sign(torch.det(cw[:3,:3]))
    sign2 = sign_determinant(c_c)

    # get extrinsics
    cc = c_c * (sign1*sign2).unsqueeze(1).unsqueeze(1)
    rep_alpha = torch.stack(M*[alphas])
    Xc_opt = torch.bmm(cc,rep_alpha)

    return Xc_opt

def gauss_newton(km,cw,betas):
    L = compute_L6_10(km)
    rho = compute_rho(cw)

    n_iterations = 15
    current_betas = betas

    # repeat below code for more iterations of gauss newton, but 4-5 should be enough
    for i in range(n_iterations):
        betas,err = gauss_newton_step(betas,rho,L)
    #betas1, _ = gauss_newton_step(betas,rho,L)
    #betas2, _ = gauss_newton_step(betas1,rho,L)
    #betas3, _ = gauss_newton_step(betas2,rho,L)
    #betas4, _ = gauss_newton_step(betas3,rho,L)
    #betas5, _ = gauss_newton_step(betas4,rho,L)
    #betas6, _ = gauss_newton_step(betas5,rho,L)
    #betas7, _ = gauss_newton_step(betas6,rho,L)
    #betas8, _ = gauss_newton_step(betas7,rho,L)
    #betas9, err = gauss_newton_step(betas8,rho,L)

    return betas, err

def gauss_newton_step(betas,rho,L):
    M = betas.shape[0]
    A,b = computeJacobian(betas,rho,L)
    ata = torch.bmm(A.permute(0,2,1),A)
    q,r = torch.qr(A)
    qtb = torch.bmm(q.permute(0,2,1),b.unsqueeze(-1))
    r_inv = torch.inverse(r)
    #try:
    #   r_inv = torch.inverse(r)
    #except:
    #    noise = 1e-4*r.mean().detach()*torch.rand(r.shape).to(betas.device)
    #    r_inv = torch.inverse(r + noise)

    rinv_qtb = torch.bmm(r_inv,qtb)
    #try:
    #    ata_inv = torch.inverse(ata)
    #except:
    #    ata_inv = torch.inverse(ata + 1e-6*ata.mean()*torch.rand(ata.shape))
    #ata_inv_at = torch.bmm(ata_inv,A.permute(0,2,1))
    #ata_inv_at = torch.bmm(torch.inverse(ata),A.permute(0,2,1))
    #dbeta = torch.bmm(ata_inv_at,b.unsqueeze(-1))
    next_betas = betas.unsqueeze(-1) + rinv_qtb

    error = torch.bmm(b.view((M,1,6)),b.view((M,6,1)))
    return next_betas.squeeze(), error

# compute the derivatives of the eigenvector summation for gauss newton
#
#INPUT:
# km            (M,12,4)
#OUTPUT:
# L             (M,6,10)
def compute_L6_10(km):

    M = km.shape[0]
    L = torch.zeros((M,6,10)).to(km.device)
    v1 = km[:,:,0]
    v2 = km[:,:,1]
    v3 = km[:,:,2]
    v4 = km[:,:,3]

    # compute differenes
    dx112 = v1[:,0] - v1[:,3];
    dx113 = v1[:,0] - v1[:,6];
    dx114 = v1[:,0] - v1[:,9];
    dx123 = v1[:,3] - v1[:,6];
    dx124 = v1[:,3] - v1[:,9];
    dx134 = v1[:,6] - v1[:,9];
    dy112 = v1[:,1] - v1[:,4];
    dy113 = v1[:,1] - v1[:,7];
    dy114 = v1[:,1] - v1[:,10];
    dy123 = v1[:,4] - v1[:,7];
    dy124 = v1[:,4] - v1[:,10];
    dy134 = v1[:,7] - v1[:,10];
    dz112 = v1[:,2] - v1[:,5];
    dz113 = v1[:,2] - v1[:,8];
    dz114 = v1[:,2] - v1[:,11];
    dz123 = v1[:,5] - v1[:,8];
    dz124 = v1[:,5] - v1[:,11];
    dz134 = v1[:,8] - v1[:,11];

    dx212 = v2[:,0] - v2[:,3];
    dx213 = v2[:,0] - v2[:,6];
    dx214 = v2[:,0] - v2[:,9];
    dx223 = v2[:,3] - v2[:,6];
    dx224 = v2[:,3] - v2[:,9];
    dx234 = v2[:,6] - v2[:,9];
    dy212 = v2[:,1] - v2[:,4];
    dy213 = v2[:,1] - v2[:,7];
    dy214 = v2[:,1] - v2[:,10];
    dy223 = v2[:,4] - v2[:,7];
    dy224 = v2[:,4] - v2[:,10];
    dy234 = v2[:,7] - v2[:,10];
    dz212 = v2[:,2] - v2[:,5];
    dz213 = v2[:,2] - v2[:,8];
    dz214 = v2[:,2] - v2[:,11];
    dz223 = v2[:,5] - v2[:,8];
    dz224 = v2[:,5] - v2[:,11];
    dz234 = v2[:,8] - v2[:,11];

    dx312 = v3[:,0] - v3[:,3];
    dx313 = v3[:,0] - v3[:,6];
    dx314 = v3[:,0] - v3[:,9];
    dx323 = v3[:,3] - v3[:,6];
    dx324 = v3[:,3] - v3[:,9];
    dx334 = v3[:,6] - v3[:,9];
    dy312 = v3[:,1] - v3[:,4];
    dy313 = v3[:,1] - v3[:,7];
    dy314 = v3[:,1] - v3[:,10];
    dy323 = v3[:,4] - v3[:,7];
    dy324 = v3[:,4] - v3[:,10];
    dy334 = v3[:,7] - v3[:,10];
    dz312 = v3[:,2] - v3[:,5];
    dz313 = v3[:,2] - v3[:,8];
    dz314 = v3[:,2] - v3[:,11];
    dz323 = v3[:,5] - v3[:,8];
    dz324 = v3[:,5] - v3[:,11];
    dz334 = v3[:,8] - v3[:,11];

    dx412 = v4[:,0] - v4[:,3];
    dx413 = v4[:,0] - v4[:,6];
    dx414 = v4[:,0] - v4[:,9];
    dx423 = v4[:,3] - v4[:,6];
    dx424 = v4[:,3] - v4[:,9];
    dx434 = v4[:,6] - v4[:,9];
    dy412 = v4[:,1] - v4[:,4];
    dy413 = v4[:,1] - v4[:,7];
    dy414 = v4[:,1] - v4[:,10];
    dy423 = v4[:,4] - v4[:,7];
    dy424 = v4[:,4] - v4[:,10];
    dy434 = v4[:,7] - v4[:,10];
    dz412 = v4[:,2] - v4[:,5];
    dz413 = v4[:,2] - v4[:,8];
    dz414 = v4[:,2] - v4[:,11];
    dz423 = v4[:,5] - v4[:,8];
    dz424 = v4[:,5] - v4[:,11];
    dz434 = v4[:,8] - v4[:,11];

    L[:,0,0] =        dx112 * dx112 + dy112 * dy112 + dz112 * dz112;      #b1*b1
    L[:,0,1] = 2.0 *  (dx112 * dx212 + dy112 * dy212 + dz112 * dz212);    #b1*b2
    L[:,0,2] =        dx212 * dx212 + dy212 * dy212 + dz212 * dz212;      #b2*b2
    L[:,0,3] = 2.0 *  (dx112 * dx312 + dy112 * dy312 + dz112 * dz312);    #b1*b3
    L[:,0,4] = 2.0 *  (dx212 * dx312 + dy212 * dy312 + dz212 * dz312);    #b2*b3
    L[:,0,5] =        dx312 * dx312 + dy312 * dy312 + dz312 * dz312;      #b3*b3
    L[:,0,6] = 2.0 *  (dx112 * dx412 + dy112 * dy412 + dz112 * dz412);    #b1*b4
    L[:,0,7] = 2.0 *  (dx212 * dx412 + dy212 * dy412 + dz212 * dz412);    #b2*b4
    L[:,0,8] = 2.0 *  (dx312 * dx412 + dy312 * dy412 + dz312 * dz412);    #b3*b4
    L[:,0,9] =       dx412 * dx412 + dy412 * dy412 + dz412 * dz412;      #b4*b4

    L[:,1,0] =        dx113 * dx113 + dy113 * dy113 + dz113 * dz113;
    L[:,1,1] = 2.0 *  (dx113 * dx213 + dy113 * dy213 + dz113 * dz213);
    L[:,1,2] =        dx213 * dx213 + dy213 * dy213 + dz213 * dz213;
    L[:,1,3] = 2.0 *  (dx113 * dx313 + dy113 * dy313 + dz113 * dz313);
    L[:,1,4] = 2.0 *  (dx213 * dx313 + dy213 * dy313 + dz213 * dz313);
    L[:,1,5] =        dx313 * dx313 + dy313 * dy313 + dz313 * dz313;
    L[:,1,6] = 2.0 *  (dx113 * dx413 + dy113 * dy413 + dz113 * dz413);
    L[:,1,7] = 2.0 *  (dx213 * dx413 + dy213 * dy413 + dz213 * dz413);
    L[:,1,8] = 2.0 *  (dx313 * dx413 + dy313 * dy413 + dz313 * dz413);
    L[:,1,9] =       dx413 * dx413 + dy413 * dy413 + dz413 * dz413;

    L[:,2,0] =        dx114 * dx114 + dy114 * dy114 + dz114 * dz114;
    L[:,2,1] = 2.0 *  (dx114 * dx214 + dy114 * dy214 + dz114 * dz214);
    L[:,2,2] =        dx214 * dx214 + dy214 * dy214 + dz214 * dz214;
    L[:,2,3] = 2.0 *  (dx114 * dx314 + dy114 * dy314 + dz114 * dz314);
    L[:,2,4] = 2.0 *  (dx214 * dx314 + dy214 * dy314 + dz214 * dz314);
    L[:,2,5] =        dx314 * dx314 + dy314 * dy314 + dz314 * dz314;
    L[:,2,6] = 2.0 *  (dx114 * dx414 + dy114 * dy414 + dz114 * dz414);
    L[:,2,7] = 2.0 *  (dx214 * dx414 + dy214 * dy414 + dz214 * dz414);
    L[:,2,8] = 2.0 *  (dx314 * dx414 + dy314 * dy414 + dz314 * dz414);
    L[:,2,9] =       dx414 * dx414 + dy414 * dy414 + dz414 * dz414;

    L[:,3,0] =        dx123 * dx123 + dy123 * dy123 + dz123 * dz123;
    L[:,3,1] = 2.0 *  (dx123 * dx223 + dy123 * dy223 + dz123 * dz223);
    L[:,3,2] =        dx223 * dx223 + dy223 * dy223 + dz223 * dz223;
    L[:,3,3] = 2.0 *  (dx123 * dx323 + dy123 * dy323 + dz123 * dz323);
    L[:,3,4] = 2.0 *  (dx223 * dx323 + dy223 * dy323 + dz223 * dz323);
    L[:,3,5] =        dx323 * dx323 + dy323 * dy323 + dz323 * dz323;
    L[:,3,6] = 2.0 *  (dx123 * dx423 + dy123 * dy423 + dz123 * dz423);
    L[:,3,7] = 2.0 *  (dx223 * dx423 + dy223 * dy423 + dz223 * dz423);
    L[:,3,8] = 2.0 *  (dx323 * dx423 + dy323 * dy423 + dz323 * dz423);
    L[:,3,9] =       dx423 * dx423 + dy423 * dy423 + dz423 * dz423;

    L[:,4,0] =        dx124 * dx124 + dy124 * dy124 + dz124 * dz124;
    L[:,4,1] = 2.0 *  (dx124 * dx224 + dy124 * dy224 + dz124 * dz224);
    L[:,4,2] =        dx224 * dx224 + dy224 * dy224 + dz224 * dz224;
    L[:,4,3] = 2.0 * ( dx124 * dx324 + dy124 * dy324 + dz124 * dz324);
    L[:,4,4] = 2.0 * (dx224 * dx324 + dy224 * dy324 + dz224 * dz324);
    L[:,4,5] =        dx324 * dx324 + dy324 * dy324 + dz324 * dz324;
    L[:,4,6] = 2.0 * ( dx124 * dx424 + dy124 * dy424 + dz124 * dz424);
    L[:,4,7] = 2.0 * ( dx224 * dx424 + dy224 * dy424 + dz224 * dz424);
    L[:,4,8] = 2.0 * ( dx324 * dx424 + dy324 * dy424 + dz324 * dz424);
    L[:,4,9] =       dx424 * dx424 + dy424 * dy424 + dz424 * dz424;

    L[:,5,0] =        dx134 * dx134 + dy134 * dy134 + dz134 * dz134;
    L[:,5,1] = 2.0 * ( dx134 * dx234 + dy134 * dy234 + dz134 * dz234);
    L[:,5,2] =        dx234 * dx234 + dy234 * dy234 + dz234 * dz234;
    L[:,5,3] = 2.0 * ( dx134 * dx334 + dy134 * dy334 + dz134 * dz334);
    L[:,5,4] = 2.0 * ( dx234 * dx334 + dy234 * dy334 + dz234 * dz334);
    L[:,5,5] =        dx334 * dx334 + dy334 * dy334 + dz334 * dz334;
    L[:,5,6] = 2.0 *  (dx134 * dx434 + dy134 * dy434 + dz134 * dz434);
    L[:,5,7] = 2.0 *  (dx234 * dx434 + dy234 * dy434 + dz234 * dz434);
    L[:,5,8] = 2.0 *  (dx334 * dx434 + dy334 * dy434 + dz334 * dz434);
    L[:,5,9] =       dx434 * dx434 + dy434 * dy434 + dz434 * dz434;

    return L

# so we don't use cw here since we use the same control points in the world system
# but if you don't do this, you must change rho accordingly using cw
#
#INPUT:
# cw            (4x4)
#OUTPUT:
# rho           (6)
def compute_rho(cw):

    rho = torch.zeros(6).to(cw.device);
    rho[0] = 2
    rho[1] = 2
    rho[2] = 1
    rho[3] = 2
    rho[4] = 1
    rho[5] = 1
    return rho

def computeJacobian(current_betas,rho,L):

    device = current_betas.device
    M = current_betas.shape[0]
    A = torch.zeros((M,6,4)).to(device)
    b = torch.zeros((M,6)).to(device)
    B = torch.zeros((M,10)).to(device)

    cb = current_betas

    B[:,0] = cb[:,0] * cb[:,0]
    B[:,1] = cb[:,0] * cb[:,1]
    B[:,2] = cb[:,1] * cb[:,1]
    B[:,3] = cb[:,0] * cb[:,2]
    B[:,4] = cb[:,1] * cb[:,2]
    B[:,5] = cb[:,2] * cb[:,2]
    B[:,6] = cb[:,0] * cb[:,3]
    B[:,7] = cb[:,1] * cb[:,3]
    B[:,8] = cb[:,2] * cb[:,3]
    B[:,9] = cb[:,3] * cb[:,3]

    A[:,0,0]=2*cb[:,0]*L[:,0,0]+cb[:,1]*L[:,0,1]+cb[:,2]*L[:,0,3]+cb[:,3]*L[:,0,6];
    A[:,0,1]=cb[:,0]*L[:,0,1]+2*cb[:,1]*L[:,0,2]+cb[:,2]*L[:,0,4]+cb[:,3]*L[:,0,7];
    A[:,0,2]=cb[:,0]*L[:,0,3]+cb[:,1]*L[:,0,4]+2*cb[:,2]*L[:,0,5]+cb[:,3]*L[:,0,8];
    A[:,0,3]=cb[:,0]*L[:,0,6]+cb[:,1]*L[:,0,7]+cb[:,2]*L[:,0,8]+2*cb[:,3]*L[:,0,9];
    b[:,0] = rho[0]-torch.bmm(L[:,0,:].view((M,1,10)),B.view((M,10,1))).squeeze();

    A[:,1,0]=2*cb[:,0]*L[:,1,0]+cb[:,1]*L[:,1,1]+cb[:,2]*L[:,1,3]+cb[:,3]*L[:,1,6];
    A[:,1,1]=cb[:,0]*L[:,1,1]+2*cb[:,1]*L[:,1,2]+cb[:,2]*L[:,1,4]+cb[:,3]*L[:,1,7];
    A[:,1,2]=cb[:,0]*L[:,1,3]+cb[:,1]*L[:,1,4]+2*cb[:,2]*L[:,1,5]+cb[:,3]*L[:,1,8];
    A[:,1,3]=cb[:,0]*L[:,1,6]+cb[:,1]*L[:,1,7]+cb[:,2]*L[:,1,8]+2*cb[:,3]*L[:,1,9];
    b[:,1] = rho[1]-torch.bmm(L[:,1,:].view((M,1,10)),B.view((M,10,1))).squeeze();

    A[:,2,0]=2*cb[:,0]*L[:,2,0]+cb[:,1]*L[:,2,1]+cb[:,2]*L[:,2,3]+cb[:,3]*L[:,2,6];
    A[:,2,1]=cb[:,0]*L[:,2,1]+2*cb[:,1]*L[:,2,2]+cb[:,2]*L[:,2,4]+cb[:,3]*L[:,2,7];
    A[:,2,2]=cb[:,0]*L[:,2,3]+cb[:,1]*L[:,2,4]+2*cb[:,2]*L[:,2,5]+cb[:,3]*L[:,2,8];
    A[:,2,3]=cb[:,0]*L[:,2,6]+cb[:,1]*L[:,2,7]+cb[:,2]*L[:,2,8]+2*cb[:,3]*L[:,2,9];
    b[:,2] = rho[2]-torch.bmm(L[:,2,:].view((M,1,10)),B.view((M,10,1))).squeeze();

    A[:,3,0]=2*cb[:,0]*L[:,3,0]+cb[:,1]*L[:,3,1]+cb[:,2]*L[:,3,3]+cb[:,3]*L[:,3,6];
    A[:,3,1]=cb[:,0]*L[:,3,1]+2*cb[:,1]*L[:,3,2]+cb[:,2]*L[:,3,4]+cb[:,3]*L[:,3,7];
    A[:,3,2]=cb[:,0]*L[:,3,3]+cb[:,1]*L[:,3,4]+2*cb[:,2]*L[:,3,5]+cb[:,3]*L[:,3,8];
    A[:,3,3]=cb[:,0]*L[:,3,6]+cb[:,1]*L[:,3,7]+cb[:,2]*L[:,3,8]+2*cb[:,3]*L[:,3,9];
    b[:,3] = rho[3]-torch.bmm(L[:,3,:].view((M,1,10)),B.view((M,10,1))).squeeze();

    A[:,4,0]=2*cb[:,0]*L[:,4,0]+cb[:,1]*L[:,4,1]+cb[:,2]*L[:,4,3]+cb[:,3]*L[:,4,6];
    A[:,4,1]=cb[:,0]*L[:,4,1]+2*cb[:,1]*L[:,4,2]+cb[:,2]*L[:,4,4]+cb[:,3]*L[:,4,7];
    A[:,4,2]=cb[:,0]*L[:,4,3]+cb[:,1]*L[:,4,4]+2*cb[:,2]*L[:,4,5]+cb[:,3]*L[:,4,8];
    A[:,4,3]=cb[:,0]*L[:,4,6]+cb[:,1]*L[:,4,7]+cb[:,2]*L[:,4,8]+2*cb[:,3]*L[:,4,9];
    b[:,4] = rho[4]-torch.bmm(L[:,4,:].view((M,1,10)),B.view((M,10,1))).squeeze();

    A[:,5,0]=2*cb[:,0]*L[:,5,0]+cb[:,1]*L[:,5,1]+cb[:,2]*L[:,5,3]+cb[:,3]*L[:,5,6];
    A[:,5,1]=cb[:,0]*L[:,5,1]+2*cb[:,1]*L[:,5,2]+cb[:,2]*L[:,5,4]+cb[:,3]*L[:,5,7];
    A[:,5,2]=cb[:,0]*L[:,5,3]+cb[:,1]*L[:,5,4]+2*cb[:,2]*L[:,5,5]+cb[:,3]*L[:,5,8];
    A[:,5,3]=cb[:,0]*L[:,5,6]+cb[:,1]*L[:,5,7]+cb[:,2]*L[:,5,8]+2*cb[:,3]*L[:,5,9];
    b[:,5] = rho[5]-torch.bmm(L[:,5,:].view((M,1,10)),B.view((M,10,1))).squeeze();

    return A,b

def sign_determinant(c):

    c0 = c[:,:,0]
    c1 = c[:,:,1]
    c2 = c[:,:,2]
    c3 = c[:,:,3]

    v1 = c0 - c3
    v2 = c1 - c3
    v3 = c2 - c3
    M = torch.stack((v1,v2,v3),2)
    signs = torch.sign(torch.det(M))

    return signs

def Rx(x):
    batchsize = x.shape[0]
    sinx = torch.sin(x)
    cosx = torch.cos(x)
    Rx = torch.zeros((batchsize,3,3)).cuda()
    Rx[:,0,0] = 1
    Rx[:,0,1] = 0
    Rx[:,0,2] = 0
    Rx[:,1,0] = 0
    Rx[:,1,1] = cosx
    Rx[:,1,2] = -sinx
    Rx[:,2,0] = 0
    Rx[:,2,1] = sinx
    Rx[:,2,2] = cosx
    return Rx

def Ry(x):
    batchsize = x.shape[0]
    sinx = torch.sin(x)
    cosx = torch.cos(x)
    Ry = torch.zeros((batchsize,3,3)).cuda()
    Ry[:,0,0] = cosx
    Ry[:,0,1] = 0
    Ry[:,0,2] = sinx
    Ry[:,1,0] = 0
    Ry[:,1,1] = 1
    Ry[:,1,2] = 0
    Ry[:,2,0] = -sinx
    Ry[:,2,1] = 0
    Ry[:,2,2] = cosx
    return Ry

def Rz(x):
    batchsize = x.shape[0]
    sinx = torch.sin(x)
    cosx = torch.cos(x)
    Rz = torch.zeros((batchsize,3,3)).cuda()
    Rz[:,0,0] = cosx
    Rz[:,0,1] = -sinx
    Rz[:,0,2] = 0
    Rz[:,1,0] = sinx
    Rz[:,1,1] = cosx
    Rz[:,1,2] = 0
    Rz[:,2,0] = 0
    Rz[:,2,1] = 0
    Rz[:,2,2] = 1
    return Rz

def R(thetax,thetay,thetaz):
    rx = Rx(thetax)
    ry = Ry(thetay)
    rz = Rz(thetaz)
    return torch.bmm(rz,torch.bmm(ry,rx))

def quat2euler(quat):

    dist = torch.norm(quat,p=2,dim=0)
    q = quat / dist.unsqueeze(0)
    rx = torch.atan2(2*(q[0]*q[1] + q[2]*q[3]),1-2*(q[1]**2+q[2]**2));
    ry = torch.asin(2*(q[0]*q[2] - q[3]*q[1]));
    rz = torch.atan2(2*(q[0]*q[3] + q[1]*q[2]),1-2*(q[2]**2+q[3]**2));

    return rx, ry,rz

# euler angles in x,y,z
def euler2rotm(rx,ry,rz):
    Rx = np.array([[1,0,0],[0,torch.cos(x),-torch.sin(x)],[0,torch.sin(x),torch.cos(x)]])
    Ry = np.array([[torch.cos(y),0,torch.sin(y)],[0,1,0],[-torch.sin(y),0,torch.cos(y)]])
    Rz = np.array([[torch.cos(z),-torch.sin(z),0],[torch.sin(z),torch.cos(z),0],[0,0,1]])
    return Rx @ Ry @ Rz

# create 3DMM using alphas for shape eigen vectors, and betas for expression eigen vectors
def create3DMM(mu_s, mu_exp, s_eigen, exp_eigen, alphas, betas):
    shape_cov = torch.matmul(s_eigen,alphas)
    exp_cov = torch.matmul(exp_eigen,betas)

    shape = (mu_s + shape_cov.view((53215,3))) + (mu_exp + exp_cov.view((53215,3)))

    return shape/1000

# rotate and translate the shape according to rotation translation and scale factor
def align(shape,s,R,T):
    return s*(torch.matmul(shape,R) + T)

# apply orthographic projection
def project(shape):
    ortho = torch.Tensor([[1,0,0],[0,1,0]]).float().cuda()
    return torch.matmul(shape,ortho.T)

# predict a 3DMM model according to parameters
def predict(s,R,T,alphas,betas):
    shape = create3DMM(alphas,betas)
    shape = align(shape,s,R,T)
    shape = project(shape)
    return shape

# visualize 2d points
def drawpts(img, pts, color=[255,255,255]):

    for p in pts:
        cv2.circle(img,(int(p[0]),int(p[1])),1,color,-1)
    cv2.imshow('img',img)
    cv2.waitKey(0)

    return img

def scatter(pts,color=[255,0,0]):
    plt.scatter(pts[:,0],pts[:,1])
    return

# epnp algorithm to solve for camera pose with gauss newton
#INPUT:
# x_img         (M,2,N)
# x_w           (N,3)
# K             (3,3)
def EPnP(x_img,x_w,K):
    M = x_img.shape[0]
    N = x_img.shape[2]
    f = K[0,0]
    px = K[0,2]
    py = K[1,2]

    # get control points
    c_w = getControlPoints(x_w)

    # solve alphas
    alphas = solveAlphas(x_w,c_w)
    if x_img.is_cuda:
        alphas = alphas.cuda()
    Matrix = setupM(alphas,x_img.permute(0,2,1),px,py,f)

    if ~torch.all(Matrix == Matrix):
        print(Matrix)
        print(f)
        print(x_w)
    #    quit()
    # get last 4 eigenvectors
    u,d,v = torch.svd(Matrix)
    #try:
    #    u,d,v = torch.svd(Matrix)
    #except:
    #    print(Matrix)
    #    u,d,v = torch.svd(Matrix + 1e-4*Matrix.mean().detach()*torch.rand(Matrix.shape))

    km = v[:,:,-4:]

    # sovle N=1
    beta_n1 = torch.zeros((M,4))
    if x_img.is_cuda:
        beta_n1 = beta_n1.cuda()
    beta_n1[:,3] = 1
    c_c_n1 = km[:,:,-1].reshape((M,4,3)).permute(0,2,1)
    _, x_c_n1,s1 = scaleControlPoints(c_c_n1,c_w[:3,:],alphas,x_w)
    mask1 = s1 == s1
    Rn1, Tn1 = getExtrinsics(x_c_n1[mask1],x_w)
    reproj_error2_n1 = getReprojError2(x_img[mask1],x_w,Rn1,Tn1,K,loss='l2')

    return km, c_w, beta_n1 / s1.unsqueeze(1), alphas

    # solve N=2
    d12, d13, d14, d23, d24, d34 = getDistances(c_w)
    distances = torch.stack([d12,d13,d14,d23,d24,d34])**2
    betasq_n2 = getBetaN2(km[:,:,-2:],distances).squeeze()
    b1_n2 = torch.sqrt(torch.abs(betasq_n2[:,0]))
    b2_n2 = torch.sqrt(torch.abs(betasq_n2[:,2])) * torch.sign(betasq_n2[:,1]) * torch.sign(betasq_n2[:,0]).detach()
    beta_n2 = torch.zeros((M,4))
    if x_img.is_cuda:
        beta_n2 = beta_n2.cuda()
    beta_n2[:,2] = b1_n2
    beta_n2[:,3] = b2_n2
    c_c_n2 = getControlPointsN2(km[:,:,-2:],beta_n2)
    _,x_c_n2,s2 = scaleControlPoints(c_c_n2,c_w[:3,:],alphas,x_w)
    mask2 = s2 == s2
    Rn2,Tn2 = getExtrinsics(x_c_n2[mask2],x_w)
    reproj_error2_n2 = getReprojError2(x_img[mask2],x_w,Rn2,Tn2,K,loss='l2')

    if torch.any(mask1 != mask2):
        error1 = torch.zeros(M).to(reproj_error2_n1.device) + 10000
        error2 = torch.zeros(M).to(reproj_error2_n2.device) + 10000
        error1[mask1] = reproj_error2_n1
        error2[mask2] = reproj_error2_n2
    else:
        error1 = reproj_error2_n1
        error2 = reproj_error2_n2


    s = torch.stack((s1,s2))

    # determine best solution in terms of 2d reprojection
    mask = error1 > error2
    mask = mask.long()
    betas = torch.stack((beta_n1,beta_n2))
    best_betas = betas.gather(0,torch.stack(4*[mask],dim=1).unsqueeze(0)).squeeze()
    best_scale = s.gather(0,mask.unsqueeze(0)).squeeze()
    scaled_betas = best_betas / best_scale.unsqueeze(1)

    return km, c_w, scaled_betas, alphas

# Gauss Newton Optimization on extrinsic
def optimizeGN(km,c_w,scaled_betas,alphas,x_w,x_img,K):
    M = km.shape[0]
    beta_opt, err = gauss_newton(km,c_w,scaled_betas)

    # get camera coordinates
    kmsum = beta_opt[:,0].unsqueeze(1)*km[:,:,0] + beta_opt[:,1].unsqueeze(1)*km[:,:,1] + beta_opt[:,2].unsqueeze(1)*km[:,:,2] + beta_opt[:,3].unsqueeze(1)*km[:,:,3]
    c_c = kmsum.reshape((M,4,3)).permute(0,2,1)

    # check sign of the determinent to keep orientation
    sign1 = torch.sign(torch.det(c_w[:3,:3]))
    sign2 = sign_determinant(c_c)

    # get extrinsics
    cc = c_c * (sign1*sign2).unsqueeze(1).unsqueeze(1)
    rep_alpha = torch.stack(M*[alphas])
    xc_opt = torch.bmm(cc,rep_alpha)

    # fix the sign
    xc_opt = xc_opt * torch.sign(xc_opt[:,2,:].sum(1)).unsqueeze(-1).unsqueeze(-1)

    # get the extrinsics
    r_opt,t_opt = getExtrinsics(xc_opt.permute(0,2,1),x_w)

    return xc_opt, r_opt, t_opt, beta_opt

# scatter 3d points
def scatter3d(pts):
    v = pptk.viewer(pts.detach().cpu().numpy())
    v.set(point_size=0.2)
    return v

# 3d shape error
#INPUT:
# s         (N,3)
# s_gt      (N,3)
def getShapeError(s,s_gt):
    error = torch.mean(torch.norm(s - s_gt,dim=1))
    return error

# solve rotation translation

if __name__ == '__main__':

    import dataloader

    #facemodel = dataloader.Face3DMM()
    M = 100;
    N = 68;
    synth = dataloader.TestLoader(400)
    sequence = synth[0]
    x_cam_gt = sequence['x_cam_gt']
    x_w_gt = sequence['x_w_gt']
    f_gt = sequence['f_gt']
    x_img = sequence['x_img']
    data3dmm = dataloader.SyntheticLoader()
    mu_lm = torch.from_numpy(data3dmm.mu_lm).float()
    shape = mu_lm
    shape[:,2] = shape[:,2] * -1;

    one  = torch.ones(M,1,68)
    x_img_one = torch.cat([x_img,one],dim=1)

    #f = torch.relu(out[:,199]).mean()
    error2d = []
    error3d = []
    fvals = []
    f_gt = 400
    for diff in np.linspace(-200,200,20):
        K = torch.zeros((3,3)).float()
        f = f_gt + diff
        fvals.append(f)
        K[0,0] = f;
        K[1,1] = f;
        K[2,2] = 1;
        K[0,2] = 320;
        K[1,2] = 240;
        px = 320;
        py = 240;

        # get control points
        Xc, R, T = EPnP(x_img,shape,K)

        reproj_error2 = getReprojError2(x_img,shape,R,T,K)
        reproj_error3 = getReprojError3(x_cam_gt,shape,R,T)
        rel_error = getRelReprojError3(x_cam_gt,shape,R,T)

        error2d.append(reproj_error2.mean().item())
        error3d.append(rel_error.mean().item())

    data = {}
    data['fvals'] = np.array(fvals)
    data['error2d'] = np.array(error2d)
    data['error3d'] = np.array(error3d)

    print(fvals)
    print(error2d)
    print(error3d)

    import scipy.io
    scipy.io.savemat('exp4.mat',data)
    quit()

    print(torch.mean(reproj_error2))
    print(torch.mean(reproj_error3))
    print(torch.mean(rel_error))
    quit()

    '''
    mu_s = facemodel.mu_shape
    mu_exp = facemodel.mu_exp
    s_eigen = facemodel.shape_eigenvec
    exp_eigen = facemodel.exp_eigenvec
    lm = facemodel.lm

    alphas  = torch.matmul(torch.randn(199),torch.eye(199)).float().cuda()
    betas = torch.matmul(torch.randn(29),torch.eye(29)).float().cuda()

    euler = np.random.rand(3)
    R = torch.Tensor(euler2rotm(euler)).float().cuda()
    T = torch.randn((1,3)).float().cuda() * 10
    s = torch.randn(1).float().cuda()

    shape = create3DMM(mu_s,mu_exp,s_eigen,exp_eigen,alphas,betas)
    shape = align(shape,s,R,T)
    shape = project(shape)

    keypoints = shape[lm,:]
    print(shape.shape)
    print(keypoints.shape)

    pts = keypoints.detach().cpu().numpy()
    print(pts.shape)

    import scipy.io
    scipy.io.savemat('pts.mat',{'pts': pts})
    '''
