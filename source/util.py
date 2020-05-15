
import math

import torch
import numpy as np
import cv2

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
    if Pw.is_cuda:
        ones = torch.ones((1,68)).cuda()
    else:
        ones = torch.ones((1,68))
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

    # solve R
    u_double,d,v_double = torch.svd(Matrix.double())
    u = u_double.float()
    v = v_double.float()
    r = torch.bmm(u,v.permute(0,2,1))

    rdet = torch.det(r)
    redetmask = (rdet < 0).float() * -2 + 1
    redetmask = redetmask.detach()
    R = r * redetmask.unsqueeze(1).unsqueeze(1)

    # solve T using centers
    rep_w_center = torch.stack(M * [w_center]).unsqueeze(2)
    rot_w_center = torch.bmm(R,rep_w_center)
    T = c_center - rot_w_center.squeeze()

    return R,T

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
def getReprojError2(pimg,pw,R,T,A):

    M = pimg.shape[0]
    N = pimg.shape[1]
    pc = torch.bmm(R,torch.stack(M*[pw]).permute(0,2,1))
    pct = pc + T.unsqueeze(2)
    proj = torch.bmm(torch.stack(M*[A]),pct)
    proj_img = proj / proj[:,-1,:].unsqueeze(1)


    pimg_pred = proj_img[:,:2,:]
    diff = pimg - pimg_pred
    error  = torch.mean(torch.norm(diff,p=2,dim=1),dim=1)

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

    diff = pct - xcam
    d = torch.norm(xcam,p=2,dim=1)
    error = torch.norm(diff,p=2,dim=1)
    return torch.mean(error / d,dim=1)

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

    # not sure which would be more stable solution
    # solve for betas using inverse of MtM
    #MtM = torch.bmm(M.permute(0,2,1),M)
    #b = torch.stack(views*[d])
    #MtM_inv = torch.inverse(MtM)
    #Mtb = torch.bmm(M.permute(0,2,1),b.unsqueeze(-1))
    #beta = torch.bmm(MtM_inv,Mtb)
    #beta = torch.bmm(torch.bmm(torch.inverse(MtM), M.permute(0,2,1)),b.unsqueeze(-1))

    # solve lstsq using svd on batch
    # https://gist.github.com/gngdb/611d8f180ef0f0baddaa539e29a4200e
    U_double,D_double,V_double = torch.svd(M.double())
    U = U_double.float()
    D = D_double.float()
    V = V_double.float()
    b = torch.stack(views*[d])
    Utb = torch.bmm(U.permute(0,2,1),b.unsqueeze(-1))
    D_inv = torch.diag_embed(1.0/D)
    VS = torch.bmm(V,D_inv)
    betas = torch.bmm(VS, Utb)

    return betas

# assumes betas are in order b11,b12,b22
#
#INPUT:
#   v               (Mx12x2)
#   beta            (Mx3x1)
#OUTPUT:
#   c_c             (Mx3x4)
def getControlPointsN2(v,beta):
    M = v.shape[0]
    b1 = torch.sqrt(torch.abs(beta[:,0]))
    b2 = torch.sqrt(torch.abs(beta[:,2])) * torch.sign(beta[:,1]) * torch.sign(beta[:,0])

    p = v[:,:,0]*b1 + v[:,:,1]*b2

    c_c = p.reshape((M,4,3)).permute(0,2,1)
    return c_c

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

# solve rotation translation

if __name__ == '__main__':

    import dataloader

    facemodel = dataloader.Face3DMM()

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
