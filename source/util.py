
import math

import torch
import numpy as np
import cv2


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
