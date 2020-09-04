
import os
import random

from skimage import io, transform
import scipy.io
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import numpy as np
from torch.utils.data import Dataset, DataLoader
#import pptk

import util
import torch
# the main data class which has iterators to all datasets
# add more datasets accordingly

def createLoader(db):
    if db == "cad120":
        loader = Cad120Loader()
    elif db == 'cad60':
        loader = Cad60Loader()
    elif db == 'biwi':
        loader = BIWILoader()
    else:
        loader = SyntheticLoader()

    return loader

class Data():
    def __init__(self,db='all'):
        self.dataloader = SyntheticLoader()
        self.M = self.dataloader.M
        self.N = self.dataloader.N
        self.mu_lm = self.dataloader.mu_lm
        self.mu_exp = self.dataloader.mu_exp
        self.lm_eigenvec = self.dataloader.lm_eigenvec
        self.exp_eigenvec = self.dataloader.exp_eigenvec
        self.sigma = self.dataloader.sigma
        self.batchsize = 4
        self.shuffle = True
        self.transform = True
        self.batchloader = DataLoader(self.dataloader,
                batch_size=self.batchsize,
                shuffle=self.shuffle,
                num_workers=4)

    def __len__(self):
        return len(self.dataloader)

    def printinfo(self):
        print(f"RANDOM TRANSFORMS: TRUE")
        print(f"SHUFFLE: TRUE")
        print(f"BATCH SIZE: {self.batchsize}")
        print()

class TestData():
    def __init__(self):
        self.batchsize = 4

    def createLoader(self,f):
        self.dataloader = TestLoader(f)
        self.batchloader = DataLoader(self.dataloader,
                batch_size = self.batchsize,
                shuffle=False,
                num_workers=0)

        return self.batchloader

    def __len__(self):
        return len(self.dataloader)

class TestLoader(Dataset):

    def __init__(self,f,addnoise=True):
        if os.path.isdir("../data"):
            root_dir = os.path.join("../data/synthetic_3dface",f"sequencef{f:04d}")
        else:
            root_dir = os.path.join("../data0/synthetic_3dface",f"sequencef{f:04d}")
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
        self.files.sort()
        self.addnoise = addnoise

    def __len__(self):

        return len(self.files)

    def __getitem__(self,idx):
        fname = self.files[idx]
        data = scipy.io.loadmat(fname)
        self.M = 100
        self.N = 68

        tmp = data['sequence'][0,0]
        x_w = tmp['x_w']
        x_img_gt = tmp['x_img_true']
        x_cam = tmp['x_cam']
        R = tmp['R']
        T = tmp['T']
        f = torch.Tensor(tmp['f'].astype(np.float)[0]).float()
        d = np.mean(T[:,2])

        if self.addnoise:
            le = x_img_gt[:,36,:]
            re = x_img_gt[:,45,:]
            std = np.max(np.linalg.norm(le - re,axis=1)*0.05)
            noise = np.random.rand(100,68,2) * std
        else:
            noise = 0

        x_img_gt[:,:,0] = x_img_gt[:,:,0] - 320
        x_img_gt[:,:,1] = x_img_gt[:,:,1] - 240
        x_img = x_img_gt + noise
        x_img = x_img.reshape((self.M*self.N,2))

        sample = {}
        sample['x_w_gt'] = torch.from_numpy(x_w).float()
        sample['x_cam_gt'] = torch.from_numpy(x_cam).float().permute(0,2,1)
        sample['x_img'] = torch.from_numpy(x_img).float()
        sample['x_img_gt'] = torch.from_numpy(x_img_gt).float().permute(0,2,1)
        sample['f_gt'] = torch.Tensor(f).float()
        sample['d_gt'] = torch.Tensor([d]).float()
        sample['T_gt'] = T

        return sample

class Cad60Loader(Dataset):
    def __init__(self):
        if os.path.isdir("../data"):
            self.root_dir = os.path.join("../data/tmp/cad60/processed")
        else:
            self.root_dir = os.path.join("../data0/tmp/cad60/processed")
        files = os.listdir(self.root_dir)
        files.sort()

        self.all_paths = [os.path.join(self.root_dir,f) for f in files]

    def __len__(self):

        return len(self.all_paths)

    def __getitem__(self,idx):
        path = self.all_paths[idx]
        print(f"load: {path}")
        data = scipy.io.loadmat(path)

        x2d = data['x2d']
        xcam = data['xcam']
        f = data['fgt'][0,0]
        M = x2d.shape[0]
        N = x2d.shape[1]

        pts = x2d.reshape((M*N,2))
        x2d = np.transpose(x2d,(0,2,1))
        xcam = np.transpose(xcam,(0,2,1))

        sample = {}
        sample['x_img_gt'] = torch.from_numpy(x2d).float()
        sample['x_img'] = torch.from_numpy(pts).float()
        sample['x_cam_gt'] = torch.from_numpy(xcam).float()
        sample['f_gt'] = torch.from_numpy(np.array([f]).astype(np.float)).float()

        return sample

class Cad120Loader(Dataset):
    def __init__(self):
        if os.path.isdir("../data"):
            self.root_dir = os.path.join("../data/tmp/cad120/processed")
        else:
            self.root_dir = os.path.join("../data0/tmp/cad120/processed")
        files = os.listdir(self.root_dir)
        files.sort()

        self.all_paths = [os.path.join(self.root_dir,f) for f in files]

    def __len__(self):

        return len(self.all_paths)

    def __getitem__(self,idx):
        path = self.all_paths[idx]
        print(f"load: {path}")
        data = scipy.io.loadmat(path)

        x2d = data['x2d']
        xcam = data['xcam']
        f = data['fgt'][0,0]
        M = x2d.shape[0]
        N = x2d.shape[1]

        pts = x2d.reshape((M*N,2))
        x2d = np.transpose(x2d,(0,2,1))
        xcam = np.transpose(xcam,(0,2,1))

        sample = {}
        sample['x_img_gt'] = torch.from_numpy(x2d).float()
        sample['x_img'] = torch.from_numpy(pts).float()
        sample['x_cam_gt'] = torch.from_numpy(xcam).float()
        sample['f_gt'] = torch.from_numpy(np.array([f]).astype(np.float)).float()

        return sample

class BIWIIDLoader(Dataset):

    def __init__(self):
        if os.path.isdir("../data"):
            self.root_dir = os.path.join("../data/tmp/biwi-i/processed")
        else:
            self.root_dir = os.path.join("../data0/tmp/biwi-i/processed")
        files = os.listdir(self.root_dir)
        files.sort()

        self.all_paths = [os.path.join(self.root_dir,f) for f in files]

    def __len__(self):

        return len(self.all_paths)

    def __getitem__(self,idx):
        path = self.all_paths[idx]
        print(f"load: {path}")
        data = scipy.io.loadmat(path)

        x2d = data['x2d']
        xcam = data['xcam']
        f = data['fgt'][0,0]
        M = x2d.shape[0]
        N = x2d.shape[1]

        pts = x2d.reshape((M*N,2))
        x2d = np.transpose(x2d,(0,2,1))
        xcam = np.transpose(xcam,(0,2,1))

        sample = {}
        sample['x_img_gt'] = torch.from_numpy(x2d).float()
        sample['x_img'] = torch.from_numpy(pts).float()
        sample['x_cam_gt'] = torch.from_numpy(xcam).float()
        sample['f_gt'] = torch.from_numpy(np.array([f]).astype(np.float)).float()

        return sample

class BIWILoader(Dataset):
    def __init__(self):
        if os.path.isdir("../data"):
            self.root_dir = os.path.join("../data/tmp/biwi/processed")
        else:
            self.root_dir = os.path.join("../data0/tmp/biwi/processed")
        files = os.listdir(self.root_dir)
        files.sort()

        self.all_paths = [os.path.join(self.root_dir,f) for f in files]
        return

    def __len__(self):

        return len(self.all_paths)

    def __getitem__(self,idx):
        path = self.all_paths[idx]
        print(f"load: {path}")
        data = scipy.io.loadmat(path)

        x2d = data['x2d']
        xcam = data['xcam']
        f = data['fgt'][0,0]
        M = x2d.shape[0]
        N = x2d.shape[1]

        pts = x2d.reshape((M*N,2))
        x2d = np.transpose(x2d,(0,2,1))
        xcam = np.transpose(xcam,(0,2,1))

        sample = {}
        sample['x_img_gt'] = torch.from_numpy(x2d).float()
        sample['x_img'] = torch.from_numpy(pts).float()
        sample['x_cam_gt'] = torch.from_numpy(xcam).float()
        sample['f_gt'] = torch.from_numpy(np.array([f]).astype(np.float)).float()

        return sample

class AblationLoader(Dataset):

    def __init__(self):
        if os.path.isdir("../data"):
            self.root_dir = os.path.join("../data/tmp/biwi/processed")
            shape_dir = "../data/face_alignment/300W_LP/Code/ModelGeneration/shape_simple.mat"
        else:
            self.root_dir = os.path.join("../data0/tmp/biwi/processed")
            shape_dir = "../data0/face_alignment/300W_LP/Code/ModelGeneration/shape_simple.mat"
        subjects = [f"{sub:02d}" for sub in range(1,25)]

        shape_data = scipy.io.loadmat(shape_dir)
        mu_lm = shape_data['mu_lm']
        mu_exp = shape_data['mu_exp']
        lm_eigenvec = shape_data['lm_eigenvec']
        exp_eigenvec = shape_data['exp_eigenvec']

        self.mu_lm = mu_lm.T
        self.mu_lm = self.mu_lm - np.mean(self.mu_lm,0)
        self.mu_exp = mu_exp.T
        self.mu_exp = self.mu_exp - np.mean(self.mu_exp,0)

        self.lm_eigenvec = lm_eigenvec
        self.exp_eigenvec = exp_eigenvec

        self.maxangle = 20

    def __len__(self):
        return 24

    def __getitem__(self,idx):
        idx = idx+1
        file = f"{idx:02d}_sequence.mat"
        full_path = os.path.join(self.root_dir,file)
        data = scipy.io.loadmat(full_path)

        x2d = data['x2d']
        xcam = data['xcam']
        f = data['fgt'][0,0]
        R = data['R']
        M = x2d.shape[0]
        N = x2d.shape[1]

        # find valid views
        validview = []
        for i in range(M):
            r = Rotation.from_matrix(R[i])
            angles = r.as_euler('zyx',degrees=False)
            angles = np.arcsin(np.sin(angles)) * 180 / np.pi
            if np.any(np.abs(angles) > self.maxangle): continue
            else: validview.append(i)

        x2d = x2d[validview]
        xcam = xcam[validview]
        M = x2d.shape[0]
        N = x2d.shape[1]

        pts = x2d.reshape((M*N,2))
        x2d = np.transpose(x2d,(0,2,1))
        xcam = np.transpose(xcam,(0,2,1))

        sample = {}
        sample['x_img_gt'] = torch.from_numpy(x2d).float()
        sample['x_img'] = torch.from_numpy(pts).float()
        sample['x_cam_gt'] = torch.from_numpy(xcam).float()
        sample['f_gt'] = torch.from_numpy(np.array([f]).astype(np.float)).float()

        return sample

'''
class BIWILoader(Dataset):

    def __init__(self):
        self.root_dir = "../data/kinect_head_pose_db/matdata/"
        shape_dir = "../data/face_alignment/300W_LP/Code/ModelGeneration/shape_simple.mat"
        subjects = [f"{sub:02d}" for sub in range(1,25)]

        shape_data = scipy.io.loadmat(shape_dir)
        mu_lm = shape_data['mu_lm']
        mu_exp = shape_data['mu_exp']
        lm_eigenvec = shape_data['lm_eigenvec']
        exp_eigenvec = shape_data['exp_eigenvec']

        self.mu_lm = mu_lm.T
        self.mu_lm = self.mu_lm - np.mean(self.mu_lm,0)
        self.mu_exp = mu_exp.T
        self.mu_exp = self.mu_exp - np.mean(self.mu_exp,0)

        self.lm_eigenvec = lm_eigenvec
        self.exp_eigenvec = exp_eigenvec

        return

    def __len__(self):
        return 24

    def __getitem__(self,idx):

        if idx == 0: idx = idx+1;
        file = f"sub{idx:02d}.mat"
        full_path = os.path.join(self.root_dir,file)
        data = scipy.io.loadmat(full_path)
        tmp = data['sequence'][0,0]

        # get pose
        R = tmp['R']
        T = tmp['T']
        M = R.shape[0]

        # limit views to poses with estimated 30 degree or less
        validview = []
        for i in range(M):
            r = Rotation.from_matrix(R[i])
            angles = r.as_euler('zyx',degrees=True)
            if np.any(np.abs(angles) > 30): continue
            else: validview.append(i)

        x_w = tmp['x_w']
        x_img_gt = tmp['x_img_gt'][validview]
        self.M = x_img_gt.shape[0]
        self.N = x_img_gt.shape[1]
        x_img = tmp['x_img'][validview].reshape((self.M*self.N,2)) - np.array([[320,240]])
        x_img[:,1] = x_img[:,1]*-1
        x_cam = tmp['x_cam'][validview]
        f = torch.Tensor(tmp['f'].astype(np.float)[0]).float()

        sample = {}
        #sample['x_w_gt'] = torch.from_numpy(x_w).float()
        sample['x_cam_gt'] = torch.from_numpy(x_cam).float().permute(0,2,1)
        sample['x_img'] = torch.from_numpy(x_img).float()
        sample['x_img_gt'] = torch.from_numpy(x_img_gt.astype(np.float)).float().permute(0,2,1)
        sample['f_gt'] = torch.Tensor(f).float()

        return sample
'''

class AnalysisLoader(Dataset):
    def __init__(self,M=100,N=68,f=1000):

        #self.transform = transforms.Compose([ToTensor()])
        if os.path.isdir("../data"):
            root_dir = "../data/face_alignment/300W_LP/Code/ModelGeneration/shape.mat"
        else:
            root_dir = "../data0/face_alignment/300W_LP/Code/ModelGeneration/shape.mat"
        shape_data = scipy.io.loadmat(root_dir)

        # load shape data
        self.mu_s = shape_data['mu_s'].T
        self.mu_exp = shape_data['mu_e'].T
        lm_eigenvec = shape_data['shape_eigenvec']
        exp_eigenvec = shape_data['exp_eigenvec']
        self.sigma = shape_data['sigma']

        nmax = self.mu_s.shape[0]
        idxlist = range(nmax)
        indices = random.sample(idxlist,N)

        lm_eigenvec = lm_eigenvec.reshape(53215,3,199)
        exp_eigenvec = exp_eigenvec.reshape(53215,3,29)
        self.lm_eigenvec = lm_eigenvec[indices].reshape(-1,199)
        self.exp_eigenvec = exp_eigenvec[indices].reshape(-1,29)
        self.mu_s = self.mu_s[indices]
        self.mu_exp = self.mu_exp[indices]

        # bideo sequence length
        self.M = M
        self.N = N

        # extra boundaries on camera coordinates
        self.w = 640
        self.h = 480
        self.f = f
        self.minf = 400; self.maxf = 1500
        self.minz = 400; self.maxz = 3000;
        self.max_rx = 20;
        self.max_ry = 20; self.max_rz = 20;
        self.xstd = 1; self.ystd = 1;

    def __len__(self):
        return 1000

    def __getitem__(self,idx):
        # data holders
        M = self.M
        N = self.N
        x_w = np.zeros((N,3));
        x_cam = np.zeros((M,N,3));
        x_img = np.zeros((M,N,2));
        x_img_true = np.zeros((M,N,2));

        # define intrinsics
        f = self.f;
        K = np.array([[f,0,0],[ 0,f,0], [0,0,1]])

        # create random 3dmm shape
        alpha = np.random.randn(199)*30
        eigenvec = self.lm_eigenvec * self.sigma.T
        s = np.matmul(self.lm_eigenvec,np.expand_dims(alpha,1))
        s = s.reshape(-1,3)
        shape = self.mu_s + s
        x_w = shape

        # visualize 3dmm shape
        #import pptk
        #v = pptk.viewer(shape)
        #v.set(point_size=0.3)
        #quit()

        # create random 3dmm expression
        #beta = np.random.randn(29) * 0.1
        #e = np.sum(np.expand_dims(beta,0)*self.exp_eigenvec,1)
        #exp = e.reshape(68,3)

        # define depth
        minz = self.minz
        maxz = self.maxz

        # get initial and final rotation
        while True:
            r_init, q_init = self.generateRandomRotation()
            r_final, q_final = self.generateRandomRotation()
            t_init = self.generateRandomTranslation(K,minz,maxz)
            t_final = self.generateRandomTranslation(K,minz,maxz)

            ximg_init = self.project2d(r_init,t_init,K,x_w)
            ximg_final = self.project2d(r_final,t_final,K,x_w)
            if np.any(np.amin(ximg_init,axis=0) < -320): continue
            if np.any(np.amin(ximg_final,axis=0) < -320): continue
            if np.any(np.amin(ximg_init,axis=1) < -240): continue
            if np.any(np.amin(ximg_final,axis=1) < -240): continue
            init = np.amax(ximg_init,axis=0)
            final = np.amax(ximg_final,axis=0)
            if init[0] > 320: continue
            if final[0] > 320: continue
            if init[1] > 240: continue
            if final[1] > 240: continue
            break
        d = (t_init[2] + t_final[2]) / 2

        # interpolate quaternion using spherical linear interpolation
        qs = np.stack((q_init,q_final))
        Rs = Rotation.from_quat(qs)
        times = np.linspace(0,1,M)
        slerper = Slerp([0,1],Rs)
        rotations = slerper(times)
        matrices = rotations.as_matrix()

        T = np.stack((np.linspace(t_init[0],t_final[0],M),
                np.linspace(t_init[1],t_final[1],M),
                np.linspace(t_init[2],t_final[2],M))).T

        x_cam = np.matmul(matrices,np.stack(M*[x_w.T])) + np.expand_dims(T,-1)
        proj = np.matmul(np.stack(M*[K]),x_cam)
        proj = proj / np.expand_dims(proj[:,2,:],1)
        proj = proj.transpose(0,2,1)
        x_img_true = proj[:,:,:2]

        le = x_img_true[:,36,:]
        re = x_img_true[:,45,:]
        std = np.linalg.norm(le - re,axis=1)*0.05
        noise = np.random.randn(M,N,2) * std.reshape(M,1,1)

        x_img = x_img_true + noise
        x_img = x_img.reshape((M*N,2))
        #x_img_norm = (x_img - np.array([[320,240]])) / 320

        # create dictionary for results
        sample = {}
        #sample['beta_gt'] = torch.from_numpy(alpha).float()
        sample['x_w_gt'] = torch.from_numpy(x_w).float()
        sample['x_cam_gt'] = torch.from_numpy(x_cam).float()
        sample['x_img'] = torch.from_numpy(x_img).float()
        sample['x_img_gt'] = torch.from_numpy(x_img_true).float().permute(0,2,1)
        sample['f_gt'] = torch.Tensor([f]).float()
        #sample['x_img_norm'] = torch.from_numpy(x_img_norm).float()
        #sample['K_gt'] = torch.from_numpy(K).float()
        #sample['R_gt'] = torch.from_numpy(R).float()
        #sample['Q_gt'] = torch.from_numpy(Q).float()
        #sample['T_gt'] = torch.from_numpy(T).float()
        return sample

    def generateRandomRotation(self):

        ax = self.max_rx;
        ay = self.max_ry;
        az = self.max_rz;
        rx = np.random.random()*2*ax - ax;
        ry = np.random.random()*2*ay - ay;
        rz = np.random.random()*2*az - az;

        r = Rotation.from_euler('zyx',[rz,ry,rx],degrees=True)
        q = r.as_quat()

        return r,q

    def generateRandomTranslation(self,K,minz,maxz,w=640,h=480):

        xvec = np.array([[w],[w/2],[1]])
        yvec = np.array([[h/2],[h],[1]])
        vz = np.array([[0],[0],[1]]);
        vx = np.matmul(np.linalg.inv(K),xvec)
        vy = np.matmul(np.linalg.inv(K),yvec)
        vx = np.squeeze(vx)
        vy = np.squeeze(vy)
        vz = np.array([0,0,1])
        thetax = np.arctan2(np.linalg.norm(np.cross(vz,vy)),np.dot(vz,vy));
        thetay = np.arctan2(np.linalg.norm(np.cross(vz,vx)),np.dot(vz,vx));

        tz = np.random.random()*(maxz-minz) + minz;
        maxx = tz * np.tan(thetax);
        maxy = tz * np.tan(thetay);
        tx = np.random.random()*maxx*2 - maxx;
        ty = np.random.random()*maxy*2 - maxy;
        t = [tx,ty,tz];
        return np.array(t)


    def generateRandomTranslation2(self,K,deltaz,deltax,deltay,w=640,h=480):

        return np.array(t)


    def project2d(self,r,t,K,pw):

        R = r.as_matrix()
        xc = np.matmul(R,pw.T) + np.expand_dims(t,1);

        proj = np.matmul(K,xc)
        proj = proj / proj[2,:]
        ximg = proj.T

        return ximg

class SyntheticLoaderFull(Dataset):
    def __init__(self):

        #self.transform = transforms.Compose([ToTensor()])
        if os.path.isdir("../data"):
            root_dir = "../data/face_alignment/300W_LP/Code/ModelGeneration/shape_simple.mat"
        else:
            root_dir = "../data0/face_alignment/300W_LP/Code/ModelGeneration/shape_simple.mat"

        # load shape data
        shape_data = scipy.io.loadmat(root_dir)
        mu_lm = shape_data['mu_lm']
        mu_exp = shape_data['mu_exp']
        lm_eigenvec = shape_data['lm_eigenvec']
        exp_eigenvec = shape_data['exp_eigenvec']
        self.sigma = shape_data['sigma']

        self.mu_lm = mu_lm.T
        self.mu_lm = self.mu_lm - np.mean(self.mu_lm,0)
        #self.mu_lm = self.mu_lm / np.max(np.abs(self.mu_lm))

        self.mu_exp= mu_exp.T
        self.mu_exp= self.mu_exp - np.mean(self.mu_exp,0)
        #self.mu_exp= self.mu_exp / torch.max(torch.abs(self.mu_exp))

        self.lm_eigenvec = lm_eigenvec
        self.exp_eigenvec = exp_eigenvec

        # video sequence length
        self.M = 100
        self.N = 68

        # extra boundaries on camera coordinates
        self.w = 640
        self.h = 480
        self.minf = 400; self.maxf = 1500
        self.minz = 380; self.maxz = 3200;
        self.max_rx = 20;
        self.max_ry = 20; self.max_rz = 20;
        self.xstd = 1; self.ystd = 1;

    def __len__(self):
        return 10000

    def __getitem__(self,idx):
        # data holders
        M = self.M
        N = self.N
        x_w = np.zeros((68,3));
        x_cam = np.zeros((M,68,3));
        x_img = np.zeros((M,68,2));
        x_img_true = np.zeros((M,68,2));

        # define intrinsics
        f = self.minf + random.random() * (self.maxf - self.minf);
        K = np.array([[f,0,0],[ 0,f,0], [0,0,1]])

        # create random 3dmm shape
        alpha = np.random.randn(199)*5
        s = np.sum( np.expand_dims(alpha,0) * self.lm_eigenvec,1)
        s = s.reshape(68,3)
        lm = self.mu_lm + s
        #lm[:,2] = lm[:,2] * -1
        x_w = lm

        #import pptk
        #v = pptk.viewer(x_w)
        #v.set(point_size=1.1)

        # create random 3dmm expression
        beta = np.random.randn(29) * 0.1
        e = np.sum(np.expand_dims(beta,0)*self.exp_eigenvec,1)
        exp = e.reshape(68,3)

        # define depth
        tz = np.random.random() * (self.maxz-self.minz) + self.minz
        minz = np.maximum(tz - 500,self.minz)
        maxz = np.minimum(tz + 500,self.maxz)

        # get initial and final rotation
        while True:
            r_init, q_init = self.generateRandomRotation()
            r_final, q_final = self.generateRandomRotation()
            t_init = self.generateRandomTranslation(K,minz,maxz)
            t_final = self.generateRandomTranslation(K,minz,maxz)

            ximg_init = self.project2d(r_init,t_init,K,x_w)
            ximg_final = self.project2d(r_final,t_final,K,x_w)
            if np.any(np.amin(ximg_init,axis=0) < -320): continue
            if np.any(np.amin(ximg_final,axis=0) < -320): continue
            if np.any(np.amin(ximg_init,axis=1) < -240): continue
            if np.any(np.amin(ximg_final,axis=1) < -240): continue
            init = np.amax(ximg_init,axis=0)
            final = np.amax(ximg_final,axis=0)
            if init[0] > 320: continue
            if final[0] > 320: continue
            if init[1] > 240: continue
            if final[1] > 240: continue
            break
        d = (t_init[2] + t_final[2]) / 2

        # interpolate quaternion using spherical linear interpolation
        qs = np.stack((q_init,q_final))
        Rs = Rotation.from_quat(qs)
        times = np.linspace(0,1,100)
        slerper = Slerp([0,1],Rs)
        rotations = slerper(times)
        matrices = rotations.as_matrix()

        T = np.stack((np.linspace(t_init[0],t_final[0],M),
                np.linspace(t_init[1],t_final[1],M),
                np.linspace(t_init[2],t_final[2],M))).T

        x_cam = np.matmul(matrices,np.stack(M*[x_w.T])) + np.expand_dims(T,-1)
        proj = np.matmul(np.stack(M*[K]),x_cam)
        proj = proj / np.expand_dims(proj[:,2,:],1)
        proj = proj.transpose(0,2,1)
        x_img_true = proj[:,:,:2]

        le = x_img_true[:,36,:]
        re = x_img_true[:,45,:]
        std = np.mean(np.linalg.norm(le - re,axis=1)*0.05)

        noise = np.random.rand(100,68,2) * std

        x_img = x_img_true + noise
        x_img = x_img.reshape((M*N,2))
        #x_img_norm = (x_img - np.array([[320,240]])) / 320

        # create dictionary for results
        sample = {}
        #sample['beta_gt'] = torch.from_numpy(alpha).float()
        sample['x_w_gt'] = torch.from_numpy(x_w).float()
        sample['x_cam_gt'] = torch.from_numpy(x_cam).float()
        sample['x_img'] = torch.from_numpy(x_img).float()
        sample['x_img_gt'] = torch.from_numpy(x_img_true).float().permute(0,2,1)
        sample['f_gt'] = torch.Tensor([f]).float()
        #sample['x_img_norm'] = torch.from_numpy(x_img_norm).float()
        #sample['K_gt'] = torch.from_numpy(K).float()
        #sample['R_gt'] = torch.from_numpy(R).float()
        #sample['Q_gt'] = torch.from_numpy(Q).float()
        #sample['T_gt'] = torch.from_numpy(T).float()
        return sample

    def generateRandomRotation(self):

        ax = self.max_rx;
        ay = self.max_ry;
        az = self.max_rz;
        rx = np.random.random()*2*ax - ax;
        ry = np.random.random()*2*ay - ay;
        rz = np.random.random()*2*az - az;

        r = Rotation.from_euler('zyx',[rz,ry,rx],degrees=True)
        q = r.as_quat()

        return r,q

    def generateRandomTranslation(self,K,minz,maxz,w=640,h=480):

        xvec = np.array([[w],[w/2],[1]])
        yvec = np.array([[h/2],[h],[1]])
        vz = np.array([[0],[0],[1]]);
        vx = np.matmul(np.linalg.inv(K),xvec)
        vy = np.matmul(np.linalg.inv(K),yvec)
        vx = np.squeeze(vx)
        vy = np.squeeze(vy)
        vz = np.array([0,0,1])
        thetax = np.arctan2(np.linalg.norm(np.cross(vz,vy)),np.dot(vz,vy));
        thetay = np.arctan2(np.linalg.norm(np.cross(vz,vx)),np.dot(vz,vx));

        tz = np.random.random()*(maxz-minz) + minz;
        maxx = tz * np.tan(thetax);
        maxy = tz * np.tan(thetay);
        tx = np.random.random()*maxx*2 - maxx;
        ty = np.random.random()*maxy*2 - maxy;
        t = [tx,ty,tz];

        return np.array(t)

    def project2d(self,r,t,K,pw):

        R = r.as_matrix()
        xc = np.matmul(R,pw.T) + np.expand_dims(t,1);

        proj = np.matmul(K,xc)
        proj = proj / proj[2,:]
        ximg = proj.T

        return ximg

class SyntheticLoader(Dataset):

    def __init__(self):

        #self.transform = transforms.Compose([ToTensor()])
        if os.path.isdir("../data"):
            root_dir = "../data/face_alignment/300W_LP/Code/ModelGeneration/shape_simple.mat"
        else:
            root_dir = "../data0/face_alignment/300W_LP/Code/ModelGeneration/shape_simple.mat"

        # load shape data
        shape_data = scipy.io.loadmat(root_dir)
        mu_lm = shape_data['mu_lm']
        mu_exp = shape_data['mu_exp']
        lm_eigenvec = shape_data['lm_eigenvec']
        exp_eigenvec = shape_data['exp_eigenvec']
        self.sigma = shape_data['sigma']

        self.mu_lm = mu_lm.T
        self.mu_lm = self.mu_lm - np.mean(self.mu_lm,0)
        #self.mu_lm = self.mu_lm / np.max(np.abs(self.mu_lm))

        self.mu_exp= mu_exp.T
        self.mu_exp= self.mu_exp - np.mean(self.mu_exp,0)
        #self.mu_exp= self.mu_exp / torch.max(torch.abs(self.mu_exp))

        self.lm_eigenvec = lm_eigenvec
        self.exp_eigenvec = exp_eigenvec

        # video sequence length
        self.M = 100
        self.N = 68

        # extra boundaries on camera coordinates
        self.w = 640
        self.h = 480
        self.minf = 400; self.maxf = 1500
        self.minz = 380; self.maxz = 3200;
        self.max_rx = 20;
        self.max_ry = 20; self.max_rz = 20;
        self.xstd = 1; self.ystd = 1;

    def __len__(self):
        return 10000

    def __getitem__(self,idx):
        # data holders
        M = self.M
        N = self.N
        x_w = np.zeros((68,3));
        x_cam = np.zeros((M,68,3));
        x_img = np.zeros((M,68,2));
        x_img_true = np.zeros((M,68,2));

        # define intrinsics
        f = self.minf + random.random() * (self.maxf - self.minf);
        K = np.array([[f,0,0],[ 0,f,0], [0,0,1]])

        # create random 3dmm shape
        alpha = np.random.randn(199)*5
        s = np.sum( np.expand_dims(alpha,0) * self.lm_eigenvec,1)
        s = s.reshape(68,3)
        lm = self.mu_lm + s
        #lm[:,2] = lm[:,2] * -1
        x_w = lm

        #import pptk
        #v = pptk.viewer(x_w)
        #v.set(point_size=1.1)

        # create random 3dmm expression
        beta = np.random.randn(29) * 0.1
        e = np.sum(np.expand_dims(beta,0)*self.exp_eigenvec,1)
        exp = e.reshape(68,3)

        # define depth
        tz = np.random.random() * (self.maxz-self.minz) + self.minz
        minz = np.maximum(tz - 500,self.minz)
        maxz = np.minimum(tz + 500,self.maxz)

        # get initial and final rotation
        while True:
            r_init, q_init = self.generateRandomRotation()
            r_final, q_final = self.generateRandomRotation()
            t_init = self.generateRandomTranslation(K,minz,maxz)
            t_final = self.generateRandomTranslation(K,minz,maxz)

            ximg_init = self.project2d(r_init,t_init,K,x_w)
            ximg_final = self.project2d(r_final,t_final,K,x_w)
            if np.any(np.amin(ximg_init,axis=0) < -320): continue
            if np.any(np.amin(ximg_final,axis=0) < -320): continue
            if np.any(np.amin(ximg_init,axis=1) < -240): continue
            if np.any(np.amin(ximg_final,axis=1) < -240): continue
            init = np.amax(ximg_init,axis=0)
            final = np.amax(ximg_final,axis=0)
            if init[0] > 320: continue
            if final[0] > 320: continue
            if init[1] > 240: continue
            if final[1] > 240: continue
            break
        d = (t_init[2] + t_final[2]) / 2

        # interpolate quaternion using spherical linear interpolation
        qs = np.stack((q_init,q_final))
        Rs = Rotation.from_quat(qs)
        times = np.linspace(0,1,100)
        slerper = Slerp([0,1],Rs)
        rotations = slerper(times)
        matrices = rotations.as_matrix()

        T = np.stack((np.linspace(t_init[0],t_final[0],M),
                np.linspace(t_init[1],t_final[1],M),
                np.linspace(t_init[2],t_final[2],M))).T

        x_cam = np.matmul(matrices,np.stack(M*[x_w.T])) + np.expand_dims(T,-1)
        proj = np.matmul(np.stack(M*[K]),x_cam)
        proj = proj / np.expand_dims(proj[:,2,:],1)
        proj = proj.transpose(0,2,1)
        x_img_true = proj[:,:,:2]

        le = x_img_true[:,36,:]
        re = x_img_true[:,45,:]
        std = np.mean(np.linalg.norm(le - re,axis=1)*0.05)

        noise = np.random.rand(100,68,2) * std

        x_img = x_img_true + noise
        x_img = x_img.reshape((M*N,2))
        #x_img_norm = (x_img - np.array([[320,240]])) / 320

        # create dictionary for results
        sample = {}
        #sample['beta_gt'] = torch.from_numpy(alpha).float()
        sample['x_w_gt'] = torch.from_numpy(x_w).float()
        sample['x_cam_gt'] = torch.from_numpy(x_cam).float()
        sample['x_img'] = torch.from_numpy(x_img).float()
        sample['x_img_gt'] = torch.from_numpy(x_img_true).float().permute(0,2,1)
        sample['f_gt'] = torch.Tensor([f]).float()
        #sample['x_img_norm'] = torch.from_numpy(x_img_norm).float()
        #sample['K_gt'] = torch.from_numpy(K).float()
        #sample['R_gt'] = torch.from_numpy(R).float()
        #sample['Q_gt'] = torch.from_numpy(Q).float()
        #sample['T_gt'] = torch.from_numpy(T).float()
        return sample

    def generateRandomRotation(self):

        ax = self.max_rx;
        ay = self.max_ry;
        az = self.max_rz;
        rx = np.random.random()*2*ax - ax;
        ry = np.random.random()*2*ay - ay;
        rz = np.random.random()*2*az - az;

        r = Rotation.from_euler('zyx',[rz,ry,rx],degrees=True)
        q = r.as_quat()

        return r,q

    def generateRandomTranslation(self,K,minz,maxz,w=640,h=480):

        xvec = np.array([[w],[w/2],[1]])
        yvec = np.array([[h/2],[h],[1]])
        vz = np.array([[0],[0],[1]]);
        vx = np.matmul(np.linalg.inv(K),xvec)
        vy = np.matmul(np.linalg.inv(K),yvec)
        vx = np.squeeze(vx)
        vy = np.squeeze(vy)
        vz = np.array([0,0,1])
        thetax = np.arctan2(np.linalg.norm(np.cross(vz,vy)),np.dot(vz,vy));
        thetay = np.arctan2(np.linalg.norm(np.cross(vz,vx)),np.dot(vz,vx));

        tz = np.random.random()*(maxz-minz) + minz;
        maxx = tz * np.tan(thetax);
        maxy = tz * np.tan(thetay);
        tx = np.random.random()*maxx*2 - maxx;
        ty = np.random.random()*maxy*2 - maxy;
        t = [tx,ty,tz];

        return np.array(t)

    def project2d(self,r,t,K,pw):

        R = r.as_matrix()
        xc = np.matmul(R,pw.T) + np.expand_dims(t,1);

        proj = np.matmul(K,xc)
        proj = proj / proj[2,:]
        ximg = proj.T

        return ximg

# LOADER FOR BIWI KINECT DATASET ONLY
class Face3DMM():

    def __init__(self,
            ):

        # load shape data
        shape_data = scipy.io.loadmat(shape_path)
        self.lm = torch.from_numpy(shape_data['keypoints'].astype(np.int32)).long().cuda().squeeze()
        self.mu_shape = torch.from_numpy(mu_shape.reshape(53215,3)).float().cuda()
        self.mu_shape = self.mu_shape - self.mu_shape.mean(0).unsqueeze(0)
        self.mu_shape = self.mu_shape / torch.max(torch.abs(self.mu_shape))
        self.shape_eigenvec = torch.from_numpy(shape_eigenvec).float().cuda()

        # load expression data
        exp_data = scipy.io.loadmat(exp_path)
        mu_exp = exp_data['mu_exp']
        exp_eigenvec = exp_data['w_exp']

        self.mu_exp = torch.from_numpy(mu_exp.reshape(53215,3)).float().cuda()
        self.exp_eigenvec = torch.from_numpy(exp_eigenvec).float().cuda()

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # torch image: C X H X W
        img = transform.resize(image, (new_h, new_w))
        img = img.transpose((2, 0, 1))

        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # swap color axis because
        # numpy image: H x W x C
        return torch.from_numpy(data).float()

#Batch LOader
#BatchLoader = DataLoader(LFWLoader(),batch_size=8,shuffle=True,num_workers=4)

# UNIT TESTING
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    loader = AnalysisLoader()
    sample = loader[1]

    print(sample['x_w_gt'].shape)
    print(sample['x_cam_gt'].shape)
    print(sample['x_img'].shape)
    print(sample['x_img_gt'].shape)
    print(sample['f_gt'].shape)

    quit()
    '''
    loader = BIWILoader()
    sample = loader[1]
    print(sample['x_w_gt'].shape)
    print(sample['x_img'].shape)
    print(sample['x_img_gt'].shape)
    print(sample['f_gt'].shape)
    print(sample['x_cam_gt'].shape)
    print("sample made")
    print(sample.keys())

    #util.scatter(sample['x_img'].numpy())
    import matplotlib.pyplot as plt
    x = sample['x_img'].numpy()
    plt.scatter(x[:68,0],x[:68,1])
    plt.show()
    '''

    '''
    loader = BIWILoader()
    sample = loader[1]
    print(sample['x_img'].shape)
    print(sample['x_img_gt'].shape)
    print(sample['f_gt'].shape)
    print(sample['x_cam_gt'].shape)

    import matplotlib.pyplot as plt
    for i in range(499):
        x = sample['x_img_gt'].numpy()[i].T
        print(x.shape)
        plt.scatter(x[:68,0],x[:68,1])
        plt.show()
    quit()
    '''

    loader = BIWIIDLoader()
    sample = loader[1]
    ximg = sample['x_img_gt']
    M = ximg.shape[0]
    for i in range(M):
        fig = plt.figure()
        ax = Axes3D(fig)
        x = sample['x_cam_gt'].numpy()[i].T
        ax.scatter(x[:,0],x[:,1],x[:,2])
        plt.show()
    print(ximg.shape)
    quit()

    #np.random.seed(0)
    '''
    loader = SyntheticLoader()
    sample = loader[1]
    ximg = sample['x_img']
    print(sample['x_w_gt'].shape)
    print(sample['x_img'].shape)
    print(sample['x_img_gt'].shape)
    print(sample['f_gt'].shape)
    print(sample['x_cam_gt'].shape)
    print("sample made")
    print(sample.keys())
    for i in range(100):
        x = sample['x_img_gt'].numpy()[i].T
        plt.scatter(x[:68,0],x[:68,1])
        plt.show()
    quit()
    '''

    loader = TestLoader(1000)
    sample = loader[1]

    for i in range(100):
        x = sample['x_img_gt'].numpy()[i].T
        plt.scatter(x[:,0],x[:,1])
        plt.show()
    quit()
    print("sample made")
    print(sample.keys())

    quit()
    # test training loader
    loader = SyntheticLoader()

    sample = loader[1]
    print("sample made")
    print(sample.keys())

