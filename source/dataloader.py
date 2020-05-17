
import os
import random

import torch
from skimage import io, transform
import scipy.io
from scipy.spatial.transform import Rotation
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#import pptk

import util
# the main data class which has iterators to all datasets
# add more datasets accordingly

class Data():
    def __init__(self,db='all'):
        self.dataloader = SyntheticLoader()
        self.M = self.dataloader.M
        self.N = self.dataloader.N
        self.mu_lm = self.dataloader.mu_lm
        self.mu_exp = self.dataloader.mu_exp
        self.lm_eigenvec = self.dataloader.lm_eigenvec
        self.exp_eigenvec = self.dataloader.exp_eigenvec
        self.batchsize = 5
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

class TestLoader(Dataset):

    def __init__(self,f):
        root_dir = os.path.join("../data/synthetic_3dface",f"sequencef{f:04d}")
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
        self.files.sort()

    def __len__(self):

        return len(self.files)

    def __getitem__(self,idx):
        fname = self.files[idx]
        data = scipy.io.loadmat(fname)

        tmp = data['sequence'][0,0]
        x_w = tmp['x_w']
        x_img = tmp['x_img']
        x_img_gt = tmp['x_img_true']
        x_cam = tmp['x_cam']
        R = tmp['R']
        T = tmp['T']
        f = torch.Tensor(tmp['f'].astype(np.float)[0]).float()

        sample = {}
        sample['x_w_gt'] = torch.from_numpy(x_w).float()
        sample['x_cam_gt'] = torch.from_numpy(x_cam).float().permute(0,2,1)
        sample['x_img'] = torch.from_numpy(x_img).float().permute(0,2,1)
        sample['x_img_gt'] = torch.from_numpy(x_img_gt).float().permute(0,2,1)
        sample['f_gt'] = torch.Tensor(f).float()

        return sample

class SyntheticLoader(Dataset):

    def __init__(self):

        #self.transform = transforms.Compose([ToTensor()])
        root_dir = "../data/face_alignment/300W_LP/Code/ModelGeneration/shape_simple.mat"

        # load shape data
        shape_data = scipy.io.loadmat(root_dir)
        mu_lm = shape_data['mu_lm']
        mu_exp = shape_data['mu_exp']
        lm_eigenvec = shape_data['lm_eigenvec']
        exp_eigenvec = shape_data['exp_eigenvec']

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
        self.minx = -160; self.maxx = 160;
        self.miny = -120; self.maxy = 120;
        self.minz = 800; self.maxz = 1500;
        self.max_rx = 20;
        self.max_ry = 30; self.max_rz = 20;
        self.xstd = 1; self.ystd = 1;

    def __len__(self):
        return 10000

    def __getitem__(self,idx):

        # data holders
        M = self.M
        x_w = np.zeros((68,3));
        x_cam = np.zeros((M,68,3));
        x_img = np.zeros((M,68,2));
        x_img_true = np.zeros((M,68,2));
        Q = np.zeros((M,4));
        R = np.zeros((M,3,3));
        T = np.zeros((M,3));

        # define intrinsics
        f = 300 + random.random() * 2500;
        K = np.array([[f,0,320],[ 0,f,240], [0,0,1]])

        # create random 3dmm shape
        alpha = np.random.randn(199)
        alphas = np.diag(alpha)
        s = np.sum(np.matmul(self.lm_eigenvec,alphas),1)
        s = s.reshape(68,3)
        lm = self.mu_lm + s
        x_w = lm;

        # create random 3dmm expression
        betas = np.diag(np.random.randn(29) * 0.2)
        e = np.sum(np.matmul(self.exp_eigenvec,betas),1)
        e = e.reshape(68,3)
        exp = e

        # get initial and final rotation
        while True:
            q_init = self.generateRandomRotation()
            q_final = self.generateRandomRotation()
            t_init = self.generateRandomTranslation()
            t_final = self.generateRandomTranslation()
            if np.dot(q_init,q_final) < 0:
                q_final = q_final * -1

            ximg_init = self.project2d(q_init,t_init,K,x_w)
            ximg_final = self.project2d(q_final,t_final,K,x_w)
            if np.any(np.amin(ximg_init,axis=0) < 0): continue
            if np.any(np.amin(ximg_final,axis=0) < 0): continue
            init = np.amax(ximg_init,axis=0)
            final = np.amax(ximg_final,axis=0)
            if init[0] > 640: continue
            if final[0] > 640: continue
            if init[1] > 480: continue
            if final[1] > 480: continue
            break

        quaternion = np.stack((np.linspace(q_init[0],q_final[0],M),
                np.linspace(q_init[1],q_final[1],M),
                np.linspace(q_init[2],q_final[2],M),
                np.linspace(q_init[3],q_final[3],M))).T

        T = np.stack((np.linspace(t_init[0],t_final[0],M),
                np.linspace(t_init[1],t_final[1],M),
                np.linspace(t_init[2],t_final[2],M))).T

        # create each view
        for i in range(M):
            # get rotation
            q = quaternion[i,:];
            q = q / np.linalg.norm(q);
            rx = np.arctan2(2*(q[0]*q[1] + q[2]*q[3]),1-2*(q[1]**2+q[2]**2));
            ry = np.arcsin(2*(q[0]*q[2] - q[3]*q[1]));
            rz = np.arctan2(2*(q[0]*q[3] + q[1]*q[2]),1-2*(q[2]**2+q[3]**2));
            r = Rotation.from_euler('zyx', [rz,ry,rx], degrees=False).as_matrix()
            Q[i,:] = q;
            R[i,:,:] = r;

            # get translation
            t = T[i,:];

            # project onto 3d and 2d using extrinsics/intrinsics
            xc = np.matmul(r,x_w.T) + np.expand_dims(t,1);
            x_cam[i,:,:] = xc.T;
            proj = np.matmul(K, xc);
            proj = proj / proj[2,:];
            x2d = proj.T;
            x2d = x2d[:,:2];
            x_img_true[i,:,:] = x2d;

            # add 2d noise
            xnoise = np.random.randn(68)*self.xstd;
            ynoise = np.random.randn(68)*self.ystd;
            noise = np.stack((xnoise,ynoise))
            x_img[i,:,:] = x2d + noise.T

            # in case you doubt the operations
            #canvas = np.zeros((480,640,3))
            #util.drawpts(canvas,x_img[i,:,:])

        # create dictionary for results
        sample = {}
        sample['beta_gt'] = torch.from_numpy(alpha).float()
        sample['x_w_gt'] = torch.from_numpy(x_w).float()
        sample['x_cam_gt'] = torch.from_numpy(x_cam).float().permute(0,2,1)
        sample['x_img'] = torch.from_numpy(x_img).float().permute(0,2,1)
        sample['x_img_gt'] = torch.from_numpy(x_img_true).float().permute(0,2,1)
        sample['f_gt'] = torch.Tensor([f]).float()
        #sample['K_gt'] = torch.from_numpy(K).float()
        #sample['R_gt'] = torch.from_numpy(R).float()
        #sample['Q_gt'] = torch.from_numpy(Q).float()
        #sample['T_gt'] = torch.from_numpy(T).float()
        return sample

    def generateRandomRotation(self):
        while True:

            q = [np.random.randn(1)[0],np.random.randn(1)[0],np.random.randn(1)[0],np.random.randn(1)[0]]
            q = q/np.linalg.norm(q)
            rx = np.arctan2(2*(q[0]*q[1] + q[2]*q[3]),1-2*(q[1]**2+q[2]**2));
            ry = np.arcsin(2*(q[0]*q[2] - q[3]*q[1]));
            rz = np.arctan2(2*(q[0]*q[3] + q[1]*q[2]),1-2*(q[2]**2+q[3]**2));
            x_degree = rx*180/np.pi;
            y_degree = ry*180/np.pi;
            z_degree = rz*180/np.pi;
            if(abs(x_degree) < self.max_rx and abs(y_degree) < self.max_ry and abs(z_degree) < self.max_rz):
                break

        return q

    def generateRandomTranslation(self):

        xyz = np.random.rand(3)
        xyz[0] = xyz[0] * (self.maxx - self.minx) + self.minx
        xyz[1] = xyz[1] * (self.maxy - self.miny) + self.miny
        xyz[2] = xyz[2] * (self.maxz - self.minz) + self.minz

        return xyz

    def project2d(self,q,t,K,pw):

        q = q/np.linalg.norm(q)
        rx = np.arctan2(2*(q[0]*q[1] + q[2]*q[3]),1-2*(q[1]**2+q[2]**2));
        ry = np.arcsin(2*(q[0]*q[2] - q[3]*q[1]));
        rz = np.arctan2(2*(q[0]*q[3] + q[1]*q[2]),1-2*(q[2]**2+q[3]**2));
        r = Rotation.from_euler('zyx', [rz,ry,rx], degrees=False).as_matrix()
        xc = np.matmul(r,pw.T) + np.expand_dims(t,1);

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

    loader = SyntheticLoader()

    sample = loader[1]
    print("sample made")
    print(sample.keys())

    quit()
    loader = TestLoader(1000)
    sample = loader[1]
    print("sample made")
    print(sample.keys())

    quit()
    # test training loader
    loader = SyntheticLoader()

    sample = loader[1]
    print("sample made")
    print(sample.keys())

