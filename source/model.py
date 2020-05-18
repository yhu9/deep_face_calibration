import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# model architecture for points (pointnet)
class STN2d(nn.Module):
    def __init__(self):
        super(STN2d,self).__init__()

        self.conv1 = torch.nn.Conv1d(2,64,1)
        self.conv2 = torch.nn.Conv1d(64,128,1)
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

    def forward(self,x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x,2,keepdim=True)[0]
        x = x.view(-1,128)

        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,1]).astype(np.float32))).view(1,4).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x+iden
        x = x.view(-1,2,2)
        return x


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 128)

        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNet(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNet, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        #x = F.relu(self.bn1(self.fc1(x)))
        #x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        #return F.log_softmax(x, dim=1), trans, trans_feat
        return x, trans, trans_feat

    def forward2(self,x):
        x,trans,trans_feat = self.feat(x)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = x.mean(0).unsqueeze(0)

        return x, trans,trans_feat

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size=136, hidden_size=512, num_layers=2, num_classes=200):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        # out = self.fc(out[:, -1, :])

        # Take average of all hidden states across the sequence
        out = self.fc(out.mean(1))

        return out

class CalibrationNet(nn.Module):

    def __init__(self):
        super(CalibrationNet,self).__init__()

        input_size=136
        hidden_size=512
        num_layers=2

        # point net for feature extraction on each view
        self.pointnet = PointNet(k=256, feature_transform=True)

        # lstm for sequence processing
        self.rnn = RNN(input_size=256, hidden_size=256, num_layers=2, num_classes=200)

    def forward(self,x):
        x,_,transfeat  = self.pointnet(x)
        feattransform_loss = feature_transform_regularizer(transfeat) * 0.001
        x = x.unsqueeze(0)

        out = self.rnn(x)
        return out,feattransform_loss

class CalibrationNet2(nn.Module):

    def __init__(self):
        super(CalibrationNet2,self).__init__()

        input_size=136
        hidden_size=512
        num_layers=2

        # point net for feature extraction on each view
        self.pointnet = PointNet(k=512, feature_transform=True)

        # get final output using pointnet as encoder
        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,200)

        self.relu = nn.ReLU()

    def forward(self,x):
        x,_,transfeat  = self.pointnet(x)
        feattransform_loss = feature_transform_regularizer(transfeat) * 0.001
        meanfeat = x.mean(0).unsqueeze(0)
        meanfeat_loss = torch.mean(torch.abs(x-meanfeat))

        out = self.relu(self.fc1(meanfeat))
        out = self.fc2(out)

        return out, meanfeat_loss, feattransform_loss

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

#densenet = models.densenet121(pretrained=True)
#densenet.classifier = nn.Linear(1024,234)
#adjustmentnet= models.densenet121(pretrained=True)
#adjustmentnet.classifier = nn.Linear(1024,136)

if __name__ == '__main__':

    x = torch.randn((1,200))
    l1 = torch.nn.Linear(200,128)
    quit()
    n_layers = 4
    batch_size = 16
    seq_len = 100
    input_dim = 138
    hidden_dim = 200
    input = torch.randn(seq_len,batch_size,input_dim)
    hidden_state = torch.randn(n_layers,batch_size,hidden_dim)
    cell_state = torch.randn(n_layers,batch_size,hidden_dim)
    hidden = (hidden_state,cell_state)

    lstm = nn.LSTM(136,200,n_layers)

    out, states = lstm(input,hidden)
    print(out.shape)
    print(len(hidden))
    print(states[0].shape)
    quit()

    sim_data = Variable(torch.rand(32,3,2500))
    #sim_data = Variable(torch.rand(32,2,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNet(k=5)
    out, _, _ = cls(sim_data)
    print('input size ',sim_data.shape)
    print('class', out.size())
