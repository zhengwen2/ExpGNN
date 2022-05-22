import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as fn

# 对稀疏矩阵执行linear操作
class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return torch.mm(x, self.weight) + self.bias

# 实现了一次解耦的表示学习，参考：Disentangled Graph Convolutional Networks
class RoutingLayer(nn.Module):
    def __init__(self, dim, num_caps):
        super(RoutingLayer, self).__init__()
        assert dim % num_caps == 0
        self.d, self.k = dim, num_caps
        # d是维度，k是划分的大小
        self._cache_zero_d = torch.zeros(1, self.d)
        self._cache_zero_k = torch.zeros(1, self.k)

    def forward(self, x, neighbors, max_iter):
        dev = x.device
        if self._cache_zero_d.device != dev:
            self._cache_zero_d = self._cache_zero_d.to(dev)
            self._cache_zero_k = self._cache_zero_k.to(dev)
        # 对每个节点的邻居进行采样，获得相同数量的邻居，对于一些邻居数量没有这么多的节点是怎么处理的？
        n, m = x.size(0), neighbors.size(0) // x.size(0)
        d, k, nhidden = self.d, self.k, self.d // self.k
        # 实施归一化操作，最大的问题可能是x的解耦的特征之间还是通过相同的特征学习倒的，最开始的那一层还是linear
        x = fn.normalize(x.view(n, k, nhidden), dim=2).view(n, d)
        z = torch.cat([x, self._cache_zero_d], dim=0)
        # 每个节点的邻居数量都是20
        z = z[neighbors].view(n, m, k, nhidden)
        u = None
        # 最大迭代次数，这个迭代次数是什么？设置是8
        # z是本身的表示，u是解耦的表示
        for clus_iter in range(max_iter):
            if u is None:
                p = self._cache_zero_k.expand(n * m, k).view(n, m, k)
            else:
                # 求取z和邻居和自己的表示之间的相似度，*是可以进行维度广播的，所以回自己适应到相同的维度
                p = torch.sum(z * u.view(n, 1, k, nhidden), dim = 3)
            p = fn.softmax(p, dim=2)
            # 根据相似度和本身的表示更新表示
            u = torch.sum(z * p.view(n, m, k, 1), dim=1)
            u += x.view(n, k, nhidden)
            if clus_iter < max_iter - 1:
                u = fn.normalize(u ,dim=2)
        # u = fn.softmax(u, dim=1)
        return u.view(n, d)

# 实现了DisGNN模型
class ExpGNN(nn.Module):
    def __init__(self, nfeat, nclass, hyperpm):
        super(ExpGNN, self).__init__()
        ncaps, rep_dim = hyperpm.ncaps, hyperpm.nhidden * hyperpm.ncaps
        # 实现稀疏矩阵的linear
        self.pca = SparseInputLinear(nfeat, rep_dim)
        conv_ls = []
        for i in range(hyperpm.nlayer):
            conv = RoutingLayer(rep_dim, ncaps)
            self.add_module('conv_%d' % i, conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls
        self.mlp = nn.Linear(rep_dim, nclass)
        self.dropout = hyperpm.dropout
        # 表示学习的迭代次数
        self.routit = hyperpm.routit
        # 设置W矩阵
        self.ncaps = ncaps
        self.rep_dim = rep_dim
        self.nhidden = rep_dim // ncaps
        self.weight = nn.Parameter(torch.zeros(self.ncaps, 1))
        self.A = nn.Parameter(torch.randn(self.ncaps, self.ncaps))
        self.avg_k = torch.zeros(self.nhidden, self.ncaps)
        self.avg_s = torch.zeros(self.nhidden, self.ncaps)

    def _dropout(self, x):
        return fn.dropout(x, self.dropout, training=self.training)

    def forward(self, x, nb):
        nb = nb.view(-1)
        # 得到解耦表示
        x = fn.relu(self.pca(x))
        for conv in self.conv_ls:
            x = self._dropout(fn.relu(conv(x, nb, self.routit)))
        # x：[n, d], d是x的特征的维度
        x = x.view(-1, self.ncaps, self.nhidden).permute(0, 2, 1)
        self.avg_k = torch.mean(x, dim=0, keepdim=True)
        self.avg_k = torch.squeeze(self.avg_k)
        self.disPresentation = x
        # 首先进行的初步的运算
        x = torch.mul(x, torch.diag(self.weight))
        # 进行迭代求解，直接用逆，S=(I-A)-1wu
        x = torch.matmul(x, torch.inverse((torch.eye(self.ncaps, self.ncaps) - (self.A - torch.triu(self.A)))))
        self.avg_s = torch.mean(x, dim=0, keepdim=True)
        self.avg_s = torch.squeeze(self.avg_s)
        # 增加连续函数，让内存连续
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.shape[0], self.rep_dim)
        # 最终的线性分类
        x = self.mlp(x)
        return fn.log_softmax(x, dim=1)


    # 对外生变量进行干预
    def intervention_exogenous_variables(self, x, nb, d_k, value=torch.zeros(26)):
        nb = nb.view(-1)
        # 得到解耦表示
        x = fn.relu(self.pca(x))
        for conv in self.conv_ls:
            x = self._dropout(fn.relu(conv(x, nb, self.routit)))
        # 此时x的维度为[n, d], d是x的特征的维度，然后进行
        x = x.view(-1, self.ncaps, self.nhidden).permute(0, 2, 1)
        # x = fn.softmax(x, dim=1)
        # 对外生变量进行干预
        x[:, :, d_k] = value.repeat(x.shape[0], 1)
        x = torch.mul(x, torch.diag(self.weight))
        # 进行迭代求解，直接用逆，S=(I-A)-1wu
        x = torch.matmul(x, torch.inverse((torch.eye(self.ncaps, self.ncaps) - (self.A - torch.triu(self.A)))))
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, self.rep_dim)
        # 最终的线性分类
        x = self.mlp(x)
        return fn.log_softmax(x, dim=1)


    # 对s进行干预
    def intervention_endogenous_variables(self, x, nb, d_s, value):
        nb = nb.view(-1)
        # 得到解耦表示，但是比较奇怪的是这里不是还是全连接网络吗，怎么就解耦了呢？
        x = fn.relu(self.pca(x))
        for conv in self.conv_ls:
            x = self._dropout(fn.relu(conv(x, nb, self.routit)))
        # 此时x的维度为[n, d], d是x的特征的维度，然后进行
        x = x.view(-1, self.ncaps, self.nhidden).permute(0, 2, 1)
        x = torch.mul(x, torch.diag(self.weight))
        fk = x
        # 进行迭代求解，直接用逆，S=(I-A)-1wu
        # x = torch.matmul(x, torch.inverse((torch.eye(self.ncaps, self.ncaps) - (self.A - torch.triu(self.A)))))
        # 对内生变量s进行干预
        for i in range(self.ncaps):
            x[:, :, d_s] = value.repeat(x.shape[0], 1)
            x = torch.matmul(x, (self.A - torch.triu(self.A))) + fk
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, self.rep_dim)
        # 最终的线性分类
        x = self.mlp(x)
        return fn.log_softmax(x, dim=1)

    # 获取平均的k值
    def get_avg_k(self):
        return self.avg_k

    # 获取平均的s值
    def get_avg_s(self):
        return self.avg_s

