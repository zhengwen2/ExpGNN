import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as fn
import pandas as pd
from model import ExpGNN
import argparse
from DataLoader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# 对邻居进行采样
class NeibSampler:
    def __init__(self, graph, nb_size, include_self=False):
        n = graph.number_of_nodes()
        assert 0 <= min(graph.nodes()) and max(graph.nodes()) < n
        # nb_all存储了包含或者不包含自身的所有节点
        # nb只包含了所有的邻居节点
        if include_self:
            nb_all = torch.zeros(n, nb_size + 1, dtype=torch.int64)
            nb_all[:, 0] = torch.arange(0, n)
            nb = nb_all[:, 1:]
        else:
            nb_all = torch.zeros(n, nb_size, dtype=torch.int64)
            nb = nb_all
        popkids = []
        # 如果邻居节点的数量小于人工设置的邻居数量，则用-1来进行填充
        # 如果邻居节点的数量大于人工设置的固定的邻居数量，则进入popkid中
        for v in range(n):
            nb_v = sorted(graph.neighbors(v))
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v)))
                nb[v] = torch.LongTensor(nb_v)
            else:
                popkids.append(v)
        self.include_self = include_self
        self.g, self.nb_all, self.pk = graph, nb_all, popkids

    def to(self, dev):
        self.nb_all = self.nb_all.to(dev)
        return self

    def sample(self):
        # 注意python的对象的复制
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all
        nb_size = nb.size(1)
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)
        # 如果邻居数量过多的话就随机选择指定数量的节点
        for i, v in enumerate(self.pk):
            pk_nb[i] = np.random.choice(sorted(self.g.neighbors(v)), nb_size)
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        return self.nb_all

# 从稀疏矩阵恢复过来
def thsprs_from_spsprs(x):
    x = x.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack((x.row, x.col)).astype(np.int32)).long()
    val = torch.from_numpy(x.data)
    return torch.sparse.FloatTensor(idx, val, torch.Size(x.shape))

# 加载数据集
class Data:
    def __init__(self, datadir, dataname, hyperpm):
        dataset = DataLoader(dataname, datadir)
        use_cuda = torch.cuda.is_available() and not hyperpm.cpu
        dev = torch.device('cuda' if use_cuda else 'cpu')
        # 数据处理
        graph, feat, targ = dataset.get_graph_feat_targ()
        targ = torch.from_numpy(targ).to(dev)
        feat = thsprs_from_spsprs(feat).to(dev)
        # 数据集划分
        trn_idx, val_idx, tst_idx = dataset.get_split()
        trn_idx = torch.from_numpy(trn_idx).to(dev)
        val_idx = torch.from_numpy(val_idx).to(dev)
        tst_idx = torch.from_numpy(tst_idx).to(dev)
        nfeat, nclass = feat.size(1), int(targ.max() + 1)
        neib_sampler = NeibSampler(graph, hyperpm.nbsz).to(dev)
        self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
        self.nfeat, self.nclass = nfeat, nclass
        self.neib_sampler = neib_sampler
        self.graph, self.feat, self.targ = graph, feat, targ



def eval(model, eval_idx, end='\n'):
    model.eval()
    prob = model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./data/')
    parser.add_argument('--dataname', type=str, default='cora')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Insist on using CPU instead of CUDA.')
    parser.add_argument('--nepoch', type=int, default=200,
                        help='Max number of epochs to train.')
    parser.add_argument('--early', type=int, default=8,
                        help='Extra iterations before early-stopping.')
    parser.add_argument('--lr', type=float, default=0.018,
                        help='Initial learning rate.')
    parser.add_argument('--reg', type=float, default=0.0018,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.62,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nlayer', type=int, default=6,
                        help='Number of conv layers.')
    parser.add_argument('--ncaps', type=int, default=7,
                        help='Maximum number of capsules per layer.')
    parser.add_argument('--nhidden', type=int, default=28,
                        help='Number of hidden units per capsule.')
    parser.add_argument('--routit', type=int, default=4,
                        help='Number of iterations when routing.')
    parser.add_argument('--nbsz', type=int, default=30,
                        help='Size of the sampled neighborhood.')

    args = parser.parse_args()

    # 加载数据
    data_info = Data(args.datadir, args.dataname, args)
    # 加载训练好的模型
    model = ExpGNN(data_info.nfeat, data_info.nclass, args)
    model.load_state_dict(torch.load('result/params_14.pkl'))
    # import scipy.io as io
    #
    # matrix= np.array(model.A.detach().numpy())
    # np.savetxt('A.txt', matrix)
    # io.savemat('A.mat', {'A': matrix})
    # exit()

    # neib = torch.load('neib.pt')
    # data_info.neib_sampler.nb_all.copy_(neib)
    # 进行预测
    model.eval()
    prob = model(data_info.feat, data_info.neib_sampler.sample())[data_info.val_idx]
    targ = data_info.targ[data_info.val_idx]
    pred = prob.max(1)[1].type_as(targ)
    acc = pred.eq(targ).double().sum() / len(targ)
    print(acc)
    # acc = []
    # for i in range(data_info.nclass):
    #     targ_i = targ[torch.where(targ == i)]
    #     pred = prob[torch.where(targ == i)]
    #     pred = pred.max(1)[1].type_as(targ_i)
    #     acc_i = pred.eq(targ_i).double().sum() / len(targ_i)
    #     acc.append(acc_i.item())
    # print(acc)

    # 获取到平均的值
    avg_k = model.get_avg_k()
    avg_s = model.get_avg_s()

    # 画相关图
    # dir1 = {}
    # for i in range(7):
    #     dir1[str(i)] = avg_k[:, i].detach().numpy()
    # data = pd.DataFrame(dir1)
    # correlation = data.corr().abs()

    # 画出来相关性图
    # f, ax = plt.subplots(figsize=(7, 7))
    # plt.title('Correlation', y=1, size=26)
    # sns.heatmap(correlation, square=True, vmax=0.8)
    # plt.show()


    # 进行干预操作
    m = []
    # prob1 = model.intervention_exogenous_variables(data_info.feat, data_info.neib_sampler.sample(), [0,1,2,3,4,5,6,7,8,9,10,11], 0)[data_info.val_idx]
    for j in range(args.ncaps):
        m.append(j)
        # 对k进行干预
        prob1 = model.intervention_exogenous_variables(data_info.feat, data_info.neib_sampler.sample(), j, avg_k[:, j])[data_info.val_idx]
        # 对s进行干预
        # prob1 = model.intervention_endogenous_variables(data_info.feat, data_info.neib_sampler.sample(), j, avg_s[:, j])[data_info.val_idx]
        # 对每个类别的准确率分别进行计算
        # 记录下来每个类别的准确率
        acc = []
        change = []
        for i in range(data_info.nclass):
            targ_i = pred[torch.where(pred == i)]
            pred1 = prob1[torch.where(pred == i)]
            pred1 = pred1.max(1)[1].type_as(targ_i)
            acc_i = pred1.eq(targ_i).double().sum() / len(targ_i)
            acc.append(acc_i.item())

            pred2 = prob1.max(1)[1].type_as(pred)
            num = pred2[torch.where(pred2 == i)]
            change.append(abs(len(num)-len(targ_i)))
            # print(pred)
        print(j)
        print(acc)
        print(change)















