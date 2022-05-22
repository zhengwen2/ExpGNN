#!/usr/bin/env python3
import argparse
import torch
import sys
import gc
import random
import time
import tempfile
import numpy as np
from DataLoader import DataLoader
from model import ExpGNN
import torch.optim as optim
import torch.nn.functional as fn

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

# 数据流的处理
class RedirectStdStreams:
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


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


# 设置随机种子
def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 反卷机的方法将是固定的
    torch.backends.cudnn.deterministic = True


# 从稀疏矩阵恢复过来
def thsprs_from_spsprs(x):
    x = x.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack((x.row, x.col)).astype(np.int32)).long()
    val = torch.from_numpy(x.data)
    return torch.sparse.FloatTensor(idx, val, torch.Size(x.shape))

# 设置矩阵为上三角矩阵的loss计算
def matrix_poly(matrix, d):
    x = torch.eye(d).to(device)+ torch.div(matrix.to(device), d).to(device)
    return torch.matrix_power(x, d)

def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

# 获取数据与模型创建
class EvalHelper:
    def __init__(self, dataset, hyperpm):
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
        # DisGNN模型
        model = ExpGNN(nfeat, nclass, hyperpm).to(dev)
        optmz = optim.Adam(model.parameters(),
                           lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.graph, self.feat, self.targ = graph, feat, targ
        self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
        self.model, self.optmz = model, optmz
        # 对邻居节点进行采样，对大于m个邻居节点的节点进行采样处理
        self.neib_sampler = NeibSampler(graph, hyperpm.nbsz).to(dev)
        # 得到控制解耦表示中相关的超参数
        # self.beta = hyperpm.beta

    def run_epoch(self, end='\n'):
        self.model.train()
        self.optmz.zero_grad()
        pro = self.model(self.feat, self.neib_sampler.sample())
        loss = fn.nll_loss(pro[self.trn_idx], self.targ[self.trn_idx])
        # loss += self.beta * (torch.matmul(self.model.disPresentation.T(), self.model.disPresentation) - torch.eye(self.model.disPresentation.))
        # h_a = _h_A(self.model.weight, self.model.weight.size()[0])
        # loss += 300 * h_a + 50 * h_a * h_a
        # print(h_a.item())
        loss.backward()
        self.optmz.step()
        print('trn-loss: %.4f' % loss.item(), end=end)
        return loss.item()

    def print_trn_acc(self):
        print('trn-', end='')
        trn_acc = self._print_acc(self.trn_idx, end=' val-')
        val_acc = self._print_acc(self.val_idx)
        return trn_acc, val_acc

    def print_tst_acc(self):
        print('tst-', end='')
        tst_acc = self._print_acc(self.tst_idx)
        return tst_acc

    def _print_acc(self, eval_idx, end='\n'):
        self.model.eval()
        prob = self.model(self.feat, self.neib_sampler.nb_all)[eval_idx]
        targ = self.targ[eval_idx]
        pred = prob.max(1)[1].type_as(targ)
        acc = pred.eq(targ).double().sum() / len(targ)
        acc = acc.item()
        print('acc: %.4f' % acc, end=end)
        return acc


# 模型运行控制
def train_and_eval(datadir, dataname, hyperpm):
    set_rng_seed(23)
    # 创建模型
    agent = EvalHelper(DataLoader(dataname, datadir), hyperpm)
    tm = time.time()
    best_val_acc, wait_cnt = 0.0, 0
    model_sav = tempfile.TemporaryFile()
    neib_sev = torch.zeros_like(agent.neib_sampler.nb_all, device='cpu')
    for t in range(hyperpm.nepoch):
        print('%3d/%d' % (t, hyperpm.nepoch), end=' ')
        agent.run_epoch(end=' ')
        _, cur_val_acc = agent.print_trn_acc()
        if cur_val_acc > best_val_acc:
            wait_cnt = 0
            best_val_acc = cur_val_acc
            model_sav.close()
            model_sav = tempfile.TemporaryFile()
            torch.save(agent.model.state_dict(), model_sav)
            neib_sev.copy_(agent.neib_sampler.nb_all)
        else:
            # 属于一种提前结束训练的trick：如果发现更新的轮数超过一定的数量但是没有达到更好的效果就停止训练
            wait_cnt += 1
            if wait_cnt > hyperpm.early:
                break
    # print(agent.model.A)
    print("time: %.4f sec." % (time.time() - tm))
    model_sav.seek(0)
    agent.model.load_state_dict(torch.load(model_sav))
    agent.neib_sampler.nb_all.copy_(neib_sev)
    torch.save(agent.model.state_dict(), 'result/params_14.pkl')
    # torch.save(neib_sev, 'neib.pt')
    return best_val_acc, agent.print_tst_acc()


def main(args_str=None):
    assert float(torch.__version__[:3]) + 1e-3 >= 0.4
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./data/')
    parser.add_argument('--dataname', type=str, default='cora')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Insist on using CPU instead of CUDA.')
    parser.add_argument('--nepoch', type=int, default=200,
                        help='Max number of epochs to train.')
    parser.add_argument('--early', type=int, default=10,
                        help='Extra iterations before early-stopping.')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='Initial learning rate.')
    parser.add_argument('--reg', type=float, default=0.0036,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.35,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nlayer', type=int, default=5,
                        help='Number of conv layers.')
    parser.add_argument('--ncaps', type=int, default=2,
                        help='Maximum number of capsules per layer.')
    parser.add_argument('--nhidden', type=int, default=16,
                        help='Number of hidden units per capsule.')
    parser.add_argument('--routit', type=int, default=6,
                        help='Number of iterations when routing.')
    parser.add_argument('--nbsz', type=int, default=30,
                        help='Size of the sampled neighborhood.')
    if args_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_str.split())
    with RedirectStdStreams(stdout=sys.stderr):
        val_acc, tst_acc = train_and_eval(args.datadir, args.dataname, args)
        print('val=%.2f%% tst=%.2f%%' % (val_acc * 100, tst_acc * 100))
    return val_acc, tst_acc

if __name__ == '__main__':
    print('(%.4f, %.4f)' % main())
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()