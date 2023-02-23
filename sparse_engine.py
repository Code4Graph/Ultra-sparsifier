import numpy as np
from torch import linalg as LA
import torch
import torch.nn.functional as F
import time
from utils import *


def c_lmatrix(tensor_adj):
    identity = torch.eye(tensor_adj.shape[0]).cuda()
    adj_identity = torch.add(tensor_adj, identity)
    row_sum = torch.sum(adj_identity, 1).flatten()
    d_inv = torch.pow(row_sum, -1).flatten()
    d_inv[torch.isinf(d_inv)] = 0.
    d_inv = torch.diag(d_inv)
    adj = torch.matmul(d_inv, adj_identity)
    adj = adj.to_sparse()
    return adj

def mask_grad(grad1, grad2, mask):
    new_grad1 = torch.mul(grad1, mask)
    new_grad2 = torch.mul(grad2, mask)
    return torch.div(new_grad1 + torch.t(new_grad1), 2), torch.div(new_grad2 + torch.t(new_grad2), 2)


def train(optimizer, model, features, l1_times, idx_train, idx_val, epoch, re_w1, re_w2, start_adj1, start_adj2, mask, labels, fastmode, lambda1):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features)
    adj1 = model.get_parameter('gc1.adj')
    adj2 = model.get_parameter('gc2.adj')
    loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + lambda1 * LA.matrix_norm(re_w1 * adj1, ord=1) + lambda1 * LA.matrix_norm(re_w2 * adj2, ord=1)\
                 +torch.norm(c_lmatrix(adj1)-c_lmatrix(start_adj1))+torch.norm(c_lmatrix(adj2)-c_lmatrix(start_adj2))
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    adj1_grad = model.get_parameter('gc1.adj').grad
    adj2_grad = model.get_parameter('gc2.adj').grad
    new_grad1, new_grad2 = mask_grad(adj1_grad, adj2_grad, mask)
    model.get_parameter('gc1.adj').grad.copy_(new_grad1)
    model.get_parameter('gc2.adj').grad.copy_(new_grad2)
    optimizer.step()

    if not fastmode:
        model.eval()
        output = model(features)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('L1_times: {:04d}'.format(l1_times+1),
           'Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(model, features, labels, idx_test):
    model.eval()
    print("In parameter, num of edges in adj1:",
          np.count_nonzero(model.get_parameter('gc1.adj').cpu().data.numpy()) / 2)
    print("In parameter, num of edges in adj2:",
          np.count_nonzero(model.get_parameter('gc2.adj').cpu().data.numpy()) / 2)
    output = model(features)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))