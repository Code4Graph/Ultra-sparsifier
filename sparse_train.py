from __future__ import division
from __future__ import print_function

import time
import argparse
import torch.optim as optim
from models import GCN
from sparse_engine import *
import matplotlib.pyplot as plt


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help='e.g: cora.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=8, help='Number of epochs to train.') #2, 20
parser.add_argument('--l1_times', type=int, default=2, help='L1 times.') #2, 10
parser.add_argument('--lr', type=float, default=0.2, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--adj_update', type=bool, default=True, help='True or False.')
parser.add_argument('--epsilon', type=float, default=1e-4, help='epsilon')#1e-4, smaller
parser.add_argument('--prune_ratio', type=float, default=0.15, help='keep ratio')
# parser.add_argument('--lambda1', type=float, default=1e5, help='lambda1')# larger
parser.add_argument('--lambda1', type=float, default=1e6, help='lambda1')# larger

if __name__ == '__main__':
    args = parser.parse_args()
    epsilon = args.epsilon
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print('cuda', args.cuda)
    print("lambda1: ", args.lambda1)
    print('epsilon: ', epsilon)
    print('prune_ratio: ', args.prune_ratio)
    print('epochs: ', args.epochs)
    print('l1_times: ', args.l1_times)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    adj, features, labels, idx_train, idx_val, idx_test, mask, dense_adj = load_data(args.dataset)
    # num_node = dense_adj.shape[0]
    # Model and optimizer
    model = GCN(adj=adj, nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout, device=args.cuda, adj_update=args.adj_update)

    model.load_state_dict(torch.load(args.dataset + '_model.pt'))
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        mask = mask.cuda()
    adj1 = model.get_parameter('gc1.adj').cpu().data.numpy()
    adj2 = model.get_parameter('gc2.adj').cpu().data.numpy()
    before_adj1 = np.count_nonzero(adj1) / 2
    before_adj2 = np.count_nonzero(adj2) / 2
    print("before L1 project, num of edges in adj1:", np.count_nonzero(adj1) / 2)
    print("before L1 project, num of edges in adj2:", np.count_nonzero(adj2) / 2)
    print('loading model and test')
    test(model, features, labels, idx_test)
    print('----------------------------')

    for name, param in model.named_parameters():
        if 'adj' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        print(name, param.requires_grad)
    start_adj1 = model.get_parameter('gc1.adj')
    start_adj2 = model.get_parameter('gc2.adj')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                           weight_decay=args.weight_decay)
    fastmode = args.fastmode
    lambda1 = args.lambda1
    # Train model
    # init reweighted:
    W1 = initialize(model.get_parameter('gc1.adj')).cuda()  # w = 1
    W2 = initialize(model.get_parameter('gc2.adj')).cuda()  # w = 1
    t_total = time.time()
    for l1_times in range(args.l1_times):
        for epoch in range(args.epochs):
            train(optimizer, model, features, l1_times, idx_train, idx_val, epoch, W1, W2, start_adj1, start_adj2, mask, labels, fastmode, lambda1)
        print("update reweighted")
        adj1 = model.get_parameter('gc1.adj').cpu().data.numpy()
        W1 = torch.FloatTensor(np.divide(1, np.abs(np.add(adj1, epsilon)))).cuda()
        adj2 = model.get_parameter('gc2.adj').cpu().data.numpy()
        W2 = torch.FloatTensor(np.divide(1, np.abs(np.add(adj2, epsilon)))).cuda()

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # finally project and prune to 0
    print("final reweight L1 minization")
    adj1 = model.get_parameter('gc1.adj').cpu().data.numpy()
    adj2 = model.get_parameter('gc2.adj').cpu().data.numpy()
    adj1 = project(adj1, dense_adj, args.prune_ratio)
    adj2 = project(adj2, dense_adj, args.prune_ratio)
    print("L1 project, num of edges in adj1:", np.count_nonzero(adj1) / 2)
    print("L1 project, num of edges in adj2:", np.count_nonzero(adj2) / 2)
    after_adj1 = np.count_nonzero(adj1) / 2
    after_adj2 = np.count_nonzero(adj2) / 2
    with torch.no_grad():
        model.get_parameter('gc1.adj').copy_(torch.FloatTensor(adj1).cuda())
        model.get_parameter('gc2.adj').copy_(torch.FloatTensor(adj2).cuda())
        print('prune ratio: ', (before_adj1 - after_adj1) / before_adj1 * 100)
        print('prune ratio: ', (before_adj2 - after_adj2) / before_adj2 * 100)

    # Testing
    test(model, features, labels, idx_test)

    print('saving adj')
    save_file = args.dataset + 'adj1_p_'+ str(args.prune_ratio)+'.npz'
    save_sparse_csr(save_file, sp.csr_matrix(adj1))
    save_file = args.dataset + 'adj2_p_' + str(args.prune_ratio) + '.npz'
    save_sparse_csr(save_file, sp.csr_matrix(adj2))
    # print('saving weights')
    # np.save('weight1.npy', model.get_parameter('gc1.weight').cpu().data.numpy())
    # np.save('weight2.npy', model.get_parameter('gc2.weight').cpu().data.numpy())
