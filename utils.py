import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import pickle as pkl
import os
import sys
import copy
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, triu

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data2(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def save_sparse_csr(filename,array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix(
        (loader['data'], loader['indices'], loader['indptr']),
        shape=loader['shape']
    )

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    dense_adj = adj.toarray()


    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    labels_list = []
    for row in range(labels.shape[0]):
        if np.where(labels[row]==1)[0].shape[0] == 0:
            labels_list.append(0)
        else:
            labels_list.append(np.where(labels[row]==1)[0][0])
    labels_arr = np.array(labels_list)
    labels = torch.LongTensor(labels_arr)


    adj = adj.toarray()
    adj = torch.FloatTensor(adj)
    mask = torch.FloatTensor(copy.deepcopy(adj))
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, mask, dense_adj


def initialize(adj):
    res = torch.ones_like(adj)
    return res


def project(ori_adj, adj, prune_ratio):
    input_adj = copy.deepcopy(ori_adj)
    # mask diag
    I = np.eye(input_adj.shape[0])
    I[I==1] = 2
    I[I==0] = 1
    I[I==2] = 0
    input_adj = np.multiply(input_adj, I)
    input_adj = np.multiply(input_adj, adj)
    ###########
    upper_adj = np.triu(adj, 1)
    idx = np.where(upper_adj!=0)
    adj_var = input_adj[idx]
    threshold = np.percentile(adj_var, int(prune_ratio*100), interpolation="lower")

    for i in range(idx[0].shape[0]):
        if input_adj[idx[0][i]][idx[1][i]] < threshold:
            input_adj[idx[0][i]][idx[1][i]] = 0.0
            input_adj[idx[1][i]][idx[0][i]] = 0.0
        else:
            input_adj[idx[0][i]][idx[1][i]] = 1.0
            input_adj[idx[1][i]][idx[0][i]] = 1.0
    # input_adj[input_adj < threshold] = 0.0
    # input_adj[input_adj >= threshold] = 1.0
    return input_adj


#problem here
def project2(ori_adj, adj, prune_ratio):
    num_edge = nx.number_of_edges(nx.from_numpy_array(adj))
    num_prune = int(num_edge * prune_ratio)
    expected = num_edge-num_prune
    input_adj = copy.deepcopy(ori_adj)
    # mask diag
    I = np.eye(input_adj.shape[0])
    I[I==1] = 2
    I[I==0] = 1
    I[I==2] = 0
    input_adj = np.multiply(input_adj, I)
    input_adj = np.multiply(input_adj, adj)
    ###########
    upper_adj = np.triu(adj, 1)
    idx = np.where(upper_adj!=0)
    adj_var = input_adj[idx]
    threshold = np.percentile(adj_var, int(prune_ratio*100), interpolation="lower")


    for i in range(idx[0].shape[0]):
        if input_adj[idx[0][i]][idx[1][i]] < threshold:
            input_adj[idx[0][i]][idx[1][i]] = 0.0
            input_adj[idx[1][i]][idx[0][i]] = 0.0
        else:
            input_adj[idx[0][i]][idx[1][i]] = 1.0
            input_adj[idx[1][i]][idx[0][i]] = 1.0
    # cur_num_edge = nx.number_of_edges(nx.from_numpy_array(input_adj))
    # while cur_num_edge > expected:
    #
    return input_adj


def sync_grad(grad1, grad2, mask):
    grad = torch.mul(grad1, mask) + torch.mul(grad2, mask)
    grad = torch.div(grad, 2)
    # upper = torch.triu(grad)
    # lower = torch.tril(grad)
    # new_grad = torch.div(torch.add(upper, torch.t(lower)),2)
    new_grad = torch.div(grad + torch.t(grad),2)
    return new_grad


def mask_grad(adj_grad, mask):
    grad = torch.mul(adj_grad, mask)
    upper = torch.triu(grad)
    lower = torch.tril(grad)
    new_grad = torch.div(torch.add(upper, torch.t(lower)), 2)
    new_grad = new_grad + torch.t(new_grad)
    return new_grad


def plot_distribution(adj, title):
    dict = {}
    for row in range(adj.shape[0]):
        for col in range(adj.shape[1]):
            if row == col:
                continue
            var = adj[row][col]
            if var in dict.keys():
                dict[var] += 1
            else:
                dict[var] = 1
    for key, value in dict.items():
        dict[key] /= 2
    return dict


def node_connection_distribution(adj):
    G = nx.from_scipy_sparse_matrix(sp.csr_matrix(adj))
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    plt.hist(degree_sequence, bins=20000, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel('Degree')
    plt.ylabel('# of Nodes')
    plt.title("Degree histogram")
    plt.show()


def degree_each_node(adj):
    dict = {}
    G = nx.from_scipy_sparse_matrix(sp.csr_matrix(adj))
    for node in G.nodes():
        d = G.degree[node]
        dict[node] = d
    return dict


def load_data_filtering(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    sparse_adj = adj
    features = normalize(features)
    adj = adj.toarray()
    return adj, features, sparse_adj



def load_data_new(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    if dataset_str == 'nell.0.01':
        # Find relation nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - allx.shape[0], :] = ty
        ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)

        if not os.path.isfile("data/{}.features.npz".format(dataset_str)):
            print("Creating feature vectors for relations - this might take a while...")
            features_extended = sp.hstack((features, sp.lil_matrix((features.shape[0], len(isolated_node_idx)))),
                                          dtype=np.int32).todense()
            features_extended[isolated_node_idx, features.shape[1]:] = np.eye(len(isolated_node_idx))
            features = sp.csr_matrix(features_extended)
            print("Done!")
            save_sparse_csr("data/{}.features".format(dataset_str), features)
        else:
            features = load_sparse_csr("data/{}.features.npz".format(dataset_str))

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    else:
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    dense_adj = adj.toarray()
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    labels_list = []
    for row in range(labels.shape[0]):
        if np.where(labels[row]==1)[0].shape[0] == 0:
            labels_list.append(0)
        else:
            labels_list.append(np.where(labels[row]==1)[0][0])
    labels_arr = np.array(labels_list)
    labels = torch.LongTensor(labels_arr)


    adj = adj.toarray()
    adj = torch.FloatTensor(adj)
    mask = torch.FloatTensor(copy.deepcopy(adj))
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, mask, dense_adj

def randomPrune_sp(adj, seed, percentile):
    tmp = triu(adj, k=1)
    mat_tmp = csr_matrix(tmp)
    # tmp = np.tril(adj, -1)
    # row = np.where(tmp == 1)[0]
    # col = np.where(tmp == 1)[1]
    # num_edge = len(row)
    num_edge = nx.number_of_edges(nx.from_scipy_sparse_matrix(adj))
    pruned_num = int(percentile * num_edge)
    print("pruned num", pruned_num)
    # print("rest of edges:", num_edge - pruned_num)
    row = mat_tmp.nonzero()[0]
    print("len of row before: ", len(row.tolist()))
    col = mat_tmp.nonzero()[1]
    idx = np.arange(len(row.tolist()), dtype=int)
    ######################################
    #using scipy.sparse.csr_matrix.nonzero
    ######################################
    np.random.seed(seed=seed)
    pruned_idx = np.random.choice(idx, size=pruned_num, replace=False)

    mat_tmp[row[pruned_idx], col[pruned_idx]] = 0
    mat_tmp.eliminate_zeros()
    row1 = mat_tmp.nonzero()[0]
    print("len of row after: ", len(row1.tolist()))
    # for i in range(len(pruned_idx)):
    #    adj[row[pruned_idx[i]], col[pruned_idx[i]]] = 0
    #    adj[col[pruned_idx[i]], row[pruned_idx[i]]] = 0

    final_mat = mat_tmp + mat_tmp.T
    return final_mat


def HiToHiCutRoad(ori_adj, pruned_ratio):
    num_edges = np.count_nonzero(ori_adj)
    delete_num_edges = int(num_edges * pruned_ratio)
    degree = ori_adj.sum(1)
    d_dict = dict(enumerate(degree, 0))
    sort_d_dict = sorted(d_dict.items(), key=lambda x:x[1], reverse=True)
    # topk = sort_d_dict[delete_num_edges:]
    count = 0
    for var in sort_d_dict:
        if count == delete_num_edges:
            break
        out_node = var[0]
        for var2 in sort_d_dict: #topk (node, degree)
            in_node = var2[0]
            if out_node == in_node:
                continue
            if ori_adj[out_node][in_node] != 0:
                ori_adj[out_node][in_node] = 0.0
                degree = ori_adj.sum(1)
                d_dict = dict(enumerate(degree, 0))
                sort_d_dict = sorted(d_dict.items(), key=lambda x: x[1], reverse=True)
                break
        count += 1
    return ori_adj


def PageRankScoreCut(ori_adj, keep_ratio):
    num_edges = np.count_nonzero(ori_adj)
    delete_num_edges = int(num_edges * (1-keep_ratio))

    adj = copy.deepcopy(ori_adj)
    graph = nx.from_numpy_matrix(adj)
    scores = nx.pagerank(graph)
    scores_d_dict = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    count = 0
    visited = set()
    while count <= delete_num_edges:
        for var in scores_d_dict:
            if count > delete_num_edges:
                break
            out_node = var[0]
            col_idx = np.where(adj[out_node]!= 0)[0]
            max_score = 0
            neigbor = None
            for i in range(col_idx.shape[0]):
                in_node = col_idx[i]
                if tuple([out_node, in_node]) in visited or out_node == in_node:
                    continue
                else:
                    if scores[in_node] > max_score:
                        neigbor = in_node
                        max_score = scores[in_node]
            if neigbor is not None and adj[out_node][neigbor] != 0:
                adj[out_node][neigbor] = 0.0
                visited.add(tuple([out_node, neigbor]))
                count+=1
    return adj

