'''Use py.test to run the testing functions here, or just run them manually.'''

import numpy as np
import scipy.sparse
import heapdict

import graphknn


def graphknn_using_dijkstra(W, mask, k):
    n = graphknn._check_sparse_edge_weights_matrix(W)

    assert mask.dtype == np.bool
    assert mask.shape == (n,)
    terminal_indices = mask.nonzero()[0]

    assert k <= len(terminal_indices)

    distances_from_terminals = np.vstack(scipy.sparse.csgraph.dijkstra(W, indices = [i])[0] for i in terminal_indices)
    assert distances_from_terminals.shape == (len(terminal_indices), n)

    knn = []
    for i in range(n):
        k_closest_terminals_to_i = np.argpartition(distances_from_terminals[:,i], k-1)[:k]
        knn.append(list(zip(distances_from_terminals[k_closest_terminals_to_i, i], terminal_indices[k_closest_terminals_to_i])))

    return knn


def build_sparse_undirected_nonnegative_csr_matrix(n):
    W = np.random.random((n,n))
    W = W + W.transpose()
    W[W < 1.5] = np.inf
    return scipy.sparse.csr_matrix(W)


def test_graphknn():
    N = 100
    p = 0.2 
    k = 5
    
    W = build_sparse_undirected_nonnegative_csr_matrix(N)
    mask = np.random.random(N) < p
    print('terminal indices:')
    print(mask.nonzero()[0])

    result0 = graphknn_using_dijkstra(W, mask, k)
    result1 = graphknn.algorithm1(W, mask, k)
    result2 = graphknn.algorithm2(W, mask, k)

    for i in range(len(result0)):
        print('result0[{0}]:\n{1}'.format(i, sorted(result0[i])))
        print('result1[{0}]:\n{1}'.format(i, sorted(result1[i])))
        print('result2[{0}]:\n{1}'.format(i, sorted(result2[i])))

        assert sorted(result0[i]) == sorted(result1[i])
        assert sorted(result0[i]) == sorted(result2[i])

