import numpy as np
import scipy.sparse
import heapdict


def _check_sparse_edge_weights_matrix(W):
    assert type(W) == scipy.sparse.csr.csr_matrix
    (n, n_other) = W.shape
    assert n == n_other
    assert (W.data >= 0).all()
    #assert (W.transpose() != W).nnz == 0
    return n


def algorithm1(W, mask, k):
    '''
    Input:
        W: n by n scipy.sparse.csr_matrix
            Edge weight matrix. Should be symmetric and positive.
            We use the scipy.sparse.csgraph convention that non-edges are denoted by non-entries.

        mask: boolean array of length n indicating which vertices belong to the terminal set T (aka seed set).

        k: how many nearest neighbors to find for each vertex.

    Output:
        knn: an array of size n such that knn[i] is a list of (up to) k terminals closest to the i-th vertex.
        Each entry of knn[i] is a pair of the form (distance, terminal_vertex_index).
        Note that knn[i] is not sorted.

    '''
    n = _check_sparse_edge_weights_matrix(W)

    assert mask.dtype == np.bool
    assert mask.shape == (n,)
    terminal_indices = mask.nonzero()[0]

    visited = set()
    knn = [[] for i in range(n)]
    heap = heapdict.heapdict()
    for s in terminal_indices:
        heap[(s, s)] = 0.0

    W_indptr = W.indptr
    W_indices = W.indices
    W_data = W.data
    while len(heap) > 0:
        ((seed, i), dist_seed_i) = heap.popitem()
        visited.add((seed, i))

        if len(knn[i]) < k:
            knn[i].append((dist_seed_i, seed))

            for pos in range(W_indptr[i], W_indptr[i+1]):
                j = W_indices[pos]
                if (seed, j) not in visited:
                    alt_dist = dist_seed_i + W_data[pos]
                    if (seed, j) not in heap or alt_dist < heap[(seed, j)]:
                        heap[(seed, j)] = alt_dist

    return knn


def algorithm2(W, mask, k):
    '''
    A variant of algorithm 1 with tighter runtime guarantees.
    '''
    n = _check_sparse_edge_weights_matrix(W)
    assert mask.dtype == np.bool
    assert mask.shape == (n,)

    terminal_indices = mask.nonzero()[0]

    # For each vertex v, Q_[v] is a priority queue of seeds such that a path from seed->v was found for some seed but v was not yet visited from seed.
    # Q is a priority queue containing just the minimum elements from all Q_[v] for vertices that were not yet finalized (=visited from k different seeds).
    
    Q = heapdict.heapdict()
    Q_ = [heapdict.heapdict() for i in range(n)]
    knn = [[] for i in range(n)]
    S = [set() for i in range(n)]
    for seed in terminal_indices:
        Q[seed] = 0.0
        Q_[seed][seed] = 0.0

    W_indptr = W.indptr
    W_indices = W.indices
    W_data = W.data
    while len(Q) > 0:
        (v0, dist) = Q.popitem()
        (seed, dist_copy) = Q_[v0].popitem()
        assert dist == dist_copy
        S[v0].add(seed)
        assert len(knn[v0]) < k
        knn[v0].append((dist, seed))

        if len(knn[v0]) < k and len(Q_[v0]) > 0:
            (_, newdist) = Q_[v0].peekitem()
            Q[v0] = newdist

        # Relax all edges of v
        for pos in range(W_indptr[v0], W_indptr[v0+1]):
            v = W_indices[pos]
            if len(knn[v]) < k and seed not in S[v]:
                alt_dist = dist + W_data[pos]

                if seed not in Q_[v] or Q_[v][seed] > alt_dist:
                    Q_[v][seed] = alt_dist

                if v not in Q or Q[v] > alt_dist:
                    Q[v] = alt_dist

    return knn
