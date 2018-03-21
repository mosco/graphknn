[![PyPI version](https://badge.fury.io/py/graphknn.svg)](https://badge.fury.io/py/graphknn)

# Graph KNN Python module

Given an undirected graph and a set of terminal (or seed) vertices T, this python package finds, for every vertex, its K nearest neighbors from the set T.

# Installation

```pip install graphknn```

# Usage

The main functions are **graphknn.algorithm1(W, mask, k)** and **graphknn.algorithm2(W, mask, k)**.
Both algorithms have the same interface with slightly different implementations.
**Input**:
* **W:** n x n matrix of edge weights, of type scipy.sparse.csr_matrix.
* **mask:** boolean array of length n indicating which vertices belong to the terminal set T.
* **k:** how many nearest neighbors to find for each vertex.

**Output:**
* **knn:** this is an array of size n such that knn[i] is a list of up to k pairs of (distance, terminal_vertex_index). Note that knn[i] is not sorted.


Algorithm 1 is simpler whereas Algorithm 2 has tighter runtime guarantees. We have seen cases where algorithm 1 is faster than algorithm 2 and vice versa, so try both on your data and choose the faster one.


## Example

```
import numpy as np
import scipy.sparse
import graphknn

def build_sparse_undirected_nonnegative_csr_matrix(n):
    W = np.random.random((n,n))
    W = W + W.transpose()
    W[W < 1.5] = np.inf
    return scipy.sparse.csr_matrix(W)


def test_graphknn():
    N = 10
    p = 0.5 
    k = 3
    
    W = build_sparse_undirected_nonnegative_csr_matrix(N)
    mask = np.random.random(N) < p

    print('Graph edges:')
    print(W,'\n')

    print('Terminal indices:')
    print(mask.nonzero()[0], '\n')

    result = graphknn.algorithm1(W, mask, k)

    print('K nearest terminal indices of all vertices:')
    for i in range(len(result)):
        print('result[{0}]:\n{1}'.format(i, sorted(result[i])))

test_graphknn()
```

# Details

A simple solution to the problem of finding the k nearest terminal vertices is
to run Dijkstra's algorithm from each of the terminal vertices, forming a |T| by |V| matrix. Then for each vertex i we examine the i-th column of the matrix and pick the k nearest cells (this can be done efficiently using Hoare's selection algorithm). The runtime of this method is O(|T||V|log|V| + |E|).
However, this approach is wasteful, since it spends a lot of time finding irrelevant shortest paths from terminals to vertices that are very far from them.

This module implements a faster approach that can be described as performing |T| Dijkstra runs in parallel combined with an early stopping rule that prevents unnecessary traversals. This stopping rule simply stops exploring vertices once we have found shortest paths from k different terminals.

For more details, including a proof of correctness and runtime bounds, see Section 4 and Appendix B of our paper:

[Amit Moscovich](https://mosco.github.io), [Ariel Jaffe](https://arieljaffe.wixsite.com/homepage), [Boaz Nadler](http://www.weizmann.ac.il/math/Nadler/home)
[**Minimax-optimal semi-supervised regression on unknown manifolds**](https://arxiv.org/abs/1611.02221),
AISTATS (2017).

Please cite our paper if using this code for your research.

