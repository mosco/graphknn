
Given an undirected graph and a set of terminal (or seed) vertices T, this python package finds, for every vertex, its K nearest neighbors from the set T.


# Interface

The main functions are graphknn.algorithm1(W, mask, k) and graphknn.algorithm1(W, mask, k).
Both have the same interface but slightly different implementations. Algorithm 1 is simpler whereas algorithm 2 has tighter runtime guarantees.

We have seen cases where algorithm 1 is faster than algorithm 2 and vice versa, so try both on your data and choose the faster one.

**Input**:
* **W:** n x n matrix of edge weights, of type scipy.sparse.csr_matrix.
* **mask:** boolean array of length n indicating which vertices belong to the terminal set T.
* **k:** how many nearest neighbors to find for each vertex.

**Output:**
* **knn:** this is an array of size n such that knn[i] is a list of up to k pairs of (distance, terminal_vertex_index). Note that knn[i] is not sorted.


# Details

A simple solution to the problem of finding the k nearest terminal vertices is
to run Dijkstra's algorithm from each of the terminal vertices, forming a |T| by |V| matrix. Then for each vertex i we examine the i-th column of the matrix and pick the k nearest cells (this can be done efficiently using Hoare's selection algorithm). The runtime of this method is O(|T||V|log|V| + |E|).

However, this approach is wasteful, since it spends a lot of time finding irrelevant shortest paths from terminals to vertices that are very far from them.

This package implements a faster approach that can be described as performing |T| Dijkstra runs in parallel combined with an early stopping rule that prevents unnecessary traversals. This stopping rule simply stops exploring vertices once we have found for them shortest paths from k different terminals.

For more details, see Section 4 and Appendix B of our paper:

Amit Moscovich, Ariel Jaffe, Boaz Nadler, [**Minimax-optimal semi-supervised regression on unknown manifolds**](https://arxiv.org/abs/1611.02221)
Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017)
Please cite this paper if using this code for research.
