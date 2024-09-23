package tree

import "math"

/*
You are given a tree with n nodes numbered from 0 to n - 1 in the form of a parent array parent where parent[i] is the parent of ith node.
The root of the tree is node 0.
Find the kth ancestor of a given node.

The kth ancestor of a tree node is the kth node in the path from that node to the root node.

Implement the TreeAncestor class:
- TreeAncestor(int n, int[] parent) Initializes the object with the number of nodes in the tree and the parent array.
- int getKthAncestor(int node, int k) return the kth ancestor of the given node node. If there is no such ancestor, return -1.

Link:
https://leetcode.com/problems/kth-ancestor-of-a-tree-node/description/?envType=problem-list-v2&envId=tree

Inspiration:
The LeetCode hints and the following video:
https://www.youtube.com/watch?v=c5O7E_PDO4U&t=247s
*/
type TreeAncestor struct {
	// For a given node, what is its kth parent?
    sols map[int]map[int]int
}


func Constructor(n int, parent []int) TreeAncestor {
	to_return := TreeAncestor{sols: make(map[int]map[int]int)}
    for i:=0; i<n; i++ {
		to_return.sols[i] = make(map[int]int)
	}
	for j:=0; j<=int(math.Log2(float64(n))); j++ {
		ancestor := 1 << j
		for i:=0; i<n; i++ {
			if ancestor == 1 {
				to_return.sols[i][j] = parent[i]
			} else {
				// Find the 2^{j-1}th ancestor's 2^{j-1}th ancestor
				intermediate_ancestor := to_return.sols[i][j-1]
				if intermediate_ancestor != -1 {
					to_return.sols[i][j] = to_return.sols[intermediate_ancestor][j-1]
				} else {
					to_return.sols[i][j] = -1
				}
			}
		}
	}

	return to_return
}


func (tree_ancestor *TreeAncestor) GetKthAncestor(node int, k int) int {
	if k >= len(tree_ancestor.sols) {
		return -1
	}
	j := int(math.Log2(float64(k)))
	if int(math.Pow(2, float64(j))) == k {
		// Even power of two - we have solved this problem
		return tree_ancestor.sols[node][j]
	} else {
		// Not an even power of 2
		intermediate_ancestor := tree_ancestor.sols[node][j]
		if intermediate_ancestor == -1 {
			return -1
		}
		return tree_ancestor.GetKthAncestor(intermediate_ancestor, k - (1 << j))
	}
}