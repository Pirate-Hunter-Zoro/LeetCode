package graph

import (
	"leetcode/heap"
	"math"
)

/*
Apply Djikstra's algorithm to find the length of the shortest path from start to end
*/
func Djikstra(nodes []*Node, start int, target int) int {
	nodes[start].Cost = 0
	node_heap := heap.CustomMinHeap[Node]{}
	node_heap.Insert(nodes[start])
	for !node_heap.Empty() {
		for !node_heap.Empty() && node_heap.Peek().IsVisited {
			node_heap.Extract()
		}
		if !node_heap.Empty() {
			next := node_heap.Extract()
			next.IsVisited = true
			for _, edge := range next.Connections {
				if !edge.To.IsVisited {
					edge.To.Cost = min(edge.To.Cost, next.Cost+edge.Weight)
					node_heap.Insert(edge.To)
				}
			}
		}
	}

	if nodes[target].Cost == math.MaxInt {
		return -1
	} else {
		return nodes[target].Cost
	}
}
