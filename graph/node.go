package graph

import "leetcode/heap"

type Edge struct {
	To     *Node
	Weight int
}

type Node struct {
	heap.Ordered
	Id          int
	IsVisited   bool
	Cost        int
	Connections []*Edge
}

func (n Node) Less(other heap.Ordered) bool {
	other_node, ok := other.(Node)
	if !ok {
		return false
	}
	return n.Cost < other_node.Cost
}
func (n Node) Equal(other heap.Ordered) bool {
	other_node, ok := other.(Node)
	if !ok {
		return false
	}
	return n.Cost == other_node.Cost
}
func (n Node) Greater(other heap.Ordered) bool {
	other_node, ok := other.(Node)
	if !ok {
		return false
	}
	return n.Cost > other_node.Cost
}
