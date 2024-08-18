package binary_tree

import (
	"leetcode/linked_list"
	"math"
)

var NULL = math.MinInt

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func NewTree(values []int) *TreeNode {
	head := &TreeNode{Val: values[0]}
	node_queue := linked_list.NewQueue[*TreeNode]()
	node_queue.Enqueue(head)
	seen_once := make(map[*TreeNode]bool)
	curr_idx := 1
	for !node_queue.Empty() {
		new_queue := linked_list.NewQueue[*TreeNode]()
		for !node_queue.Empty() {
			if curr_idx >= len(values) {
				break
			}
			parent := node_queue.Peek()
			_, ok := seen_once[parent]
			next_val := values[curr_idx]
			if ok {
				// Right child
				node_queue.Dequeue()
				if next_val != NULL {
					parent.Right = &TreeNode{Val: next_val}
					new_queue.Enqueue(parent.Right)
				}
			} else {
				seen_once[parent] = true
				// Left child
				if next_val != NULL {
					parent.Left = &TreeNode{Val: next_val}
					new_queue.Enqueue(parent.Left)
				}
			}
			curr_idx++
		}
		if curr_idx >= len(values) {
			break
		}
		node_queue = new_queue
	}

	return head
}