package binary_tree

import "math"

var NULL = math.MinInt

type TreeNode struct {
	Val		int
	Left	*TreeNode
	Right 	*TreeNode
}

func New(values []int) *TreeNode {
	return new(values, 0)
}

func new(values []int, idx int) *TreeNode {
	if idx >= len(values) || values[idx] == NULL {
		return nil
	} else {
		current := &TreeNode{Val: values[idx]}
		left_idx := 2 * idx + 1
		right_idx := 2 * idx + 2
		current.Left = new(values, left_idx)
		current.Right = new(values, right_idx)
		return current
	}
}