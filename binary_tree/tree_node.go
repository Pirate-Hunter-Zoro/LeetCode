package binary_tree

import "math"

var NULL = math.MinInt

type TreeNode struct {
	Val		int
	Left	*TreeNode
	Right 	*TreeNode
}

func New(values []int) *TreeNode {
	return new(values, 0, 0)
}

func new(values []int, idx int, child_offset int) *TreeNode {
	if idx >= len(values) || values[idx] == NULL {
		return nil
	} else {
		current := &TreeNode{Val: values[idx]}
		left_idx := 2 * idx + 1 - child_offset
		right_idx := 2 * idx + 2 - child_offset
		additional_offset := 0
		if left_idx < len(values) && values[left_idx] == NULL {
			additional_offset += 2
		}
		current.Left = new(values, left_idx, additional_offset)
		if right_idx < len(values) && values[right_idx] == NULL {
			additional_offset += 2
		}
		current.Right = new(values, right_idx, additional_offset)
		return current
	}
}