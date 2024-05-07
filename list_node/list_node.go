package list_node

type ListNode struct {
	Val  int
	Next *ListNode
}

func NewList(values []int) *ListNode {
	if (len(values) == 0) {
		return nil
	}
	idx := 0
	head := &ListNode{Val: values[idx]}
	current := head
	for idx < len(values)-1 {
		idx++
		next := &ListNode{Val: values[idx]}
		current.Next = next
		current = next
	}

	return head
}