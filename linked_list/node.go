package linked_list

type Node[T any] struct {
	Val  T
	Next *Node[T]
}

func NewList[T any](values []T) *Node[T] {
	if len(values) == 0 {
		return nil
	}
	idx := 0
	head := &Node[T]{Val: values[idx]}
	current := head
	for idx < len(values)-1 {
		idx++
		next := &Node[T]{Val: values[idx]}
		current.Next = next
		current = next
	}

	return head
}