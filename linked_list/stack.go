package linked_list;

type Stack[T any] struct {
	values_head *Node[T]
	length      int
}

func NewStack[T any]() *Stack[T] {
	return &Stack[T]{nil, 0}
}

func (s *Stack[T]) Push(t T) {
	s.length++
	new_head := &Node[T]{Val: t}
	new_head.Next = s.values_head
	s.values_head = new_head
}

func (s *Stack[T]) Pop() T {
	v := s.values_head.Val
	s.values_head = s.values_head.Next
	s.length--
	return v
}

func (s *Stack[T]) Peek() T {
	return s.values_head.Val
}

func (s *Stack[T]) Empty() bool {
	return s.length == 0
}

func (s *Stack[T]) Length() int {
	return s.length
}