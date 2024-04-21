package stack

type Stack[T any] struct {
	values []T
	length int
}

func New[T any]() *Stack[T] {
	return &Stack[T]{[]T{}, 0}
}

func (s *Stack[T]) Push(t T) {
	s.length++
	s.values = append(s.values, t)
}

func (s *Stack[T]) Pop() T {
	v := s.values[len(s.values)-1]
	if len(s.values) == 1 {
		s.values = []T{}
	} else {
		s.values = s.values[:len(s.values)-1]
	}
	s.length--
	return v
}

func (s *Stack[T]) Empty() bool {
	return s.length == 0
}