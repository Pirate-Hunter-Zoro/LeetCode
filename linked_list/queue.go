package linked_list

type Queue[T any] struct {
	values_head *Node[T]
	values_tail *Node[T]
	length      int
}

func NewQueue[T any]() *Queue[T] {
	return &Queue[T]{nil, nil, 0}
}

func (q *Queue[T]) Enqueue(v T) {
	if q.values_head == nil {
		new_head := &Node[T]{Val: v}
		q.values_head = new_head
		q.values_tail = new_head
	} else {
		new_tail := &Node[T]{Val: v}
		q.values_tail.Next = new_tail
		q.values_tail = new_tail
	}
	q.length++
}

func (q *Queue[T]) Dequeue() T {
	v := q.values_head.Val
	q.values_head = q.values_head.Next
	q.length--
	return v
}

func (q *Queue[T]) Peek() T {
	return q.values_head.Val
}

func (q *Queue[T]) Empty() bool {
	return q.length == 0
}

func (q *Queue[T]) Length() int {
	return q.length
}