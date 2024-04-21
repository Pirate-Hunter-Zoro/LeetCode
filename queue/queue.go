package queue

type Queue[T any] struct {
	values []T
	length int
}

func New[T any]() *Queue[T] {
	return &Queue[T]{[]T{}, 0}
}

func (q *Queue[T]) Enqueue(v T) {
	q.values = append(q.values, v)
	q.length++
}

func (q *Queue[T]) Dequeue() T {
	v := q.values[0]
	if len(q.values) == 1 {
		q.values = []T{}
	} else {
		q.values = q.values[1:]
	}
	q.length--
	return v
}

func (q *Queue[T]) Empty() bool {
	return q.length == 0
}

func (q *Queue[T]) Length() int {
	return q.length
}