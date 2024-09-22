package linked_list

type Deque[T any] struct {
	front *Stack[T]
	back *Queue[T]
}

func NewDeque[T any]() *Deque[T] {
	return &Deque[T]{front: NewStack[T](), back: NewQueue[T]()}
}

func (d *Deque[T]) PushFront(v T) {
	d.front.Push(v)
}

func (d *Deque[T]) EnqueueBack(v T) {
	d.back.Enqueue(v)
}

func (d *Deque[T]) GetNext() T {
	if d.front.Empty() {
		return d.back.Dequeue()
	} else {
		return d.front.Pop()
	}
}

func (d *Deque[T]) Empty() bool {
	return d.front.Empty() && d.back.Empty()
}