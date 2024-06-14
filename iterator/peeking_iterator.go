package iterator

/*
Design an iterator that supports the peek operation on an existing iterator in addition to the hasNext and the next operations.

Implement the PeekingIterator class:

- PeekingIterator(Iterator<int> nums) Initializes the object with the given integer iterator iterator.
- int next() Returns the next element in the array and moves the pointer to the next element.
- boolean hasNext() Returns true if there are still elements in the array.
- int peek() Returns the next element in the array without moving the pointer.
Note: Each language may have a different implementation of the constructor and Iterator, but they all support the int next() and boolean hasNext() functions.

Link:
https://leetcode.com/problems/peeking-iterator/description/
*/
type PeekingIterator[T any] struct {
    iterator *Iterator[T]
}

func PeekingIteratorConstructor[T any](iter *Iterator[T]) *PeekingIterator[T] {
    return &PeekingIterator[T]{iter}
}

func (pit *PeekingIterator[T]) HasNext() bool {
    return false
}

func (pit *PeekingIterator[T]) Next() T {
    return (*pit.iterator).Next()
}

func (pit *PeekingIterator[T]) Peek() T {
    return (*pit.iterator).Next()
}