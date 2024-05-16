package linked_list

import (
	"testing"
)

func TestQueue(t *testing.T) {
	q := NewQueue[int]()
	for i:=0; i<10; i++ {
		q.Enqueue(i)
	}
	for i:=0; i<10; i++ {
		v := q.Dequeue()
		if v != i {
			t.Fatalf("Error - expected %d, but got %d", i, v)
		}
	}
	if !q.Empty() {
		t.Fatalf("Expected q.Empty() to be %t, but was %t", true, false)
	}
}

func TestStack(t *testing.T) {
	s := NewStack[int]()
	for i:=0; i<10; i++ {
		s.Push(i)
	}
	for i:=9; i>=0; i-- {
		v := s.Pop()
		if v != i {
			t.Fatalf("Error - expected %d, but got %d", i, v)
		}
	}
	if !s.Empty() {
		t.Fatalf("Expected s.Empty() to be %t, but was %t", true, false)
	}
}