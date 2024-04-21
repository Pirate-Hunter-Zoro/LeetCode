package queue

import (
	"testing"
)

func TestQueue(t *testing.T) {
	q := New[int]()
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