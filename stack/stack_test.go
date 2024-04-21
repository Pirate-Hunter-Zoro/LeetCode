package stack

import (
	"testing"
)

func TestStack(t *testing.T) {
	s := New[int]()
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