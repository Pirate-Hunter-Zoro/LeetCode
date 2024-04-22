package heap

import (
	"testing"
)

func TestMaxHeap(t *testing.T) {
	h := &MaxHeap[int]{}
	for i := 0; i < 10; i++ {
		h.Insert(i)
	}
	for i := 9; i >= 0; i-- {
		v := h.Extract()
		if v != i {
			t.Fatalf("Error - expected %d, but got %d", i, v)
		}
	}
}

func TestMinHeap(t *testing.T) {
	h := &MinHeap[int]{}
	for i := 0; i < 10; i++ {
		h.Insert(i)
	}
	for i := 0; i < 9; i++ {
		v := h.Extract()
		if v != i {
			t.Fatalf("Error - expected %d, but got %d", i, v)
		}
	}
}
