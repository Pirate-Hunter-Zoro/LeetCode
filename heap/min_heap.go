package heap

import (
	"cmp"
)

// MinHeap struct has a slice that holds the array
type MinHeap[K cmp.Ordered] struct {
	array []K
	count int
}

func NewMinHeap[K cmp.Ordered]() *MinHeap[K] {
	return &MinHeap[K]{count: 0}
}

// Insert adds an element to the heap
func (h *MinHeap[K]) Insert(key K) {
	h.array = append(h.array, key)
	h.MinHeapifyUp(len(h.array) - 1)
	h.count++
}

// Helps to fix the heap after inserting a value at the bottom
func (h *MinHeap[K]) MinHeapifyUp(idx int) {
	// If I am bigger than my parent, I switch places with my parent, and go from there
	if idx == 0 {
		return
	}
	parent_idx := parent(idx)
	for h.array[idx] < h.array[parent_idx] {
		h.swap(idx, parent_idx)
		idx = parent_idx
		parent_idx = parent(idx)
	}
}

// Swap two values in the underlying array
func (h *MinHeap[K]) swap(i1, i2 int) {
	h.array[i1], h.array[i2] = h.array[i2], h.array[i1]
}

// Determine if the heap is empty
func (h *MinHeap[K]) Empty() bool {
	return h.count == 0
}

// Extract returns the largest key, and removes it from the heap
func (h *MinHeap[K]) Extract() K {
	if h.count > 0 {
		h.count--
	}

	v := h.array[0]
	last := len(h.array) - 1

	h.array[0] = h.array[last]
	h.array = h.array[:last]

	h.minHeapifyDown(0)

	return v
}

// Helps to fix the heap after replacing a value at the root
func (h *MinHeap[K]) minHeapifyDown(idx int) {
	if right(idx) < len(h.array) && (h.array[right(idx)] < h.array[idx] || h.array[left(idx)] < h.array[idx]) {
		if h.array[right(idx)] < h.array[left(idx)] {
			h.swap(right(idx), idx)
			h.minHeapifyDown(right(idx))
		} else {
			h.swap(left(idx), idx)
			h.minHeapifyDown(left(idx))
		}
	} else if left(idx) < len(h.array) && (h.array[left(idx)] < h.array[idx]) {
		h.swap(left(idx), idx)
		h.minHeapifyDown(left(idx))
	}
}

// Peek the lowest value on top without removing it
func (h *MinHeap[K]) Peek() K {
	return h.array[0]
}
