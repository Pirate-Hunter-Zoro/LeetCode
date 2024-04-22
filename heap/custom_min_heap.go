package heap

// CustomMinHeap struct has a slice that holds the array
type CustomMinHeap[K Ordered] struct {
	array []*K
}

// Insert adds an element to the heap
func (h *CustomMinHeap[K]) Insert(key *K) {
	h.array = append(h.array, key)
	h.CustomMinHeapifyUp(len(h.array) - 1)

}

// Helps to fix the heap after inserting a value at the bottom
func (h *CustomMinHeap[K]) CustomMinHeapifyUp(idx int) {
	// If I am bigger than my parent, I switch places with my parent, and go from there
	if idx == 0 {
		return
	}
	parent_idx := parent(idx)
	for (*h.array[idx]).Less(*h.array[parent_idx]) {
		h.swap(idx, parent_idx)
		idx = parent_idx
		parent_idx = parent(idx)
	}
}

// Swap two values in the underlying array
func (h *CustomMinHeap[K]) swap(i1, i2 int) {
	h.array[i1], h.array[i2] = h.array[i2], h.array[i1]
}

// Extract returns the largest key, and removes it from the heap
func (h *CustomMinHeap[K]) Extract() *K {
	v := h.array[0]
	last := len(h.array) - 1

	h.array[0] = h.array[last]
	h.array = h.array[:last]

	h.CustomMinHeapifyDown(0)

	return v
}

// Helps to fix the heap after replacing a value at the root
func (h *CustomMinHeap[K]) CustomMinHeapifyDown(idx int) {
	if right(idx) < len(h.array) && ((*h.array[right(idx)]).Less(*h.array[idx]) || (*h.array[left(idx)]).Less(*h.array[idx])) {
		if (*h.array[right(idx)]).Less(*h.array[left(idx)]) {
			h.swap(right(idx), idx)
			h.CustomMinHeapifyDown(right(idx))
		} else {
			h.swap(left(idx), idx)
			h.CustomMinHeapifyDown(left(idx))
		}
	} else if left(idx) < len(h.array) && ((*h.array[left(idx)]).Less(*h.array[idx])) {
		h.swap(left(idx), idx)
		h.CustomMinHeapifyDown(left(idx))
	}
}

// Peek the lowest value on top without removing it
func (h *CustomMinHeap[K]) Peek() *K {
	return h.array[0]
}

func (h *CustomMinHeap[K]) Empty() bool {
	return len(h.array) == 0
}