package heap

// CustomMaxHeap struct has a slice that holds the array
type CustomMaxHeap[K any] struct {
	array []K
	greater func(first K, second K) bool
}

func NewCustomMaxHeap[K any](greater func(first K, second K) bool) *CustomMaxHeap[K] {
	return &CustomMaxHeap[K]{
		array: []K{},
		greater: greater,
	}
}

// Insert adds an element to the heap
func (h *CustomMaxHeap[K]) Insert(key K) {
	h.array = append(h.array, key)
	h.CustomMaxHeapifyUp(len(h.array) - 1)
}

// Helps to fix the heap after inserting a value at the bottom
func (h *CustomMaxHeap[K]) CustomMaxHeapifyUp(idx int) {
	// If I am bigger than my parent, I switch places with my parent, and go from there
	if idx == 0 {
		return
	}
	parent_idx := parent(idx)
	for h.greater(h.array[idx],h.array[parent_idx]) {
		h.swap(idx, parent_idx)
		idx = parent_idx
		parent_idx = parent(idx)
	}
}

// Swap two values in the underlying array
func (h *CustomMaxHeap[K]) swap(i1, i2 int) {
	h.array[i1], h.array[i2] = h.array[i2], h.array[i1]
}

// Extract returns the largest key, and removes it from the heap
func (h *CustomMaxHeap[K]) Extract() K {
	v := h.array[0]
	last := len(h.array) - 1

	h.array[0] = h.array[last]
	h.array = h.array[:last]

	h.CustomMaxHeapifyDown(0)

	return v
}

// Helps to fix the heap after replacing a value at the root
func (h *CustomMaxHeap[K]) CustomMaxHeapifyDown(idx int) {
	if right(idx) < len(h.array) && (h.greater(h.array[right(idx)],h.array[idx]) || h.greater(h.array[left(idx)],h.array[idx])) {
		if h.greater(h.array[right(idx)],(h.array[left(idx)])) {
			h.swap(right(idx), idx)
			h.CustomMaxHeapifyDown(right(idx))
		} else {
			h.swap(left(idx), idx)
			h.CustomMaxHeapifyDown(left(idx))
		}
	} else if left(idx) < len(h.array) && (h.greater(h.array[left(idx)],h.array[idx])) {
		h.swap(left(idx), idx)
		h.CustomMaxHeapifyDown(left(idx))
	}
}

// Peek the lowest value on top without removing it
func (h *CustomMaxHeap[K]) Peek() K {
	return h.array[0]
}

func (h *CustomMaxHeap[K]) Empty() bool {
	return len(h.array) == 0
}