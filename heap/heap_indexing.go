package heap

// Parent of the given index
func parent(i int) int {
	return (i - 1) / 2
}

// Left child of the given index
func left(i int) int {
	return 2*i + 1
}

// Right child of the given index
func right(i int) int {
	return 2 * (i + 1)
}