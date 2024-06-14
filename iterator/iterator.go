package iterator

type Iterator[T any] struct {
	idx     int
	max_idx int
	values  []T
}
func(it *Iterator[T]) Next() T {
	v := it.values[it.idx]
	it.idx++
	return v
}
func(it *Iterator[T]) HasNext() bool {
	return it.idx <= it.max_idx
}