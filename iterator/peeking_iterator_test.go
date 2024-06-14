package iterator

import (
	"reflect"
	"testing"
)

// Helper function to compare two lists of values
func compare[T any](t *testing.T, got []T, expect []T) {
	for idx := range got {
		if !reflect.DeepEqual(got[idx], expect[idx]) {
			t.Fatalf("Error - expected %T(%v), but got %T(%v)", expect[idx], expect[idx], got[idx], got[idx])
		}
	}
}

func TestPeekIterator(t *testing.T) {
	iter := &Iterator[int]{
		0,
		2,
		[]int{1,2,3},
	}

	peek_iter := &PeekingIterator[int]{iter}

	num_values := []int{}
	num_values = append(num_values, peek_iter.Next())
	num_values = append(num_values, peek_iter.Peek())
	num_values = append(num_values, peek_iter.Next())
	num_values = append(num_values, peek_iter.Next())
	expected_nums := []int{1,2,2,3}

	bool_values := []bool{}
	bool_values = append(bool_values, peek_iter.HasNext())
	expected_bools := []bool{false}

	compare[int](t, num_values, expected_nums)
	compare[bool](t, bool_values, expected_bools)
}