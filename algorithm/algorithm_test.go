package algorithm

import (
	"testing"
	"reflect"
)

/*
Helper method to test the results from the given method
*/
func testResults[T any, I any](t *testing.T, f func(I) T, inputs []I, expected_outputs []T) {
	for i := 0; i < len(inputs); i++ {
		input := inputs[i]
		expected := expected_outputs[i]
		output := f(input)
		if !reflect.DeepEqual(expected, output) {
			t.Fatalf("Error - expected %T(%v), but got %T(%v)", expected, expected, output, output)
		}
	}
}

func TestLongestIncreasingSubsequence(t *testing.T) {
	type input struct {
		vals	[]int
	}
	inputs := []input{
		{[]int{10,9,2,5,3,7,101,18}},
		{[]int{4,10,4,3,8,9}},
	}
	expected_outputs := []int{
		4,
		3,
	}

	f := func(i input) int {
		return LongestIncreasingSubsequence(i.vals)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestBinarySearch(t *testing.T) {
	type input struct {
		nums 	[]int
		target 	int
	}
	inputs := []input{
		{[]int{0,4,5,6,7,8}, 5},
		{[]int{3,4,7,8,10,11}, 11},
		{[]int{0}, 1},
	}
	expected_outputs := []int{
		2,
		5,
		-1,
	}

	f := func(i input) int {
		return BinarySearch(i.nums, i.target)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestBinarySearchMeetOrLower(t *testing.T) {
	type input struct {
		nums 	[]int
		target 	int
	}
	inputs := []input{
		{[]int{0,4,5,6,7,8}, 5},
		{[]int{3,4,7,8,10,11,11,11,12}, 11},
		{[]int{1}, 2},
		{[]int{3,4,7,8,10,11}, 9},
		{[]int{4,5,6,7,10}, 3},
	}
	expected_outputs := []int{
		2,
		7,
		0,
		3,
		-1,
	}

	f := func(i input) int {
		return BinarySearchMeetOrLower(i.nums, i.target)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestBinarySearchMeetOrExceed(t *testing.T) {
	type input struct {
		nums 	[]int
		target 	int
	}
	inputs := []input{
		{[]int{0,4,6,7,8}, 5},
		{[]int{3,4,7,8,10,10,10,11,11,11,12}, 11},
		{[]int{1}, 2},
		{[]int{3,4,7,8,10,11}, 9},
		{[]int{4,5,6,7,10}, 11},
		{[]int{3,4,7,7,9,11}, 9},
	}
	expected_outputs := []int{
		2,
		7,
		-1,
		4,
		-1,
		4,
	}

	f := func(i input) int {
		return BinarySearchMeetOrHigher(i.nums, i.target)
	}

	testResults(t, f, inputs, expected_outputs)
}