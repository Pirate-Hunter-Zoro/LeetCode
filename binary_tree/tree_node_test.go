package binary_tree

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

func TestNewTree(t *testing.T) {
	type input struct {
		values []int
	}
	inputs := []input{
		{[]int{1,2,5,3,NULL,6,NULL,4,NULL,7}},
		{[]int{1,401,NULL,349,88,90,NULL,92,NULL,NULL,NULL,NULL,63}},
	}

	expected_outputs := []*TreeNode{
		{
			Val: 1,
			Left: &TreeNode{
				Val: 2,
				Left: &TreeNode{
					Val: 3,
					Left: &TreeNode{
						Val: 4,
					},
				},
			},
			Right: &TreeNode{
				Val: 5,
				Left: &TreeNode{
					Val: 6,
					Left: &TreeNode{
						Val: 7,
					},
				},
			},
		},
		{
			Val: 1,
			Left: &TreeNode{
				Val: 401,
				Left: &TreeNode{
					Val: 349,
					Left: &TreeNode{
						Val: 90,
					},
				},
				Right: &TreeNode{
					Val: 88,
					Left: &TreeNode{
						Val: 92,
						Right: &TreeNode{
							Val: 63,
						},
					},
				},
			},
		},
	}

	f := func(i input) *TreeNode {
		return NewTree(i.values)
	}

	testResults(t, f, inputs, expected_outputs)
}