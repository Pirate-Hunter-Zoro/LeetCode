package modulo

import (
	"reflect"
	"testing"
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

func TestModularPow(t *testing.T) {
	type input struct {
		base  int
		power int
	}
	inputs := []input{
		{2, 20},
		{2, 10100},
		{2, 10098},
		{2, 10099},
	}

	expected_outputs := []int{
		1048576,
		569525940,
		142381485,
		284762970,
	}

	f := func(i input) int {
		return ModularPow(i.base, i.power)
	}

	testResults(t, f, inputs, expected_outputs)
}
