package solution

import (
	"fmt"
	"leetcode/binary_tree"
	"leetcode/float_rounding"
	"leetcode/list_node"
	"reflect"
	"sort"
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

func TestRemoveZeroSum(t *testing.T) {
	type input struct {
		head *list_node.ListNode
	}
	inputs := []input{
		{
			&list_node.ListNode{
				Val: -1,
				Next: &list_node.ListNode{
					Val: -2,
					Next: &list_node.ListNode{
						Val: 2,
						Next: &list_node.ListNode{
							Val: -1,
							Next: &list_node.ListNode{
								Val: 0,
							},
						},
					},
				},
			},
		},
	}

	expected_outputs := []*list_node.ListNode{
		{
			Val: -1,
			Next: &list_node.ListNode{
				Val: -1,
			},
		},
	}

	f := func(i input) *list_node.ListNode {
		return removeZeroSumSublists(i.head)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestCountSubarraysWithSum(t *testing.T) {
	type test_case struct {
		nums []int
		goal int
	}

	inputs := []test_case{
		{
			nums: []int{1, 0, 1, 0, 1},
			goal: 2,
		},
		{
			nums: []int{0, 0, 0, 0, 0},
			goal: 0,
		},
		{
			nums: []int{0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0},
			goal: 5,
		},
		{
			nums: []int{0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0},
			goal: 3,
		},
		{
			nums: []int{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			goal: 1,
		},
	}

	expected_outputs := []int{
		4,
		15,
		10,
		48,
		10,
	}

	f := func(input test_case) int {
		return numSubArraysWithSum(input.nums, input.goal)
	}
	testResults(t, f, inputs, expected_outputs)
}

func TestMinimumDifference(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{3, 1, 2}},
		{[]int{7, 9, 5, 8, 1, 3}},
	}

	expected_outputs := []int{
		-1,
		1,
	}

	f := func(i input) int {
		return int(minimumDifference(i.nums))
	}
	testResults(t, f, inputs, expected_outputs)
}

func TestFindMaxLength(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{0, 1}},
		{[]int{0, 1, 1, 0, 1, 1, 1, 0}},
		{[]int{0, 0, 1, 0, 0, 0, 1, 1}},
		{[]int{1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1}},
	}

	expected_outputs := []int{
		2,
		4,
		6,
		670,
	}

	f := func(i input) int {
		return findMaxLength(i.nums)
	}
	testResults(t, f, inputs, expected_outputs)
}

func TestFindMindArrowShots(t *testing.T) {
	type input struct {
		balloons [][]int
	}
	inputs := []input{
		{[][]int{{10, 16}, {2, 8}, {1, 6}, {7, 12}}},
	}

	expected_outputs := []int{
		2,
	}

	f := func(i input) int {
		return findMinArrowShots(i.balloons)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestNthSuperUglyNumber(t *testing.T) {
	type input struct {
		primes []int
		n      int
	}
	inputs := []input{
		{[]int{2, 7, 13, 19}, 12},
	}

	expected_outputs := []int{
		32,
	}

	f := func(i input) int {
		return nthSuperUglyNumber(i.n, i.primes)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestLeastInterval(t *testing.T) {
	type input struct {
		tasks     []byte
		num_tasks int
	}
	inputs := []input{
		{[]byte{65, 65, 65, 66, 66, 66}, 2},
		{[]byte{65, 67, 65, 66, 68, 66}, 1},
		{[]byte{65, 65, 65, 66, 66, 66}, 3},
	}

	expected_outputs := []int{
		8,
		6,
		10,
	}

	f := func(i input) int {
		return leastInterval(i.tasks, i.num_tasks)
	}
	testResults(t, f, inputs, expected_outputs)
}

func TestPivotSearch(t *testing.T) {
	type input struct {
		nums   []int
		target int
	}
	inputs := []input{
		{[]int{1, 3, 5}, 0},
		{[]int{2, 3, 4, 5, 1}, 1},
		{[]int{2, 4, 7, 9, 0}, 9},
		{[]int{5, 7, 8, 0, 3, 4}, 7},
		{[]int{1, 3, 5}, 1},
	}
	expected_outputs_1 := []int{
		-1,
		4,
		3,
		1,
		0,
	}
	f_1 := func(i input) int {
		return search(i.nums, i.target)
	}
	testResults(t, f_1, inputs, expected_outputs_1)

	inputs = []input{
		{[]int{2, 5, 6, 0, 0, 1, 2}, 0},
		{[]int{2, 5, 6, 0, 0, 1, 2}, 3},
		{[]int{2, 2, 2, 3, 2, 2, 2}, 3},
		{[]int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1}, 2},
	}
	expected_outputs_2 := []bool{
		true,
		false,
		true,
		true,
	}
	f_2 := func(i input) bool {
		return searchRepeats(i.nums, i.target)
	}
	testResults(t, f_2, inputs, expected_outputs_2)

}

func TestFindDuplicate(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{1, 3, 4, 2, 2}},
		{[]int{3, 1, 3, 4, 2}},
		{[]int{3, 3, 3, 3, 3}},
	}

	expected_outputs := []int{
		2,
		3,
		3,
	}

	f := func(i input) int {
		return findDuplicate(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestFindDuplicates(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{1, 2, 3, 5, 3, 2}},
	}

	expected_outputs := [][]int{
		{2, 3},
	}

	f := func(i input) []int {
		repeats := findDuplicates(i.nums)
		sort.SliceStable(repeats, func(i, j int) bool {
			return repeats[i] < repeats[j]
		})
		return repeats
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestCountVowelPermuations(t *testing.T) {
	type input struct {
		n int
	}
	inputs := []input{
		{1},
		{2},
		{5},
	}

	expected_outputs := []int{
		5,
		10,
		68,
	}

	f := func(i input) int {
		return countVowelPermutation(i.n)
	}
	testResults(t, f, inputs, expected_outputs)
}

func TestPoorPigs(t *testing.T) {
	type input struct {
		buckets       int
		minutesToDie  int
		minutesToTest int
	}
	inputs := []input{
		{4, 15, 15},
		{4, 15, 30},
	}

	expected_outputs := []int{
		2,
		2,
	}

	f := func(i input) int {
		return poorPigs(i.buckets, i.minutesToDie, i.minutesToTest)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestNumSubarrayProductLessThanK(t *testing.T) {
	type input struct {
		nums []int
		k    int
	}
	inputs := []input{
		{[]int{10, 5, 2, 6}, 100},
		{[]int{1, 2, 3}, 0},
		{[]int{10, 9, 10, 4, 3, 8, 3, 3, 6, 2, 10, 10, 9, 3}, 19},
	}

	expected_outputs := []int{
		8,
		0,
		18,
	}

	f := func(i input) int {
		return numSubarrayProductLessThanK(i.nums, i.k)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestFindCheapestPrice(t *testing.T) {
	type input struct {
		n       int
		flights [][]int
		src     int
		dst     int
		k       int
	}
	inputs := []input{
		{4, [][]int{{0, 1, 100}, {1, 2, 100}, {2, 0, 100}, {1, 3, 600}, {2, 3, 200}}, 0, 3, 1},
		{3, [][]int{{0, 1, 100}, {1, 2, 100}, {0, 2, 500}}, 0, 2, 1},
		{3, [][]int{{0, 1, 100}, {1, 2, 100}, {0, 2, 500}}, 0, 2, 0},
		{5, [][]int{{0, 1, 5}, {1, 2, 5}, {0, 3, 2}, {3, 1, 2}, {1, 4, 1}, {4, 2, 1}}, 0, 2, 2},
		{5, [][]int{{0, 1, 1}, {0, 2, 5}, {1, 2, 1}, {2, 3, 1}, {3, 4, 1}}, 0, 4, 2},
		{5, [][]int{{0, 1, 100}, {0, 2, 100}, {0, 3, 10}, {1, 2, 100}, {1, 4, 10}, {2, 1, 10}, {2, 3, 100}, {2, 4, 100}, {3, 2, 10}, {3, 4, 100}}, 0, 4, 3},
	}

	expected_outputs := []int{
		700,
		200,
		500,
		7,
		7,
		40,
	}

	f := func(i input) int {
		return findCheapestPrice(i.n, i.flights, i.src, i.dst, i.k)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestFindAllPeople(t *testing.T) {
	type input struct {
		n           int
		meetings    [][]int
		firstPerson int
	}
	inputs := []input{
		{6, [][]int{{1, 2, 5}, {2, 3, 8}, {1, 5, 10}}, 1},
		{4, [][]int{{3, 1, 3}, {1, 2, 2}, {0, 3, 3}}, 3},
		{5, [][]int{{3, 4, 2}, {1, 2, 1}, {2, 3, 1}}, 1},
		{5, [][]int{{1, 4, 3}, {0, 4, 3}}, 3},
	}

	expected_outputs := [][]int{
		{0, 1, 2, 3, 5},
		{0, 1, 3},
		{0, 1, 2, 3, 4},
		{0, 1, 3, 4},
	}

	f := func(i input) []int {
		people := findAllPeople(i.n, i.meetings, i.firstPerson)
		sort.SliceStable(people, func(i, j int) bool { return people[i] < people[j] })
		return people
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestCanTraverseAllPairs(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{2, 3, 6}},
		{[]int{3, 9, 5}},
		{[]int{4, 3, 12, 8}},
		{[]int{40, 22, 15}},
		{[]int{30, 30, 30, 49, 14, 14, 50, 42, 42, 35, 20, 24}},
		{[]int{91, 66, 66, 100, 30, 60, 84, 90}},
	}

	expected_outputs := []bool{
		true,
		false,
		true,
		true,
		true,
		true,
	}

	f := func(i input) bool {
		return canTraverseAllPairs(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMaxEnvelopes(t *testing.T) {
	type input struct {
		envelopes [][]int
	}
	inputs := []input{
		{[][]int{{5, 4}, {6, 4}, {6, 7}, {2, 3}}},
		{[][]int{{1, 1}, {1, 1}, {1, 1}}},
		{[][]int{{46, 89}, {50, 53}, {52, 68}, {72, 45}, {77, 81}}},
		{[][]int{{1, 3}, {3, 5}, {6, 7}, {6, 8}, {8, 4}, {9, 5}}},
	}

	expected_outputs := []int{
		3,
		1,
		3,
		3,
	}

	f := func(i input) int {
		return maxEnvelopes(i.envelopes)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestCountNumbersWithUniqueDigits(t *testing.T) {
	type input struct {
		n int
	}
	inputs := []input{
		{0},
		{2},
	}

	expected_outputs := []int{
		1,
		91,
	}

	f := func(i input) int {
		return countNumbersWithUniqueDigits(i.n)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMinRemoveToMakeValidParentheses(t *testing.T) {
	type input struct {
		s string
	}
	inputs := []input{
		{"lee(t(c)o)de)"},
		{"a)b(c)d"},
		{"))(("},
	}

	expected_outputs := []string{
		"lee(t(c)o)de",
		"ab(c)d",
		"",
	}

	f := func(i input) string {
		return minRemoveToMakeValid(i.s)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMinSwaps(t *testing.T) {
	type input struct {
		s string
	}
	inputs := []input{
		{"][]["},
		{"]]][[["},
		{"[]"},
	}

	expected_outputs := []int{
		1,
		2,
		0,
	}

	f := func(i input) int {
		return minSwaps(i.s)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestCheckValidString(t *testing.T) {
	type input struct {
		s string
	}
	inputs := []input{
		{"()"},
		{"(*)"},
		{"(*))"},
		{"((((()(()()()*()(((((*)()*(**(())))))(())()())(((())())())))))))(((((())*)))()))(()((*()*(*)))(*)()"},
	}

	expected_outputs := []bool{
		true,
		true,
		true,
		true,
	}

	f := func(i input) bool {
		return checkValidString(i.s)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestCountDigitOne(t *testing.T) {
	type input struct {
		n int
	}
	inputs := []input{
		{13},
		{0},
		{10},
		{20},
	}

	expected_outputs := []int{
		6,
		0,
		2,
		12,
	}

	f := func(i input) int {
		return countDigitOne(i.n)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestCoinChange(t *testing.T) {
	type input struct {
		coins  []int
		amount int
	}

	inputs := []input{
		{[]int{1, 2, 5}, 11},
		{[]int{2}, 3},
		{[]int{1}, 0},
	}

	expected_outputs := []int{
		3,
		-1,
		0,
	}

	f := func(i input) int {
		return coinChange(i.coins, i.amount)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMaxValueOfCoins(t *testing.T) {
	type input struct {
		piles [][]int
		k     int
	}
	inputs := []input{
		{[][]int{{1, 100, 3}, {7, 8, 9}}, 2},
		{[][]int{{100}, {100}, {100}, {100}, {100}, {100}, {1, 1, 1, 1, 1, 1, 700}}, 7},
	}

	expected_outputs := []int{
		101,
		706,
	}

	f := func(i input) int {
		return maxValueOfCoins(i.piles, i.k)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestShortestSuperstring(t *testing.T) {
	type input struct {
		words []string
	}
	inputs := []input{
		{[]string{"catg", "ctaagt", "gcta", "ttca", "atgcatc"}},
	}

	expected_outputs := []string{
		"gctaagttcatgcatc",
	}

	f := func(i input) string {
		return shortestSuperstring(i.words)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestLargestDivisibleSubset(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{1, 2, 3}},
		{[]int{1, 2, 4, 8}},
	}

	expected_outputs := [][]int{
		{1, 2},
		{1, 2, 4, 8},
	}

	f := func(i input) []int {
		return largestDivisibleSubset(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMinimumOneBitOperations(t *testing.T) {
	type input struct {
		n int
	}
	inputs := []input{
		{3},
		{6},
		{0},
		{9},
		{333},
	}
	expected_outputs := []int{
		2,
		4,
		0,
		14,
		393,
	}

	f := func(i input) int {
		return minimumOneBitOperations(i.n)
	}

	testResults(t, f, inputs, expected_outputs)

	fmt.Printf("%d\n", minimumOneBitOperations(339963))

}

func TestMinOperations(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{4, 2, 5, 3}},
		{[]int{1, 2, 3, 5, 6}},
		{[]int{1, 10, 100, 1000}},
		{[]int{1, 2, 3, 10, 20}},
		{[]int{8, 10, 16, 18, 10, 10, 16, 13, 13, 16}},
	}
	expected_outputs := []int{
		0,
		1,
		3,
		2,
		6,
	}

	f := func(i input) int {
		return minOperations(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestSumOfDistancesInTree(t *testing.T) {
	type input struct {
		n     int
		edges [][]int
	}
	inputs := []input{
		{6, [][]int{{0, 1}, {0, 2}, {2, 3}, {2, 4}, {2, 5}}},
		{1, [][]int{}},
		{2, [][]int{{1, 0}}},
		{13, [][]int{{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {0, 8}, {0, 9}, {0, 10}, {0, 11}, {0, 12}}},
		{3, [][]int{{2, 1}, {0, 2}}},
	}
	expected_outputs := [][]int{
		{8, 12, 6, 10, 10, 10},
		{0},
		{1, 1},
		{12, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23},
		{3, 3, 2},
	}

	f := func(i input) []int {
		return sumOfDistancesInTree(i.n, i.edges)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestNumIslands(t *testing.T) {
	type input struct {
		grid [][]byte
	}
	inputs := []input{
		{
			[][]byte{
				{'1', '1', '1', '1', '0'},
				{'1', '1', '0', '1', '0'},
				{'1', '1', '0', '0', '0'},
				{'0', '0', '0', '0', '0'},
			},
		},
		{
			[][]byte{
				{'1', '1', '0', '0', '0'},
				{'1', '1', '0', '0', '0'},
				{'0', '0', '1', '0', '0'},
				{'0', '0', '0', '1', '1'},
			},
		},
	}
	expected_outputs := []int{
		1,
		3,
	}

	f := func(i input) int {
		return numIslands(i.grid)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestFindFarmland(t *testing.T) {
	type input struct {
		land [][]int
	}
	inputs := []input{
		{[][]int{
			{1, 0, 0},
			{0, 1, 1},
			{0, 1, 1},
		},
		},
		{[][]int{
			{1, 1},
			{1, 1},
		},
		},
		{[][]int{
			{0},
		},
		},
	}
	expected_outputs := [][][]int{
		{
			{0, 0, 0, 0},
			{1, 1, 2, 2},
		},
		{
			{0, 0, 1, 1},
		},
		{},
	}

	f := func(i input) [][]int {
		return findFarmland(i.land)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestGetMoneyAmount(t *testing.T) {
	type input struct {
		n int
	}
	inputs := []input{
		{10},
		{1},
		{2},
	}
	expected_outputs := []int{
		16,
		0,
		1,
	}

	f := func(i input) int {
		return getMoneyAmount(i.n)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestOpenLock(t *testing.T) {
	type input struct {
		deadends []string
		target   string
	}
	inputs := []input{
		{
			[]string{
				"0201", "0101", "0102", "1212", "2002",
			},
			"0202",
		},
		{
			[]string{
				"8888",
			},
			"0009",
		},
		{
			[]string{
				"8887", "8889", "8878", "8898", "8788", "8988", "7888", "9888",
			},
			"8888",
		},
		{
			[]string{
				"0000",
			},
			"8888",
		},
		{
			[]string{
				"8430", "5911", "4486", "7174", "9772", "0731", "9550", "3449", "4437", "3837", "1870", "5798", "9583", "9512", "5686", "5131", "0736", "3051", "2141", "2989", "6368", "2004", "1012", "8736", "0363", "3589", "8568", "6457", "3467", "1967", "1055", "6637", "1951", "0575", "4603", "2606", "0710", "4169", "7009", "6554", "6128", "2876", "8151", "4423", "0727", "8130", "3571", "4801", "8968", "6084", "3156", "3087", "0594", "9811", "3902", "4690", "6468", "2743", "8560", "9064", "4231", "6056", "2551", "8556", "2541", "5460", "5657", "1151", "5123", "3521", "2200", "9333", "9685", "4871", "9138", "5807", "2191", "2601", "1792", "3470", "9096", "0185", "0367", "6862", "1757", "6904", "4485", "7973", "7201", "2571", "3829", "0868", "4632", "6975", "2026", "3463", "2341", "4647", "3680", "3282", "3761", "4410", "3397", "3357", "4038", "6505", "1655", "3812", "3558", "4759", "1112", "8836", "5348", "9113", "1627", "3249", "0537", "4227", "7952", "8855", "3592", "2054", "3175", "6665", "4088", "9959", "3809", "7379", "6949", "8063", "3686", "8078", "0925", "5167", "2075", "4665", "2628", "8242", "9831", "1397", "5547", "9449", "6512", "6083", "9682", "2215", "3236", "2457", "6211", "5536", "8674", "2647", "9752", "5433", "0186", "5904", "1526", "5347", "1387", "3153", "1353", "6069", "9995", "9496", "0003", "3400", "1692", "6870", "4445", "3063", "0708", "3278", "6961", "3063", "0249", "0375", "1763", "1804", "4695", "6493", "7573", "9977", "1108", "0856", "5631", "4799", "4164", "0844", "2600", "1785", "1587", "4510", "9012", "7497", "4923", "2560", "0338", "3839", "5624", "1980", "1514", "4634", "2855", "7012", "3626", "7032", "6145", "5663", "4395", "0724", "4711", "1573", "6904", "8100", "2649", "3890", "8110", "8067", "1460", "0186", "6098", "2459", "6991", "9372", "8539", "8418", "7944", "0499", "9276", "1525", "1281", "8738", "5054", "7869", "6599", "8018", "7530", "2327", "3681", "5248", "4291", "7300", "8854", "2591", "8744", "3052", "6369", "3669", "8501", "8455", "5726", "1211", "8793", "6889", "9315", "0738", "6805", "5980", "7485", "2333", "0140", "4708", "9558", "9026", "4349", "5978", "4989", "5238", "3217", "5938", "9660", "5858", "2118", "7657", "5896", "3195", "8997", "1688", "2863", "9356", "4208", "5438", "2642", "4138", "7466", "6154", "0926", "2556", "9574", "4497", "9633", "0585", "1390", "5093", "3047", "0430", "7482", "0750", "6229", "8714", "4765", "0941", "1780", "6262", "0925", "5631", "9167", "0885", "7713", "5576", "3775", "9652", "0733", "7467", "5301", "9365", "7978", "4736", "3309", "6965", "4703", "5897", "8460", "9619", "0572", "6297", "7701", "7554", "8669", "5426", "6474", "5540", "5038", "3880", "1657", "7574", "1108", "4369", "7782", "9742", "5301", "6984", "3158", "2869", "0599", "2147", "6962", "9722", "3597", "9015", "3115", "9051", "8269", "6967", "5392", "4401", "6579", "8997", "8933", "9297", "0151", "8820", "3297", "6723", "1755", "1163", "8896", "7122", "4859", "5504", "0857", "4682", "8177", "8702", "9167", "9410", "0130", "2789", "7492", "5938", "3012", "4137", "3414", "2245", "4292", "6945", "5446", "6614", "2977", "8640", "9242", "7603", "8349", "9420", "0538", "4222", "0599", "8459", "8738", "4764", "6717", "7575", "5965", "9816", "9975", "4994", "2612", "0344", "6450", "9088", "4898", "6379", "4127", "1574", "9044", "0434", "5928", "6679", "1753", "8940", "7563", "0545", "4575", "6407", "6213", "8327", "3978", "9187", "2996", "1956", "8819", "9591", "7802", "4747", "9094", "0179", "0806", "2509", "4026", "4850", "2495", "3945", "4994", "5971", "3401", "0218", "6584", "7688", "6138", "7047", "9456", "0173", "1406", "1564", "3055", "8725", "4835", "4737", "6279", "5291", "0145", "0002", "1263", "9518", "1251", "8224", "6779", "4113", "8680", "2946", "1685", "2057", "9520", "4099", "7785", "1134", "2152", "4719", "6038", "1599", "6750", "9273", "7755", "3134", "2345", "8208", "5750", "5850", "2019", "0350", "9013", "6911", "6095", "6843", "3157", "9049", "0801", "2739", "9691", "3511",
			},
			"2248",
		},
	}
	expected_outputs := []int{
		6,
		1,
		-1,
		-1,
		10,
	}

	f := func(i input) int {
		return openLock(i.deadends, i.target)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMinHeightTree(t *testing.T) {
	type input struct {
		n     int
		edges [][]int
	}
	inputs := []input{
		{
			4,
			[][]int{
				{1, 0}, {1, 2}, {1, 3},
			},
		},
		{
			6,
			[][]int{
				{3, 0}, {3, 1}, {3, 2}, {3, 4}, {5, 4},
			},
		},
	}

	expected_outputs := [][]int{
		{1},
		{3, 4},
	}

	f := func(i input) []int {
		return findMinHeightTrees(i.n, i.edges)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestWiggleMaxLength(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{1, 7, 4, 9, 2, 5}},
		{[]int{1, 17, 5, 10, 13, 15, 10, 5, 16, 8}},
		{[]int{1, 2, 3, 4, 5, 6, 7, 8, 9}},
		{[]int{3, 3, 3, 2, 5}},
	}

	expected_outputs := []int{
		6,
		7,
		2,
		3,
	}

	f := func(i input) int {
		return wiggleMaxLength(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestLongestIdealString(t *testing.T) {
	type input struct {
		s string
		k int
	}
	inputs := []input{
		{"acfgbd", 2},
		{"abcd", 3},
	}
	expected_outputs := []int{
		4,
		4,
	}

	f := func(i input) int {
		return longestIdealString(i.s, i.k)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMinFallingPathSum(t *testing.T) {
	type input struct {
		grid [][]int
	}
	inputs := []input{
		{
			[][]int{
				{1, 2, 3},
				{4, 5, 6},
				{7, 8, 9},
			},
		},
		{
			[][]int{
				{7},
			},
		},
		{
			[][]int{
				{-73, 61, 43, -48, -36},
				{3, 30, 27, 57, 10},
				{96, -76, 84, 59, -15},
				{5, -49, 76, 31, -7},
				{97, 91, 61, -46, 67},
			},
		},
		{
			[][]int{
				{50, -18, -38, 39, -20, -37, -61, 72, 22, 79},
				{82, 26, 30, -96, -1, 28, 87, 94, 34, -89},
				{55, -50, 20, 76, -50, 59, -58, 85, 83, -83},
				{39, 65, -68, 89, -62, -53, 74, 2, -70, -90},
				{1, 57, -70, 83, -91, -32, -13, 49, -11, 58},
				{-55, 83, 60, -12, -90, -37, -36, -27, -19, -6},
				{76, -53, 78, 90, 70, 62, -81, -94, -32, -57},
				{-32, -85, 81, 25, 80, 90, -24, 10, 27, -55},
				{39, 54, 39, 34, -45, 17, -2, -61, -81, 85},
				{-77, 65, 76, 92, 21, 68, 78, -13, 39, 22},
			},
		},
	}

	expected_outputs := []int{
		13,
		7,
		-192,
		-807,
	}

	f := func(i input) int {
		return minFallingPathSum(i.grid)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestFindRotateSteps(t *testing.T) {
	type input struct {
		ring string
		key  string
	}
	inputs := []input{
		{"godding", "gd"},
		{"godding", "godding"},
	}
	expected_outputs := []int{
		4,
		13,
	}

	f := func(i input) int {
		return findRotateSteps(i.ring, i.key)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestDistributeCoins(t *testing.T) {
	type input struct {
		root *binary_tree.TreeNode
	}
	inputs := []input{
		{binary_tree.NewTree([]int{3, 0, 0})},
		{binary_tree.NewTree([]int{0, 3, 0})},
	}
	expected_outputs := []int{
		2,
		3,
	}

	f := func(i input) int {
		return distributeCoins(i.root)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMinBitOperations(t *testing.T) {
	type input struct {
		nums []int
		k    int
	}

	inputs := []input{
		{
			[]int{2, 1, 3, 4},
			1,
		},
		{
			[]int{2, 0, 2, 0},
			0,
		},
	}

	expected_outputs := []int{
		2,
		0,
	}

	f := func(i input) int {
		return minBitOperations(i.nums, i.k)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestCanIWin(t *testing.T) {
	type input struct {
		maxChoosableInteger int
		desiredTotal        int
	}
	inputs := []input{
		{10, 11},
		{10, 0},
		{10, 1},
	}

	expected_outputs := []bool{
		false,
		true,
		true,
	}

	f := func(i input) bool {
		return canIWin(i.maxChoosableInteger, i.desiredTotal)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestCombinationSum(t *testing.T) {
	type input struct {
		nums   []int
		target int
	}
	inputs := []input{
		{[]int{1, 2, 3}, 4},
		{[]int{9}, 3},
		{[]int{3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}, 10},
		{[]int{154, 34, 208, 358, 427, 52, 328, 493, 304, 346, 118, 325, 7, 226, 169, 178, 499, 460, 349, 430, 259, 172, 400, 43, 451, 82, 409, 313, 175, 91, 289, 40, 205, 391, 343, 214, 307, 28, 418, 199, 241, 310, 238, 268, 244, 319, 1, 457, 124, 265, 496, 490, 130, 49, 181, 148, 316, 448, 397, 88, 337, 424, 136, 160, 229, 25, 100, 112, 46, 76, 166, 211, 94, 247, 142, 334, 322, 271, 352, 70, 367, 232, 58, 379, 133, 361, 394, 292, 4, 115, 286, 13, 64, 472, 16, 364, 196, 466, 433, 22, 415, 193, 445, 421, 301, 220, 31, 250, 340, 277, 145, 184, 382, 262, 202, 121, 373, 190, 388, 475, 478, 223, 163, 454, 370, 481, 109, 19, 73, 10, 376, 217, 487, 283, 151, 187, 439, 295, 67, 355, 385, 106, 463, 139, 37, 298, 253, 61, 442, 127, 103, 403, 97, 274, 484, 469, 412, 280, 235, 256, 406, 436, 157, 79, 85, 55}, 50},
	}

	expected_outputs := []int{
		7,
		0,
		9,
		83316385,
	}

	f := func(i input) int {
		return combinationSum(i.nums, i.target)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMaxSumBST(t *testing.T) {
	type input struct {
		root *binary_tree.TreeNode
	}
	inputs := []input{
		{binary_tree.NewTree([]int{1, 4, 3, 2, 4, 2, 5, binary_tree.NULL, binary_tree.NULL, binary_tree.NULL, binary_tree.NULL, binary_tree.NULL, binary_tree.NULL, 4, 6})},
		{binary_tree.NewTree([]int{4, 3, binary_tree.NULL, 1, 2})},
		{binary_tree.NewTree([]int{-4, -2, -5})},
	}
	expected_outputs := []int{
		20,
		2,
		0,
	}

	f := func(i input) int {
		return maxSumBST(i.root)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestIntegerReplacement(t *testing.T) {
	type input struct {
		n int
	}
	inputs := []input{
		{8},
		{7},
		{4},
	}
	expected_outputs := []int{
		3,
		4,
		2,
	}

	f := func(i input) int {
		return integerReplacement(i.n)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestWinnerSquareGame(t *testing.T) {
	type input struct {
		n int
	}
	inputs := []input{
		{1},
		{2},
		{4},
	}
	expected_outputs := []bool{
		true,
		false,
		true,
	}

	f := func(i input) bool {
		return winnerSquareGame(i.n)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestStoneGameII(t *testing.T) {
	type input struct {
		piles []int
	}
	inputs := []input{
		{[]int{2, 7, 9, 4, 4}},
		{[]int{1, 2, 3, 4, 5, 100}},
	}

	expected_outputs := []int{
		10,
		104,
	}

	f := func(i input) int {
		return stoneGameII(i.piles)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestStoneGameIII(t *testing.T) {
	type input struct {
		stoneValue []int
	}
	inputs := []input{
		{[]int{1, 2, 3, 7}},
		{[]int{1, 2, 3, -9}},
		{[]int{1, 2, 3, 6}},
	}

	expected_outputs := []string{
		"Bob",
		"Alice",
		"Tie",
	}

	f := func(i input) string {
		return stoneGameIII(i.stoneValue)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestStoneGameV(t *testing.T) {
	type input struct {
		stoneValue []int
	}
	inputs := []input{
		{[]int{6, 2, 3, 4, 5, 5}},
		{[]int{7, 7, 7, 7, 7, 7, 7}},
		{[]int{4}},
	}
	expected_outputs := []int{
		18,
		28,
		0,
	}

	f := func(i input) int {
		return stoneGameV(i.stoneValue)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestStoneGameVI(t *testing.T) {
	type input struct {
		aliceValues []int
		bobValues   []int
	}
	inputs := []input{
		{[]int{1, 3}, []int{2, 1}},
		{[]int{1, 2}, []int{3, 1}},
		{[]int{2, 4, 3}, []int{1, 6, 7}},
	}
	expected_outputs := []int{
		1,
		0,
		-1,
	}

	f := func(i input) int {
		return stoneGameVI(i.aliceValues, i.bobValues)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestStoneGameVII(t *testing.T) {
	type input struct {
		stones []int
	}
	inputs := []input{
		{[]int{5, 3, 1, 4, 2}},
		{[]int{7, 90, 5, 1, 100, 10, 10, 2}},
	}

	expected_outputs := []int{
		6,
		122,
	}

	f := func(i input) int {
		return stoneGameVII(i.stones)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestStoneGameVIII(t *testing.T) {
	type input struct {
		stones []int
	}
	inputs := []input{
		{[]int{-1, 2, -3, 4, -5}},
		{[]int{7, -6, 5, 10, 5, -2, -6}},
		{[]int{-10, -12}},
		{[]int{-10, -12, -10, -12}},
		{[]int{25, -35, -37, 4, 34, 43, 16, -33, 0, -17, -31, -42, -42, 38, 12, -5, -43, -10, -37, 12}},
	}

	expected_outputs := []int{
		5,
		13,
		-22,
		12,
		38,
	}

	f := func(i input) int {
		return stoneGameVIII(i.stones)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestStoneGameIX(t *testing.T) {
	type input struct {
		stones []int
	}
	inputs := []input{
		{[]int{2, 1}},
		{[]int{2}},
		{[]int{5, 1, 2, 4, 3}},
		{[]int{2, 3}},
		{[]int{2, 2, 2, 3}},
		{[]int{20, 3, 20, 17, 2, 12, 15, 17, 4}},
		{[]int{19, 2, 17, 20, 7, 17}},
		{[]int{1, 11, 12, 17, 6}},
		{[]int{7, 10, 1, 9, 19, 17, 1, 9, 19}},
		{[]int{4, 1}},
	}

	expected_outputs := []bool{
		true,
		false,
		false,
		false,
		true,
		true,
		true,
		true,
		true,
		false,
	}

	f := func(i input) bool {
		return stoneGameIX(i.stones)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestNumRescueBoats(t *testing.T) {
	type input struct {
		people []int
		limit  int
	}
	inputs := []input{
		{[]int{1, 2}, 3},
		{[]int{3, 2, 2, 1}, 3},
		{[]int{3, 5, 3, 4}, 5},
		{[]int{3, 1, 7}, 7},
		{[]int{5, 1, 4, 2}, 6},
	}
	expected_outputs := []int{
		1,
		3,
		4,
		2,
		2,
	}

	f := func(i input) int {
		return numRescueBoats(i.people, i.limit)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestDoubleIt(t *testing.T) {
	type input struct {
		head *list_node.ListNode
	}
	inputs := []input{
		{list_node.NewList([]int{1, 8, 9})},
		{list_node.NewList([]int{9, 9, 9})},
	}

	expected_outputs := []list_node.ListNode{
		*list_node.NewList([]int{3, 7, 8}),
		*list_node.NewList([]int{1, 9, 9, 8}),
	}

	f := func(i input) list_node.ListNode {
		return *doubleIt(i.head)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMaximumHappinessSum(t *testing.T) {
	type input struct {
		happiness []int
		k         int
	}
	inputs := []input{
		{[]int{1, 2, 3}, 2},
		{[]int{1, 1, 1, 1}, 2},
		{[]int{2, 3, 4, 5}, 1},
	}

	expected_outputs := []int64{
		int64(4),
		int64(1),
		int64(5),
	}

	f := func(i input) int64 {
		return maximumHappinessSum(i.happiness, i.k)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestKthSmallestPrimeFraction(t *testing.T) {
	type input struct {
		arr []int
		k   int
	}

	inputs := []input{
		{[]int{1, 2, 3, 5}, 3},
		{[]int{1, 7}, 1},
		{[]int{1, 13, 17, 59}, 6},
	}

	expected_outputs := [][]int{
		{2, 5},
		{1, 7},
		{13, 17},
	}

	f := func(i input) []int {
		return kthSmallestPrimeFraction(i.arr, i.k)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMincostToHireWorkers(t *testing.T) {
	type input struct {
		quality []int
		wage    []int
		k       int
	}
	inputs := []input{
		{[]int{10, 20, 5}, []int{70, 50, 30}, 2},
		{[]int{3, 1, 10, 10, 1}, []int{4, 8, 2, 2, 7}, 3},
		{[]int{2, 1, 5}, []int{17, 6, 4}, 2},
	}

	expected_outputs := []float64{
		float_rounding.RoundFloat(float64(105), 5),
		float_rounding.RoundFloat(float64(30)+float64(2)/float64(3), 5),
		float_rounding.RoundFloat(25.5, 5),
	}

	f := func(i input) float64 {
		return float_rounding.RoundFloat(mincostToHireWorkers(i.quality, i.wage, i.k), 5)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMatrixScore(t *testing.T) {
	type input struct {
		grid [][]int
	}
	inputs := []input{
		{
			[][]int{
				{0, 0, 1, 1},
				{1, 0, 1, 0},
				{1, 1, 0, 0},
			},
		},
		{
			[][]int{
				{0},
			},
		},
	}

	expected_outputs := []int{
		39,
		1,
	}

	f := func(i input) int {
		return matrixScore(i.grid)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestGetMaximumGold(t *testing.T) {
	type input struct {
		grid [][]int
	}
	inputs := []input{
		{
			[][]int{
				{0, 6, 0},
				{5, 8, 7},
				{0, 9, 0},
			},
		},
		{
			[][]int{
				{1, 0, 7},
				{2, 0, 6},
				{3, 4, 5},
				{0, 3, 0},
				{9, 0, 20},
			},
		},
	}
	expected_outputs := []int{
		24,
		28,
	}

	f := func(i input) int {
		return getMaximumGold(i.grid)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMaximumSafenessFactor(t *testing.T) {
	type input struct {
		grid [][]int
	}
	inputs := []input{
		{
			[][]int{
				{1, 0, 0},
				{0, 0, 0},
				{0, 0, 1},
			},
		},
		{
			[][]int{
				{0, 0, 1},
				{0, 0, 0},
				{0, 0, 0},
			},
		},
		{
			[][]int{
				{0, 0, 0, 1},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{1, 0, 0, 0},
			},
		},
		{
			[][]int{
				{1},
			},
		},
		{
			[][]int{
				{0, 1, 1},
				{0, 1, 1},
				{1, 1, 1},
			},
		},
		{
			[][]int{
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
				{0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1},
				{0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			},
		},
	}

	expected_outputs := []int{
		0,
		2,
		2,
		0,
		0,
		3,
	}

	f := func(i input) int {
		return maximumSafenessFactor(i.grid)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestRemoveLeafNodes(t *testing.T) {
	type input struct {
		root   *binary_tree.TreeNode
		target int
	}
	inputs := []input{
		{
			binary_tree.NewTree([]int{1, 2, 3, 2, binary_tree.NULL, 2, 4}),
			2,
		},
		{
			binary_tree.NewTree([]int{1, 3, 3, 3, 2}),
			3,
		},
		{
			binary_tree.NewTree([]int{1, 2, binary_tree.NULL, 2, binary_tree.NULL, 2}),
			2,
		},
	}

	expected_outputs := []*binary_tree.TreeNode{
		binary_tree.NewTree([]int{1, binary_tree.NULL, 3, binary_tree.NULL, 4}),
		binary_tree.NewTree([]int{1, 3, binary_tree.NULL, binary_tree.NULL, 2}),
		binary_tree.NewTree([]int{1}),
	}

	f := func(i input) *binary_tree.TreeNode {
		return removeLeafNodes(i.root, i.target)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMinCameraCover(t *testing.T) {
	type input struct {
		root *binary_tree.TreeNode
	}

	inputs := []input{
		{binary_tree.NewTree([]int{0, 0, binary_tree.NULL, 0, 0})},
		{binary_tree.NewTree([]int{0, 0, binary_tree.NULL, 0, binary_tree.NULL, 0, binary_tree.NULL, binary_tree.NULL, 0})},
		{binary_tree.NewTree([]int{0, 0, 0, binary_tree.NULL, 0, binary_tree.NULL, 0})},
		{binary_tree.NewTree([]int{0, 0, binary_tree.NULL, binary_tree.NULL, 0, 0, binary_tree.NULL, binary_tree.NULL, 0, 0})},
	}

	expected_outputs := []int{
		1,
		2,
		2,
		2,
	}

	f := func(i input) int {
		return minCameraCover(i.root)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMaximumValueSum(t *testing.T) {
	type input struct {
		nums  []int
		k     int
		edges [][]int
	}
	inputs := []input{
		{
			[]int{1, 2, 1},
			3,
			[][]int{{0, 1}, {0, 2}},
		},
		{
			[]int{2, 3},
			7,
			[][]int{{0, 1}},
		},
		{
			[]int{7, 7, 7, 7, 7, 7},
			3,
			[][]int{{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}},
		},
		{
			[]int{24, 78, 1, 97, 44},
			6,
			[][]int{{0, 2}, {1, 2}, {4, 2}, {3, 4}},
		},
	}

	expected_outputs := []int64{
		6,
		9,
		42,
		260,
	}

	f := func(i input) int64 {
		return maximumValueSum(i.nums, i.k, i.edges)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestPlacedCoins(t *testing.T) {
	type input struct {
		edges [][]int
		cost  []int
	}

	inputs := []input{
		{
			[][]int{
				{0, 1},
				{0, 2},
				{0, 3},
				{0, 4},
				{0, 5},
			},
			[]int{
				1,
				2,
				3,
				4,
				5,
				6,
			},
		},
		{
			[][]int{
				{0, 1},
				{0, 2},
				{1, 3},
				{1, 4},
				{1, 5},
				{2, 6},
				{2, 7},
				{2, 8},
			},
			[]int{
				1,
				4,
				2,
				3,
				5,
				7,
				8,
				-4,
				2,
			},
		},
		{
			[][]int{
				{0, 1},
				{0, 2},
			},
			[]int{
				1,
				2,
				-2,
			},
		},
	}

	expected_outputs := [][]int64{
		{120, 1, 1, 1, 1, 1},
		{280, 140, 32, 1, 1, 1, 1, 1, 1},
		{0, 1, 1},
	}

	f := func(i input) []int64 {
		return placedCoins(i.edges, i.cost)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestCollectTheCoins(t *testing.T) {
	type input struct {
		coins []int
		edges [][]int
	}

	inputs := []input{
		{
			[]int{1, 0, 0, 0, 0, 1},
			[][]int{
				{0, 1},
				{1, 2},
				{2, 3},
				{3, 4},
				{4, 5},
			},
		},
		{
			[]int{0, 0, 0, 1, 1, 0, 0, 1},
			[][]int{
				{0, 1},
				{0, 2},
				{1, 3},
				{1, 4},
				{2, 5},
				{5, 6},
				{5, 7},
			},
		},
		{
			[]int{1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0},
			[][]int{
				{0, 1},
				{1, 2},
				{1, 3},
				{3, 4},
				{3, 5},
				{4, 6},
				{2, 7},
				{7, 8},
				{3, 9},
				{8, 10},
				{8, 11},
				{6, 12},
				{7, 13},
				{11, 14},
				{10, 15},
			},
		},
	}

	expected_outputs := []int{
		2,
		2,
		4,
	}

	f := func(i input) int {
		return collectTheCoins(i.coins, i.edges)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestBeautifulSubsets(t *testing.T) {
	type input struct {
		nums []int
		k    int
	}

	inputs := []input{
		{
			[]int{2, 4, 6},
			2,
		},
		{
			[]int{1},
			1,
		},
		{
			[]int{10, 4, 5, 7, 2, 1},
			3,
		},
	}

	expected_outputs := []int{
		4,
		1,
		23,
	}

	f := func(i input) int {
		return beautifulSubsets(i.nums, i.k)
	}

	fast_f := func(i input) int {
		return countBeautifulSubsets(i.nums, i.k)
	}

	testResults(t, f, inputs, expected_outputs)
	testResults(t, fast_f, inputs, expected_outputs)
}

func TestMaxScoreWords(t *testing.T) {
	type input struct {
		words   []string
		letters []byte
		score   []int
	}

	inputs := []input{
		{
			[]string{"dog", "cat", "dad", "good"},
			[]byte{'a', 'a', 'c', 'd', 'd', 'd', 'g', 'o', 'o'},
			[]int{1, 0, 9, 5, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			[]string{"xxxz", "ax", "bx", "cx"},
			[]byte{'z', 'a', 'b', 'c', 'x', 'x', 'x'},
			[]int{4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 10},
		},
		{
			[]string{"leetcode"},
			[]byte{'l', 'e', 't', 'c', 'o', 'd'},
			[]int{0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
		},
	}

	expected_outputs := []int{
		23, // "dad" and "good"
		27, // "ax", "bx", and "cx"
		0,  // No words can be formed
	}

	f := func(i input) int {
		return maxScoreWords(i.words, i.letters, i.score)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestCountArrangements(t *testing.T) {
	type input struct {
		n int
	}

	inputs := []input{
		{2},
		{1},
	}

	expected_outputs := []int{
		2,
		1,
	}

	f := func(i input) int {
		return countArrangement(i.n)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestCheckRecord(t *testing.T) {
	type input struct {
		n int
	}

	inputs := []input{
		{3},
		{2},
		{1},
		{10101},
	}

	expected_outputs := []int{
		19,
		8,
		3,
		183236316,
	}

	f := func(i input) int {
		return checkRecord(i.n)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestRemoveBoxes(t *testing.T) {
	type input struct {
		boxes []int
	}
	inputs := []input{
		{[]int{1, 3, 2, 2, 2, 3, 4, 3, 1}},
		{[]int{1, 1, 1}},
		{[]int{1}},
		{[]int{1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1}},
	}

	expected_outputs := []int{
		23,
		9,
		1,
		2758,
	}

	f := func(i input) int {
		return removeBoxes(i.boxes)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestVerticalTraversal(t *testing.T) {
	type input struct {
		root *binary_tree.TreeNode
	}
	inputs := []input{
		{binary_tree.NewTree([]int{3, 9, 20, binary_tree.NULL, binary_tree.NULL, 15, 7})},
		{binary_tree.NewTree([]int{1, 2, 3, 4, 5, 6, 7})},
		{binary_tree.NewTree([]int{1, 2, 3, 4, 6, 5, 7})},
		{binary_tree.NewTree([]int{1, 2, 3, 4, 6, 5, 7})},
	}

	expected_outputs := [][][]int{
		{{9}, {3, 15}, {20}, {7}},
		{{4}, {2}, {1, 5, 6}, {3}, {7}},
		{{4}, {2}, {1, 5, 6}, {3}, {7}},
		{{4}, {2}, {1, 5, 6}, {3}, {7}},
	}

	f := func(i input) [][]int {
		return verticalTraversal(i.root)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestNumSteps(t *testing.T) {
	type input struct {
		s string
	}
	inputs := []input{
		{"1101"},
		{"10"},
		{"1"},
		{"111"},
	}

	expected_outputs := []int{
		6,
		1,
		0,
		4,
	}

	f := func(i input) int {
		return numSteps(i.s)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMaxRotateFunction(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{4, 3, 2, 6}},
		{[]int{100}},
	}

	expected_outputs := []int{
		26,
		0,
	}

	f := func(i input) int {
		return maxRotateFunction(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestEraseOverlapIntervals(t *testing.T) {
	type input struct {
		intervals [][]int
	}
	inputs := []input{
		{[][]int{{1, 2}, {2, 3}, {3, 4}, {1, 3}}},
		{[][]int{{1, 2}, {1, 2}, {1, 2}}},
		{[][]int{{1, 2}, {2, 3}}},
	}

	expected_outputs := []int{
		1,
		2,
		0,
	}

	f := func(i input) int {
		return eraseOverlapIntervals(i.intervals)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestFindMinStep(t *testing.T) {
	type input struct {
		board string
		hand  string
	}
	inputs := []input{
		{"WRRBBW", "RB"},
		{"WWRRBBWW", "WRBRW"},
		{"G", "GGGGG"},
		{"WR", "WWRR"},
		{"RRGGBBYYWWRRGGBB", "RGBYW"},
		{"RRWWRRBBRR", "WB"},
		{"RRYGGYYRRYYGGYRR", "GGBBB"},
	}

	expected_outputs := []int{
		-1,
		2,
		2,
		4,
		-1,
		2,
		5,
	}

	f := func(i input) int {
		return findMinStep(i.board, i.hand)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestEvaluate(t *testing.T) {
	type input struct {
		expression string
	}
	inputs := []input{
		{"(let x 2 (mult x (let x 3 y 4 (add x y))))"},
		{"(let x 3 x 2 x)"},
		{"(let x 1 y 2 x (add x y) (add x y))"},
		{"(let x 2 (add (let x 3 (let x 4 x)) x))"},
		{"(let a1 3 b2 (add a1 1) b2)"},
		{"(let x 7 -12)"},
		{"(let a (add 1 2) b (mult a 3) c 4 d (add a b) (mult d d))"},
	}
	expected_outputs := []int{
		14,
		2,
		5,
		6,
		4,
		-12,
		144,
	}

	f := func(i input) int {
		return evaluate(i.expression)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestCountOfAtoms(t *testing.T) {
	type input struct {
		formula string
	}
	inputs := []input{
		{"H2O"},
		{"Mg(OH)2"},
		{"K4(ON(SO3)2)2"},
		{"Be32"},
		{"Mg(H2O)N"},
	}

	expected_outputs := []string{
		"H2O",
		"H2MgO2",
		"K4N2O14S4",
		"Be32",
		"H2MgNO",
	}

	f := func(i input) string {
		return countOfAtoms(i.formula)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestIsNStraightHand(t *testing.T) {
	type input struct {
		hand []int
		size int
	}
	inputs := []input{
		{[]int{1, 2, 3, 6, 2, 3, 4, 7, 8}, 3},
		{[]int{1, 2, 3, 4, 5}, 4},
		{[]int{9, 13, 15, 23, 22, 25, 4, 4, 29, 15, 8, 23, 12, 19, 24, 17, 18, 11, 22, 24, 17, 17, 10, 23, 21, 18, 14, 18, 7, 6, 3, 6, 19, 11, 16, 11, 12, 13, 8, 26, 17, 20, 13, 19, 22, 21, 27, 9, 20, 15, 20, 27, 8, 13, 25, 23, 22, 15, 9, 14, 20, 10, 6, 5, 14, 12, 7, 16, 21, 18, 21, 24, 23, 10, 21, 16, 18, 16, 18, 5, 20, 19, 20, 10, 14, 26, 2, 9, 19, 12, 28, 17, 5, 7, 25, 22, 16, 17, 21, 11}, 10},
		{[]int{1, 1, 2, 2, 3, 3}, 2},
	}

	expected_outputs := []bool{
		true,
		false,
		false,
		false,
	}

	f := func(i input) bool {
		return isNStraightHand(i.hand, i.size)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestNumberOfArithmeticSlices(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{1, 2, 3, 4}},
		{[]int{1}},
	}

	expected_outputs := []int{
		3,
		0,
	}

	f := func(i input) int {
		return numberOfArithmeticSlices(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestNumberOfArithmeticSubsequences(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{2, 4, 6, 8, 10}},
		{[]int{7, 7, 7, 7, 7}},
		{[]int{2, 2, 3, 4}},
	}

	expected_outputs := []int{
		7,
		16,
		2,
	}

	f := func(i input) int {
		return numberOfArithmeticSubsequences(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestSubarraysDivByK(t *testing.T) {
	type input struct {
		nums []int
		k    int
	}

	inputs := []input{
		{[]int{4, 5, 0, -2, -3, 1}, 5},
		{[]int{5}, 9},
		{[]int{1, -10, 5}, 9},
	}

	expected_outputs := []int{
		7,
		0,
		1,
	}

	f := func(i input) int {
		return subarraysDivByK(i.nums, i.k)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestCanPartition(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{1, 5, 11, 5}},
		{[]int{1, 2, 3, 5}},
	}

	expected_outputs := []bool{
		true,
		false,
	}

	f := func(i input) bool {
		return canPartition(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestGetMaxRepetitions(t *testing.T) {
	type input struct {
		s1 string
		n1 int
		s2 string
		n2 int
	}
	inputs := []input{
		{"acb", 4, "ab", 2},
		{"acb", 1, "acb", 1},
	}

	expected_outputs := []int{
		2,
		1,
	}

	f := func(i input) int {
		return getMaxRepetitions(i.s1, i.n1, i.s2, i.n2)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestFindSubstringInWraproundString(t *testing.T) {
	type input struct {
		s string
	}
	inputs := []input{
		{"a"},
		{"cac"},
		{"zab"},
		{"zaba"},
	}

	expected_outputs := []int{
		1,
		2,
		6,
		6,
	}

	f := func(i input) int {
		return findSubstringInWraproundString(i.s)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestFindMaximizedCapital(t *testing.T) {
	type input struct {
		k       int
		w       int
		profits []int
		capital []int
	}
	inputs := []input{
		{2, 0, []int{1, 2, 3}, []int{0, 1, 1}},
		{3, 0, []int{1, 2, 3}, []int{0, 1, 2}},
		{1, 2, []int{1, 2, 3}, []int{1, 1, 2}},
	}

	expected_outputs := []int{
		4,
		6,
		5,
	}

	f := func(i input) int {
		return findMaximizedCapital(i.k, i.w, i.profits, i.capital)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMinPatches(t *testing.T) {
	type input struct {
		nums []int
		n    int
	}
	inputs := []input{
		{[]int{1, 3}, 6},
		{[]int{1, 5, 10}, 20},
		{[]int{1, 2, 2}, 5},
		{[]int{1, 2, 32}, 2147483647},
	}

	expected_outputs := []int{
		1,
		2,
		0,
		28,
	}

	f := func(i input) int {
		return minPatches(i.nums, i.n)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMaxPoints(t *testing.T) {
	type input struct {
		points [][]int
	}
	inputs := []input{
		{[][]int{{1, 1}, {2, 2}, {3, 3}}},
		{[][]int{{1, 1}, {3, 2}, {5, 3}, {4, 1}, {2, 3}, {1, 4}}},
		{[][]int{{0, 1}, {0, 0}}},
	}

	expected_outputs := []int{
		3,
		4,
		2,
	}

	f := func(i input) int {
		return maxPoints(i.points)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestNumSubmatrixSumTarget(t *testing.T) {
	type input struct {
		matrix [][]int
		target int
	}
	inputs := []input{
		{[][]int{{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}, 0},
		{[][]int{{1, -1}, {-1, 1}}, 0},
		{[][]int{{904}}, 0},
		{[][]int{{0, 0, 0, 1, 1}, {1, 1, 1, 0, 1}, {1, 1, 1, 1, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 1, 1}}, 0},
		{[][]int{{0, 1, 0, 0, 1}, {0, 0, 1, 1, 1}, {1, 1, 1, 0, 1}, {1, 1, 0, 1, 1}, {0, 1, 1, 0, 0}}, 1},
	}

	expected_outputs := []int{
		4,
		5,
		0,
		28,
		47,
	}

	f := func(i input) int {
		return numSubmatrixSumTarget(i.matrix, i.target)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestBagOfTokensScore(t *testing.T) {
	type input struct {
		tokens []int
		power  int
	}
	inputs := []input{
		{[]int{100}, 50},
		{[]int{200, 100}, 150},
		{[]int{100, 200, 300, 400}, 200},
	}

	expected_outputs := []int{
		0,
		1,
		2,
	}

	f := func(i input) int {
		return bagOfTokensScore(i.tokens, i.power)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestLongestPalindrome(t *testing.T) {
	type input struct {
		s string
	}
	inputs := []input{
		{"babad"},
		{"cbbd"},
	}

	expected_outputs := []string{
		"bab",
		"bb",
	}

	f := func(i input) string {
		return longestPalindrome(i.s)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestIsMatch(t *testing.T) {
	type input struct {
		s string
		p string
	}
	inputs := []input{
		{"aa", "a"},
		{"aa", "a*"},
		{"ab", ".*"},
		{"aab", "c*a*b"},
	}

	expected_outputs := []bool{
		false,
		true,
		true,
		true,
	}

	f := func(i input) bool {
		return isMatch(i.s, i.p)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestGenerateParentheses(t *testing.T) {
	type input struct {
		n int
	}
	inputs := []input{
		{3},
		{1},
	}

	expected_outputs := [][]string{
		{"((()))", "(()())", "(())()", "()(())", "()()()"},
		{"()"},
	}

	f := func(i input) []string {
		parens := generateParenthesis(i.n)
		sort.SliceStable(parens, func(i, j int) bool {
			return parens[i] < parens[j]
		})
		return parens
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestLongestValidParentheses(t *testing.T) {
	type input struct {
		s string
	}
	inputs := []input{
		{"(()"},
		{")()())"},
		{""},
		{"()(())"},
		{"(()()"},
		{"()(()"},
		{"((()))())"},
	}

	expected_outputs := []int{
		2,
		4,
		0,
		6,
		4,
		2,
		8,
	}

	f := func(i input) int {
		return longestValidParentheses(i.s)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestTrap(t *testing.T) {
	type input struct {
		height []int
	}
	inputs := []input{
		{[]int{0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1}},
		{[]int{4, 2, 0, 3, 2, 5}},
	}

	expected_outputs := []int{
		6,
		9,
	}

	f := func(i input) int {
		return trap(i.height)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestCanJump(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{2, 3, 1, 1, 4}},
		{[]int{3, 2, 1, 0, 4}},
		{[]int{2, 5, 0, 0}},
	}

	expected_outputs := []bool{
		true,
		false,
		true,
	}

	f := func(i input) bool {
		return canJump(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestJump(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{2, 3, 1, 1, 4}},
		{[]int{2, 3, 0, 1, 4}},
	}

	expected_outputs := []int{
		2,
		2,
	}

	f := func(i input) int {
		return jump(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMaxSubArray(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{-2, 1, -3, 4, -1, 2, 1, -5, 4}},
		{[]int{1}},
		{[]int{5, 4, -1, 7, 8}},
	}

	expected_outputs := []int{
		6,
		1,
		23,
	}

	f := func(i input) int {
		return maxSubArray(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestTwoSum(t *testing.T) {
	type input struct {
		nums   []int
		target int
	}
	inputs := []input{
		{[]int{2, 7, 11, 15}, 9},
		{[]int{3, 2, 4}, 6},
		{[]int{3, 3}, 6},
	}

	expected_outputs := [][]int{
		{0, 1},
		{1, 2},
		{0, 1},
	}

	f := func(i input) []int {
		values := twoSum(i.nums, i.target)
		sort.SliceStable(values, func(i, j int) bool {
			return values[i] < values[j]
		})
		return values
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestBalanceBST(t *testing.T) {
	type input struct {
		root *binary_tree.TreeNode
	}
	inputs := []input{
		{binary_tree.NewTree([]int{1, binary_tree.NULL, 2, binary_tree.NULL, 3, binary_tree.NULL, 4, binary_tree.NULL, binary_tree.NULL})},
		{binary_tree.NewTree([]int{2, 1, 3})},
	}

	expected_outputs := []*binary_tree.TreeNode{
		binary_tree.NewTree([]int{3, 2, 4, 1}),
		binary_tree.NewTree([]int{2, 1, 3}),
	}

	f := func(i input) *binary_tree.TreeNode {
		return balanceBST(i.root)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestUniquePaths(t *testing.T) {
	type input struct {
		m int
		n int
	}
	inputs := []input{
		{3, 7},
		{3, 2},
	}

	expected_outputs := []int{
		28,
		3,
	}

	f := func(i input) int {
		return uniquePaths(i.m, i.n)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestUniquePathsWithObstacles(t *testing.T) {
	type input struct {
		obstacleGrid [][]int
	}
	inputs := []input{
		{
			[][]int{
				{0, 0, 0},
				{0, 1, 0},
				{0, 0, 0},
			},
		},
		{
			[][]int{
				{0, 1},
				{0, 0},
			},
		},
	}

	expected_outputs := []int{
		2,
		1,
	}

	f := func(i input) int {
		return uniquePathsWithObstacles(i.obstacleGrid)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMinPathSum(t *testing.T) {
	type input struct {
		grid [][]int
	}
	inputs := []input{
		{
			[][]int{
				{1, 3, 1},
				{1, 5, 1},
				{4, 2, 1},
			},
		},
		{
			[][]int{
				{1, 2, 3},
				{4, 5, 6},
			},
		},
	}

	expected_outputs := []int{
		7,
		12,
	}

	f := func(i input) int {
		return minPathSum(i.grid)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestClimbStairs(t *testing.T) {
	type input struct {
		n int
	}
	inputs := []input{
		{2},
		{3},
	}

	expected_outputs := []int{
		2,
		3,
	}

	f := func(i input) int {
		return climbStairs(i.n)
	}
	testResults(t, f, inputs, expected_outputs)
}

func TestMinDistance(t *testing.T) {
	type input struct {
		word1 string
		word2 string
	}
	inputs := []input{
		{"horse", "ros"},
		{"intention", "execution"},
	}

	expected_outputs := []int{
		3,
		5,
	}

	f := func(i input) int {
		return minDistance(i.word1, i.word2)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestNextGreaterElement(t *testing.T) {
	type input struct {
		nums1 []int
		nums2 []int
	}
	inputs := []input{
		{[]int{4, 1, 2}, []int{1, 3, 4, 2}},
		{[]int{2, 4}, []int{1, 2, 3, 4}},
	}

	expected_outputs := [][]int{
		{-1, 3, -1},
		{3, -1},
	}

	f := func(i input) []int {
		return nextGreaterElement(i.nums1, i.nums2)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestNextGreaterElements(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{1, 2, 1}},
		{[]int{1, 2, 3, 4, 3}},
	}

	expected_outputs := [][]int{
		{2, -1, 2},
		{2, 3, 4, -1, 4},
	}

	f := func(i input) []int {
		return nextGreaterElements(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestNextGreaterElementIII(t *testing.T) {
	type input struct {
		n int
	}
	inputs := []input{
		{12},
		{21},
		{230241},
		{2147483486},
		{2147483476},
	}

	expected_outputs := []int{
		21,
		-1,
		230412,
		-1,
		2147483647,
	}

	f := func(i input) int {
		return nextGreaterElementIII(i.n)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestLargestRectangleArea(t *testing.T) {
	type input struct {
		heights []int
	}
	inputs := []input{
		{[]int{2, 1, 5, 6, 2, 3}},
		{[]int{2, 4}},
	}

	expected_outputs := []int{
		10,
		4,
	}

	f := func(i input) int {
		return largestRectangleArea(i.heights)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMaximalRectangle(t *testing.T) {
	type input struct {
		matrix [][]byte
	}
	inputs := []input{
		{
			[][]byte{
				{'1', '0', '1', '0', '0'},
				{'1', '0', '1', '1', '1'},
				{'1', '1', '1', '1', '1'},
				{'1', '0', '0', '1', '0'},
			},
		},
		{
			[][]byte{
				{'0'},
			},
		},
		{
			[][]byte{
				{'1'},
			},
		},
	}

	expected_outputs := []int{
		6,
		0,
		1,
	}

	f := func(i input) int {
		return maximalRectangle(i.matrix)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMaxNumEdgesToRemove(t *testing.T) {
	type input struct {
		n     int
		edges [][]int
	}
	inputs := []input{
		{4, [][]int{
			{3, 1, 2},
			{3, 2, 3},
			{1, 1, 3},
			{1, 2, 4},
			{1, 1, 2},
			{2, 3, 4},
		},
		},
		{4, [][]int{
			{3, 1, 2},
			{3, 2, 3},
			{1, 1, 4},
			{2, 1, 4},
		},
		},
		{4, [][]int{
			{3, 2, 3},
			{1, 1, 2},
			{2, 3, 4},
		},
		},
		{12, [][]int{{3, 1, 2}, {2, 2, 3}, {3, 1, 4}, {2, 3, 5}, {1, 2, 6}, {2, 4, 7}, {3, 3, 8}, {3, 2, 9}, {2, 1, 10}, {2, 1, 11}, {1, 11, 12}, {1, 10, 11}, {2, 5, 9}, {2, 7, 10}, {2, 4, 12}, {3, 9, 10}, {1, 6, 9}, {2, 10, 12}, {1, 2, 5}, {3, 5, 6}, {1, 7, 11}, {1, 8, 9}, {1, 1, 11}, {3, 4, 5}, {1, 5, 9}, {2, 4, 9}, {1, 8, 11}, {3, 6, 8}, {1, 8, 10}, {2, 2, 4}, {2, 3, 8}, {3, 2, 6}, {3, 10, 11}, {2, 3, 11}, {3, 5, 9}, {3, 3, 5}, {2, 6, 11}, {3, 2, 7}, {1, 5, 11}, {1, 1, 5}, {2, 9, 10}, {1, 6, 7}, {3, 2, 3}, {2, 8, 9}, {3, 2, 8}}},
	}

	expected_outputs := []int{
		2,
		0,
		-1,
		33,
	}

	f := func(i input) int {
		return maxNumEdgesToRemove(i.n, i.edges)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestIsScramble(t *testing.T) {
	type input struct {
		s1 string
		s2 string
	}
	inputs := []input{
		{"great", "rgeat"},
		{"abcde", "caebd"},
		{"a", "a"},
	}

	expected_outputs := []bool{
		true,
		false,
		true,
	}

	f := func(i input) bool {
		return isScramble(i.s1, i.s2)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestNumDecodings(t *testing.T) {
	type input struct {
		s string
	}
	inputs := []input{
		{"12"},
		{"226"},
		{"06"},
	}

	expected_outputs := []int{
		2,
		3,
		0,
	}

	f := func(i input) int {
		return numDecodings(i.s)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMinDifference(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{5, 3, 2, 4}},
		{[]int{1, 5, 0, 10, 14}},
		{[]int{3, 100, 20}},
	}

	expected_outputs := []int{
		0,
		1,
		0,
	}

	f := func(i input) int {
		return minDifference(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMergeNodes(t *testing.T) {
	type input struct {
		head *list_node.ListNode
	}
	inputs := []input{
		{list_node.NewList([]int{0, 3, 1, 0, 4, 5, 2, 0})},
		{list_node.NewList([]int{0, 1, 0, 3, 0, 2, 2, 0})},
	}

	expected_outputs := []*list_node.ListNode{
		list_node.NewList([]int{4, 11}),
		list_node.NewList([]int{1, 3, 4}),
	}

	f := func(i input) *list_node.ListNode {
		return mergeNodes(i.head)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestGenerateTrees(t *testing.T) {
	type input struct {
		n int
	}
	inputs := []input{
		{3},
		{1},
	}

	expected_outputs := [][]*binary_tree.TreeNode{
		{
			binary_tree.NewTree([]int{1, binary_tree.NULL, 2, binary_tree.NULL, 3}),
			binary_tree.NewTree([]int{1, binary_tree.NULL, 3, 2}),
			binary_tree.NewTree([]int{2, 1, 3}),
			binary_tree.NewTree([]int{3, 1, binary_tree.NULL, binary_tree.NULL, 2}),
			binary_tree.NewTree([]int{3, 2, binary_tree.NULL, 1, binary_tree.NULL}),
		},
		{
			binary_tree.NewTree([]int{1}),
		},
	}

	f := func(i input) []*binary_tree.TreeNode {
		return generateTrees(i.n)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestNumTrees(t *testing.T) {
	type input struct {
		n int
	}
	inputs := []input{
		{3},
		{1},
	}

	expected_outputs := []int{
		5,
		1,
	}

	f := func(i input) int {
		return numTrees(i.n)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestIsInterleave(t *testing.T) {
	type input struct {
		s1 string
		s2 string
		s3 string
	}
	inputs := []input{
		{"aabcc", "dbbca", "aadbbcbcac"},
		{"aabcc", "dbbca", "aadbbbaccc"},
		{"", "", ""},
	}

	expected_outputs := []bool{
		true,
		false,
		true,
	}

	f := func(i input) bool {
		return isInterleave(i.s1, i.s2, i.s3)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestNumDistinct(t *testing.T) {
	type input struct {
		s string
		t string
	}
	inputs := []input{
		{"rabbbit", "rabbit"},
		{"babgbag", "bag"},
	}

	expected_outputs := []int{
		3,
		5,
	}

	f := func(i input) int {
		return numDistinct(i.s, i.t)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestGenerate(t *testing.T) {
	type input struct {
		numRows int
	}
	inputs := []input{
		{5},
		{1},
	}

	expected_outputs := [][][]int{
		{
			{1},{1,1},{1,2,1},{1,3,3,1},{1,4,6,4,1},
		},
		{
			{1},
		},
	}

	f := func(i input) [][]int {
		return generate(i.numRows)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestGetRow(t *testing.T) {
	type input struct {
		rowIndex int
	}
	inputs := []input{
		{3},
		{0},
		{1},
	}

	expected_outputs := [][]int{
		{1,3,3,1},
		{1},
		{1,1},
	}

	f := func(i input) []int {
		return getRow(i.rowIndex)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMinimumTotal(t *testing.T) {
	type input struct {
		triangle [][]int
	}
	inputs := []input{
		{
			[][]int{
				{2},{3,4},{6,5,7},{4,1,8,3},
			},
		},
		{
			[][]int{
				{-10},
			},
		},
	}

	expected_outputs := []int{
		11,
		-10,
	}

	f := func(i input) int {
		return minimumTotal(i.triangle)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestTopStudents(t *testing.T) {
	type input struct {
		positive_feedback []string
		negative_feedback []string
		report []string
		student_id []int
		k int
	}
	inputs := []input{
		{
			[]string{"smart","brilliant","studious"}, 
			[]string{"not"},
			[]string{"this student is studious", "the student is smart"},
			[]int{1,2},
			2,
		},
		{
			[]string{"smart","brilliant","studious"}, 
			[]string{"not"},
			[]string{"this student is not studious", "the student is smart"},
			[]int{1,2},
			2,
		},
	}

	expected_outputs := [][]int{
		{1,2},
		{2,1},
	}

	f := func(i input) []int{
		return topStudents(i.positive_feedback, i.negative_feedback, i.report, i.student_id, i.k)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMaxProfit(t *testing.T) {
	type input struct {
		prices []int
	}
	inputs := []input{
		{[]int{7,1,5,3,6,4}},
		{[]int{7,6,4,3,1}},
	}

	expected_outputs := []int{
		5,
		0,
	}

	f := func(i input) int {
		return maxProfit(i.prices)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMaxProfit2(t *testing.T) {
	type input struct {
		prices []int
	}
	inputs := []input{
		{[]int{7,1,5,3,6,4}},
		{[]int{1,2,3,4,5}},
		{[]int{7,6,4,3,1}},
	}

	expected_outputs := []int{
		7,
		4,
		0,
	}

	f := func(i input) int {
		return maxProfit2(i.prices)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMaxProfit3(t *testing.T) {
	type input struct {
		prices []int
	}
	inputs := []input{
		{[]int{3,3,5,0,0,3,1,4}},
		{[]int{1,2,3,4,5}},
		{[]int{7,6,4,3,1}},
	}

	expected_outputs := []int{
		6,
		4,
		0,
	}

	f := func(i input) int {
		return maxProfit3(i.prices)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMaxProfit4(t *testing.T) {
	type input struct {
		k int
		prices []int
	}
	inputs := []input{
		{2, []int{2,4,1}},
		{2, []int{3,2,6,5,0,3}},
	}

	expected_outputs := []int{
		2,
		7,
	}

	f := func(i input) int {
		return maxProfit4(i.k, i.prices)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMaxProfit5(t *testing.T) {
	type input struct {
		prices []int
	}
	inputs := []input{
		{[]int{1,2,3,0,2}},
		{[]int{1}},
	}

	expected_outputs := []int{
		3,
		0,
	}

	f := func(i input) int {
		return maxProfit5(i.prices)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestAverageWaitingTime(t *testing.T) {
	type input struct {
		customers [][]int
	}
	inputs := []input{
		{[][]int{
			{1,2},
			{2,5},
			{4,3},
		}},
		{[][]int{
			{5,2},
			{5,4},
			{10,3},
			{20,1},
		}},
	}
	
	expected_outputs := []float64{
		5.0,
		3.25,
	}

	f := func(i input) float64 {
		return averageWaitingTime(i.customers)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMaxPathSum(t *testing.T) {
	type input struct {
		root *binary_tree.TreeNode
	}
	inputs := []input{
		{binary_tree.NewTree([]int{1,2,3})},
		{binary_tree.NewTree([]int{-10,9,20,binary_tree.NULL, binary_tree.NULL,15,7})},
	}

	expected_outputs := []int{
		6,
		42,
	}

	f := func(i input) int {
		return maxPathSum(i.root)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestPartition(t *testing.T) {
	type input struct {
		s string
	}
	inputs := []input{
		{"aab"},
		{"a"},
		{"cdcef"},
		{"cccef"},
	}

	expected_outputs := [][][]string{
		{
			{"a", "a", "b"},
			{"aa", "b"},
		},
		{
			{"a"},
		},
		{
			{"c", "d", "c", "e", "f"},
			{"cdc", "e", "f"},
		},
		{
			{"c", "c", "c", "e", "f"},
			{"cc", "c", "e", "f"},
			{"c", "cc", "e", "f"},
			{"ccc", "e", "f"},
		},
	}

	f := func(i input) [][]string{
		return partition(i.s)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMinCut(t *testing.T) {
	type input struct {
		s string
	}
	inputs := []input{
		{"aab"},
		{"a"},
		{"ab"},
		{"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
	}

	expected_outputs := []int{
		1,
		0,
		1,
		0,
	}

	f := func(i input) int {
		return minCut(i.s)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestWordBreak(t *testing.T) {
	type input struct {
		s string
		wordDict []string
	}
	inputs := []input{
		{"leetcode", []string{"leet","code"}},
		{"applepenapple", []string{"apple","pen"}},
		{"catsandog", []string{"cats", "dog", "sand","and","cat"}},
	}

	expected_outputs := []bool{
		true,
		true,
		false,
	}

	f := func(i input) bool {
		return wordBreak(i.s, i.wordDict)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestWordBreak2(t *testing.T) {
	type input struct {
		s string
		wordDict []string
	}
	inputs := []input{
		{"catsanddog", []string{"cat","cats","and","sand","dog"}},
		{"pineapplepenapple", []string{"apple","pen","applepen","pine","pineapple"}},
		{"catsandog", []string{"cats", "dog", "sand","and","cat"}},
	}

	expected_outputs := [][]string{
		{"cat sand dog","cats and dog"},
		{"pine apple pen apple","pine applepen apple","pineapple pen apple"},
		{},
	}

	f := func(i input) []string {
		return wordBreak2(i.s, i.wordDict)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestReverseParentheses(t *testing.T) {
	type input struct {
		s string
	}
	inputs := []input{
		{"(abcd)"},
		{"(u(love)i)"},
		{"(ed(et(oc))el)"},
		{"a(bcdefghijkl(mno)p)q"},
		{"ta()usw((((a))))"},
	}

	expected_outputs := []string{
		"dcba",
		"iloveu",
		"leetcode",
		"apmnolkjihgfedcbq",
		"tauswa",
	}

	f := func(i input) string {
		return reverseParentheses(i.s)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMaximumGain(t *testing.T) {
	type input struct {
		s string
		x int
		y int
	}
	inputs := []input{
		{"cdbcbbaaabab", 4, 5},
		{"aabbaaxybbaabb", 5, 4},
	}

	expected_outputs := []int{
		19,
		20,
	}

	f := func(i input) int {
		return maximumGain(i.s, i.x, i.y)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestSurvivedRobotsHealths(t *testing.T) {
	type input struct {
		positions []int
		healths []int
		directions string
	}
	inputs := []input{
		{[]int{5,4,3,2,1}, []int{2,17,9,15,10}, "RRRRR"},
		{[]int{3,5,2,6}, []int{10,10,15,12}, "RLRL"},
		{[]int{1,2,5,6}, []int{10,10,11,11}, "RLRL"},
	}

	expected_outputs := [][]int{
		{2,17,9,15,10},
		{14},
		{},
	}

	f := func(i input) []int{
		return survivedRobotsHealths(i.positions, i.healths, i.directions)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestMaxProduct(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{2,3,-2,4}},
		{[]int{-2,0,-1}},
		{[]int{0,10,10,10,10,10,10,10,10,10,-10,10,10,10,10,10,10,10,10,10,0}},
	}

	expected_outputs := []int{
		6,
		0,
		1000000000,
	}

	f := func(i input) int {
		return maxProduct(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestCalculateMinimumHP(t *testing.T) {
	type input struct {
		dungeon [][]int
	}
	inputs := []input{
		{[][]int{{-2,-3,3},{-5,-10,1},{10,30,-5}}},
		{[][]int{{0}}},
		{[][]int{{-3,5}}},
		{[][]int{{0,-5},{0,0}}},
	}

	expected_outputs := []int{
		7,
		1,
		4,
		1,
	}

	f := func(i input) int {
		return calculateMinimumHP(i.dungeon)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestGetDirections(t *testing.T) {
	type input struct {
		root *binary_tree.TreeNode
		startValue int
		destValue int
	}
	inputs := []input{
		{binary_tree.NewTree([]int{5,1,2,3,binary_tree.NULL,6,4}), 3, 6},
		{binary_tree.NewTree([]int{2,1}), 2, 1},
	}

	expected_outputs := []string{
		"UURL",
		"L",
	}

	f := func(i input) string {
		return getDirections(i.root, i.startValue, i.destValue)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestDelNodes(t *testing.T) {
	type input struct {
		root *binary_tree.TreeNode
		to_delete []int
	}
	inputs := []input{
		{binary_tree.NewTree([]int{1,2,3,4,5,6,7}), []int{3,5}},
		{binary_tree.NewTree([]int{1,2,4,binary_tree.NULL,3}), []int{3}},
	}

	expected_outputs := [][]*binary_tree.TreeNode{
		{binary_tree.NewTree([]int{1,2,binary_tree.NULL,4}), binary_tree.NewTree([]int{6}), binary_tree.NewTree([]int{7})},
		{binary_tree.NewTree([]int{1,2,4})},
	}

	f := func(i input) []*binary_tree.TreeNode {
		return delNodes(i.root, i.to_delete)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestCountPairs(t *testing.T) {
	type input struct {
		root *binary_tree.TreeNode
		distance int
	}
	inputs := []input{
		{binary_tree.NewTree([]int{1,2,3,binary_tree.NULL,4}), 3},
		{binary_tree.NewTree([]int{1,2,3,4,5,6,7}), 3},
		{binary_tree.NewTree([]int{7,1,4,6,binary_tree.NULL,5,3,binary_tree.NULL,binary_tree.NULL,binary_tree.NULL,binary_tree.NULL,binary_tree.NULL,2}), 3},
		{binary_tree.NewTree([]int{1,1,1}), 2},
	}

	expected_outputs := []int{
		1,
		2,
		1,
		1,
	}

	f := func(i input) int {
		return countPairs(i.root, i.distance)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestRob(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{1,2,3,1}},
		{[]int{2,7,9,3,1}},
	}

	expected_outputs := []int{
		4,
		12,
	}

	f := func(i input) int {
		return rob(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestRob2(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{2,3,2}},
		{[]int{1,2,3,1}},
		{[]int{1,2,3}},
	}

	expected_outputs := []int{
		3,
		4,
		3,
	}

	f := func(i input) int {
		return rob2(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestRob3(t *testing.T) {
	type input struct {
		root *binary_tree.TreeNode
	}
	inputs := []input{
		{binary_tree.NewTree([]int{3,2,3,binary_tree.NULL,3,binary_tree.NULL,1})},
		{binary_tree.NewTree([]int{3,4,5,1,3,binary_tree.NULL,1})},
	}

	expected_outputs := []int{
		7,
		9,
	}

	f := func(i input) int {
		return rob3(i.root)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestMinCapability(t *testing.T) {
	type input struct {
		nums []int
		k int
	}
	inputs := []input{
		{[]int{2,3,5,9}, 2},
		{[]int{2,7,9,3,1}, 2},
	}

	expected_outputs := []int{
		5,
		2,
	}

	f := func(i input) int {
		return minCapability(i.nums, i.k)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestMaximalSquare(t *testing.T) {
	type input struct {
		matrix [][]byte
	}
	inputs := []input{
		{[][]byte{{'1','0','1','0','0'},{'1','0','1','1','1'},{'1','1','1','1','1'},{'1','0','0','1','0'}}},
		{[][]byte{{'0','1'},{'1','0'}}},
	}

	expected_outputs := []int{
		4,
		1,
	}

	f := func(i input) int {
		return maximalSquare(i.matrix)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestDiffWaysToCompute(t *testing.T) {
	type input struct {
		expression string
	}
	inputs := []input{
		{"2-1-1"},
		{"2*3-4*5"},
	}

	expected_outputs := [][]int{
		{0, 2},
		{-34,-14,-10,-10,10},
	}

	f := func(i input) []int {
		nums := diffWaysToCompute(i.expression)
		sort.SliceStable(nums, func(i, j int) bool {
			return nums[i] < nums[j]
		})
		return nums
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestIsAnagram(t *testing.T) {
	type input struct {
		s string
		t string
	}
	inputs := []input{
		{"anagram", "nagaram"},
		{"rat", "car"},
	}

	expected_outputs := []bool{
		true,
		false,
	}

	f := func(i input) bool {
		return isAnagram(i.s, i.t)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestLengthOfLIS(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{10,9,2,5,3,7,101,18}},
		{[]int{0,1,0,3,2,3}},
		{[]int{7,7,7,7,7,7,7}},
	}

	expected_outputs := []int{
		4,
		4,
		1,
	}

	f := func(i input) int {
		return lengthOfLIS(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestNthUglyNumber(t *testing.T) {
	type input struct {
		n int
	}
	inputs := []input{
		{10},
		{1},
		{11},
	}

	expected_outputs := []int{
		12,
		1,
		15,
	}

	f := func(i input) int {
		return nthUglyNumber(i.n)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestNumSquares(t *testing.T) {
	type input struct {
		n int
	}
	inputs := []input{
		{12},
		{3},
	}

	expected_outputs := []int{
		3,
		3,
	}

	f := func(i input) int {
		return numSquares(i.n)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestSortJumbled(t *testing.T) {
	type input struct {
		mapping []int
		nums []int
	}
	inputs := []input{
		{[]int{8,9,4,0,2,1,3,5,7,6}, []int{991,338,38}},
		{[]int{0,1,2,3,4,5,6,7,8,9}, []int{789,456,123}},
	}

	expected_outputs := [][]int{
		{338,38,991},
		{123,456,789},
	}

	f := func(i input) []int{
		return sortJumbled(i.mapping, i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestSortArray(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{5,2,3,1}},
		{[]int{5,1,1,2,0,0}},
	}

	expected_outputs := [][]int{
		{1,2,3,5},
		{0,0,1,1,2,5},
	}

	f := func(i input) []int {
		return sortArray(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestMaxCoins(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{3,1,5,8}},
		{[]int{1,5}},
	}

	expected_outputs := []int{
		167,
		10,
	}

	f := func(i input) int {
		return maxCoins(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestLongestIncreasingPath(t *testing.T) {
	type input struct {
		matrix [][]int
	}
	inputs := []input{
		{[][]int{
			{9,9,4},
			{6,6,8},
			{2,1,1},
		}},
		{[][]int{
			{3,4,5},
			{3,2,6},
			{2,2,1},
		}},
	}

	expected_outputs := []int{
		4,
		4,
	}

	f := func(i input) int {
		return longestIncreasingPath(i.matrix)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestFindTheCity(t *testing.T) {
	type input struct {
		n int
		edges [][]int
		distanceThreshold int
	}
	inputs := []input{
		{4, [][]int{
			{0,1,3},
			{1,2,1},
			{1,3,4},
			{2,3,1},
			},
		4,},
		{5, [][]int{
			{0,1,2},
			{0,4,8},
			{1,2,3},
			{1,4,2},
			{2,3,1},
			{3,4,1},
			},
		2,},
	}

	expected_outputs := []int{
		3,
		0,
	}

	f := func(i input) int {
		return findTheCity(i.n, i.edges, i.distanceThreshold)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestMinimumCost(t *testing.T) {
	type input struct {
		source string
		target string
		original []byte
		changed []byte
		cost []int
	}
	inputs := []input{
		{"abcd", "acbe", []byte{'a','b','c','c','e','d'}, []byte{'b','c','b','e','b','e'}, []int{2,5,5,1,2,20}},
		{"aaaa", "bbbb", []byte{'a','c'}, []byte{'c','b'}, []int{1,2}},
		{"abcd", "abce", []byte{'a'}, []byte{'e'}, []int{10000}},
		{"abadcdadac", "baddbccdac", []byte{'d','c','d','c','b','a'}, []byte{'b','b','c','a','d','d'}, []int{8,5,9,1,10,2}},
	}

	expected_outputs := []int64{
		int64(28),
		int64(12),
		int64(-1),
		int64(57),
	}

	f := func(i input) int64 {
		return minimumCost(i.source, i.target, i.original, i.changed, i.cost)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestSecondMinimum(t *testing.T) {
	type input struct {
		n int
		edges [][]int
		time int
		change int
	}
	inputs := []input{
		{5, [][]int{{1,2},{1,3},{1,4},{3,4},{4,5}}, 3, 5},
		{2, [][]int{{1,2}}, 3, 2},
		{2, [][]int{{1,2}}, 1, 2},
		{2, [][]int{{1,2}}, 2, 1},
	}

	expected_outputs := []int{
		13,
		11,
		5,
		6,
	}

	f := func(i input) int {
		return secondMinimum(i.n, i.edges, i.time, i.change)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestCountBits(t *testing.T) {
	type input struct {
		n int
	}
	inputs := []input{
		{2},
		{5},
	}

	expected_outputs := [][]int{
		{0,1,1},
		{0,1,1,2,1,2},
	}

	f := func(i input) []int {
		return countBits(i.n)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestIntegerBreak(t *testing.T) {
	type input struct {
		n int
	}
	inputs := []input{
		{2},
		{10},
	}

	expected_outputs := []int{
		1,
		36,
	}

	f := func(i input) int {
		return integerBreak(i.n)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestIsSubsequence(t *testing.T) {
	type input struct {
		s string
		t string
	}
	inputs := []input{
		{"abc","ahbgdc"},
		{"axc","ahbgdc"},
	}

	expected_outputs := []bool{
		true,
		false,
	}
	
	f := func(i input) bool {
		return isSubsequence(i.s, i.t)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestMinSwapsBinary(t *testing.T) {
	type input struct {
		nums []int
	}
	inputs := []input{
		{[]int{0,1,0,1,1,0,0}},
		{[]int{0,1,1,1,0,0,1,1,0}},
		{[]int{1,1,0,0,1}},
		{[]int{1}},
	}

	expected_outputs := []int{
		1,
		2,
		0,
		0,
	}

	f := func(i input) int {
		return minSwapsBinary(i.nums)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestCanCross(t *testing.T) {
	type input struct {
		stones []int
	}
	inputs := []input{
		{[]int{0,1,3,5,6,8,12,17}},
		{[]int{0,1,2,3,4,8,9,11}},
		{[]int{0,1,3,6,7}},
	}

	expected_outputs := []bool{
		true,
		false,
		false,
	}

	f := func(i input) bool {
		return canCross(i.stones)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestSplitArray(t *testing.T) {
	type input struct {
		nums []int
		k int
	}
	inputs := []input{
		{[]int{7,2,5,10,8}, 2},
		{[]int{1,2,3,4,5}, 2},
	}

	expected_outputs := []int{
		18,
		9,
	}

	f := func(i input) int {
		return splitArray(i.nums, i.k)
	}

	testResults(t, f, inputs, expected_outputs)
 }

 func TestMinimumPushes(t *testing.T) {
	type input struct {
		word string
	}
	inputs := []input{
		{"abcde"},
		{"xyzxyzxyzxyz"},
		{"aabbccddeeffgghhiiiiii"},
	}

	expected_outputs := []int{
		5,
		12,
		24,
	}

	f := func(i input) int {
		return minimumPushes(i.word)
	}

	testResults(t, f, inputs, expected_outputs)
 }