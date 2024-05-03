package solution

import (
	"fmt"
	"leetcode/binary_tree"
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
		{		
		},
	}

	f := func(i input) [][]int {
		return findFarmland(i.land)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestGetMoneyAmount(t *testing.T) {
	type input struct {
		n 	int
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
		deadends 	[]string
		target 		string
	}
	inputs := []input{
		{
			[]string{
				"0201","0101","0102","1212","2002",
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
				"8887","8889","8878","8898","8788","8988","7888","9888",
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
				"8430","5911","4486","7174","9772","0731","9550","3449","4437","3837","1870","5798","9583","9512","5686","5131","0736","3051","2141","2989","6368","2004","1012","8736","0363","3589","8568","6457","3467","1967","1055","6637","1951","0575","4603","2606","0710","4169","7009","6554","6128","2876","8151","4423","0727","8130","3571","4801","8968","6084","3156","3087","0594","9811","3902","4690","6468","2743","8560","9064","4231","6056","2551","8556","2541","5460","5657","1151","5123","3521","2200","9333","9685","4871","9138","5807","2191","2601","1792","3470","9096","0185","0367","6862","1757","6904","4485","7973","7201","2571","3829","0868","4632","6975","2026","3463","2341","4647","3680","3282","3761","4410","3397","3357","4038","6505","1655","3812","3558","4759","1112","8836","5348","9113","1627","3249","0537","4227","7952","8855","3592","2054","3175","6665","4088","9959","3809","7379","6949","8063","3686","8078","0925","5167","2075","4665","2628","8242","9831","1397","5547","9449","6512","6083","9682","2215","3236","2457","6211","5536","8674","2647","9752","5433","0186","5904","1526","5347","1387","3153","1353","6069","9995","9496","0003","3400","1692","6870","4445","3063","0708","3278","6961","3063","0249","0375","1763","1804","4695","6493","7573","9977","1108","0856","5631","4799","4164","0844","2600","1785","1587","4510","9012","7497","4923","2560","0338","3839","5624","1980","1514","4634","2855","7012","3626","7032","6145","5663","4395","0724","4711","1573","6904","8100","2649","3890","8110","8067","1460","0186","6098","2459","6991","9372","8539","8418","7944","0499","9276","1525","1281","8738","5054","7869","6599","8018","7530","2327","3681","5248","4291","7300","8854","2591","8744","3052","6369","3669","8501","8455","5726","1211","8793","6889","9315","0738","6805","5980","7485","2333","0140","4708","9558","9026","4349","5978","4989","5238","3217","5938","9660","5858","2118","7657","5896","3195","8997","1688","2863","9356","4208","5438","2642","4138","7466","6154","0926","2556","9574","4497","9633","0585","1390","5093","3047","0430","7482","0750","6229","8714","4765","0941","1780","6262","0925","5631","9167","0885","7713","5576","3775","9652","0733","7467","5301","9365","7978","4736","3309","6965","4703","5897","8460","9619","0572","6297","7701","7554","8669","5426","6474","5540","5038","3880","1657","7574","1108","4369","7782","9742","5301","6984","3158","2869","0599","2147","6962","9722","3597","9015","3115","9051","8269","6967","5392","4401","6579","8997","8933","9297","0151","8820","3297","6723","1755","1163","8896","7122","4859","5504","0857","4682","8177","8702","9167","9410","0130","2789","7492","5938","3012","4137","3414","2245","4292","6945","5446","6614","2977","8640","9242","7603","8349","9420","0538","4222","0599","8459","8738","4764","6717","7575","5965","9816","9975","4994","2612","0344","6450","9088","4898","6379","4127","1574","9044","0434","5928","6679","1753","8940","7563","0545","4575","6407","6213","8327","3978","9187","2996","1956","8819","9591","7802","4747","9094","0179","0806","2509","4026","4850","2495","3945","4994","5971","3401","0218","6584","7688","6138","7047","9456","0173","1406","1564","3055","8725","4835","4737","6279","5291","0145","0002","1263","9518","1251","8224","6779","4113","8680","2946","1685","2057","9520","4099","7785","1134","2152","4719","6038","1599","6750","9273","7755","3134","2345","8208","5750","5850","2019","0350","9013","6911","6095","6843","3157","9049","0801","2739","9691","3511",
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
		n 		int
		edges 	[][]int
	}
	inputs := []input{
		{
			4,
			[][]int{
				{1,0},{1,2},{1,3},
			},
		},
		{
			6,
			[][]int{
				{3,0},{3,1},{3,2},{3,4},{5,4},
			},
		},
	}

	expected_outputs := [][]int{
		{1},
		{3,4},
	}

	f := func(i input) []int {
		return findMinHeightTrees(i.n, i.edges)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestWiggleMaxLength(t *testing.T) {
	type input struct {
		nums 	[]int
	}
	inputs := []input{
		{[]int{1,7,4,9,2,5}},
		{[]int{1,17,5,10,13,15,10,5,16,8}},
		{[]int{1,2,3,4,5,6,7,8,9}},
		{[]int{3,3,3,2,5}},
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
		s 	string
		k 	int
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
		grid 	[][]int
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
				{-73,61,43,-48,-36},
				{3,30,27,57,10},
				{96,-76,84,59,-15},
				{5,-49,76,31,-7},
				{97,91,61,-46,67},
			},
		},
		{
			[][]int{
				{50,-18,-38,39,-20,-37,-61,72,22,79},
				{82,26,30,-96,-1,28,87,94,34,-89},
				{55,-50,20,76,-50,59,-58,85,83,-83},
				{39,65,-68,89,-62,-53,74,2,-70,-90},
				{1,57,-70,83,-91,-32,-13,49,-11,58},
				{-55,83,60,-12,-90,-37,-36,-27,-19,-6},
				{76,-53,78,90,70,62,-81,-94,-32,-57},
				{-32,-85,81,25,80,90,-24,10,27,-55},
				{39,54,39,34,-45,17,-2,-61,-81,85},
				{-77,65,76,92,21,68,78,-13,39,22},
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
		ring 	string
		key 	string
	}
	inputs := []input {
		{"godding", "gd"},
		{"godding", "godding"},
	}
	expected_outputs := []int {
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
		root 	*binary_tree.TreeNode
	}
	inputs := []input{
		{binary_tree.New([]int{3,0,0})},
		{binary_tree.New([]int{0,3,0})},
	}
	expected_outputs := []int{
		2,
		3,
	}

	f := func (i input) int  {
		return distributeCoins(i.root)
	}

	testResults(t, f, inputs, expected_outputs)
}

func TestMinBitOperations(t *testing.T) {
	type input struct {
		nums 	[]int
		k 		int
	}

	inputs := []input{
		{
			[]int{2,1,3,4},
			1,
		},
		{
			[]int{2,0,2,0},
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
		maxChoosableInteger 	int
		desiredTotal			int
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
		nums 	[]int
		target 	int
	}
	inputs := []input{
		{[]int{1,2,3}, 4},
		{[]int{9}, 3},
		{[]int{3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}, 10},
		{[]int{154,34,208,358,427,52,328,493,304,346,118,325,7,226,169,178,499,460,349,430,259,172,400,43,451,82,409,313,175,91,289,40,205,391,343,214,307,28,418,199,241,310,238,268,244,319,1,457,124,265,496,490,130,49,181,148,316,448,397,88,337,424,136,160,229,25,100,112,46,76,166,211,94,247,142,334,322,271,352,70,367,232,58,379,133,361,394,292,4,115,286,13,64,472,16,364,196,466,433,22,415,193,445,421,301,220,31,250,340,277,145,184,382,262,202,121,373,190,388,475,478,223,163,454,370,481,109,19,73,10,376,217,487,283,151,187,439,295,67,355,385,106,463,139,37,298,253,61,442,127,103,403,97,274,484,469,412,280,235,256,406,436,157,79,85,55}, 50},
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
		root 	*binary_tree.TreeNode
	}
	inputs := []input{
		{binary_tree.New([]int{1,4,3,2,4,2,5,binary_tree.NULL,binary_tree.NULL,binary_tree.NULL,binary_tree.NULL,binary_tree.NULL,binary_tree.NULL,4,6})},
		{binary_tree.New([]int{4,3,binary_tree.NULL,1,2})},
		{binary_tree.New([]int{-4,-2,-5})},
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
		n 	int
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

func TestStoneGameIII(t *testing.T) {
	type input struct {
		stoneValue 	[]int
	}
	inputs := []input{
		{[]int{1,2,3,7}},
		{[]int{1,2,3,-9}},
		{[]int{1,2,3,6}},
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
		stoneValue 	[]int
	}
	inputs := []input {
		{[]int{6,2,3,4,5,5}},
		{[]int{7,7,7,7,7,7,7}},
		{[]int{4}},
	}
	expected_outputs := []int {
		18,
		28,
		0,
	}

	f := func(i input) int {
		return stoneGameV(i.stoneValue)
	}

	testResults(t, f, inputs, expected_outputs)
}