package solution

import (
	"fmt"
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
