package solution

import (
	"bytes"
	"leetcode/algorithm"
	"leetcode/binary_tree"
	"leetcode/combinations"
	"leetcode/disjoint_set"
	"leetcode/prime_numbers"
	"leetcode/float_rounding"
	"leetcode/graph"
	"leetcode/heap"
	"leetcode/linked_list"
	"leetcode/list_node"
	"leetcode/modulo"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
*
Given the head of a linked list, we repeatedly delete consecutive sequences of nodes that sum to 0 until there are no such sequences.

After doing so, return the head of the final linked list.  You may return any such answer.

Link:
https://leetcode.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/description/?envType=daily-question&envId=2024-03-12
*/
func removeZeroSumSublists(head *list_node.ListNode) *list_node.ListNode {
	if head.Next == nil {
		if head.Val == 0 {
			return nil
		}
		return head
	}

	// Get instant access for each node value
	count := 0
	current := head
	for current != nil {
		count++
		current = current.Next
	}
	vals := make([]int, count)
	nodes := make([]*list_node.ListNode, count)
	count = 0
	current = head
	for current != nil {
		nodes[count] = current
		vals[count] = current.Val
		count++
		current = current.Next
	}

	// Otherwise we have some work to do
	sums := make([][]int, count)
	for i := 0; i < count; i++ {
		sums[i] = make([]int, count)
	}

	for i := 0; i < count; i++ {
		sums[i][i] = vals[i]
	}
	for row := 0; row < count; row++ {
		for col := row + 1; col < count; col++ {
			sums[row][col] = sums[row][col-1] + vals[col]
		}
	}

	// WLOG, we can find the greatest length 0 sequence starting at index 0, find where it ends, and repeat for whatever precedes the ending index, etc.
	prev_end := -1
	start := 0
	current_head := head
	for start < count {
		found_zero_sum := false
		end := count - 1
		for end >= start {
			if sums[start][end] == 0 {
				found_zero_sum = true
				if nodes[start] == current_head {
					if end == count-1 {
						return nil
					} else {
						start = end + 1
						current_head = nodes[start]
					}
				} else {
					if end == count-1 {
						nodes[prev_end].Next = nil
						return current_head
					} else {
						start = end + 1
						nodes[prev_end].Next = nodes[start]
					}
				}
				break
			}
			end--
		}
		if !found_zero_sum {
			prev_end = start
			start++
		}
	}

	return current_head
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a binary array nums and an integer goal, return the number of non-empty subarrays with a sum goal.

A subarray is a contiguous part of the array.

Link:
https://leetcode.com/problems/binary-subarrays-with-sum/description/?envType=daily-question&envId=2024-03-14
*/
func numSubArraysWithSum(nums []int, goal int) int {
	// Handle zero separately
	if goal == 0 {
		count := 0
		left := 0
		for left < len(nums) && nums[left] == 1 {
			left++
		}
		right := left
		for right < len(nums) {
			if nums[right] == 0 {
				count += right - left + 1
				right++
			} else {
				for right < len(nums) && nums[right] == 1 {
					right++
				}
				left = right
			}
		}

		return count
	}

	// we KNOW the sum goal is greater than 0
	left := 0
	right := 0
	sum := nums[0]
	count := 0
	// progress until we hit our goal, and move up left until it hits a 1
	for right < len(nums) && sum < goal {
		right++
		if right >= len(nums) {
			break
		}
		sum += nums[right]
	}
	if sum < goal { // not possible to reach goal
		return 0
	}
	// otherwise, it was possible to reach the goal - move up left until it hits a 1
	for nums[left] == 0 {
		left++
	}

	// at each restart of the following loop, nums[left] == 1 AND nums[right] == 1, and we need to count how many consecutive zeros lie left, and how many consecutive zeros lie right
	for right < len(nums) {
		zeros_left := 0
		left_scanner := left - 1
		for left_scanner >= 0 && nums[left_scanner] == 0 {
			zeros_left++
			left_scanner--
		}

		zeros_right := 0
		right_scanner := right + 1
		for right_scanner < len(nums) && nums[right_scanner] == 0 {
			zeros_right++
			right_scanner++
		}

		count += (zeros_left + 1) * (zeros_right + 1)

		left++
		for left < len(nums) && nums[left] == 0 {
			left++
		}
		right++
		for right < len(nums) && nums[right] == 0 {
			right++
		}
	}

	return count
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
*
You are given a 0-indexed integer array nums consisting of 3 * n elements.

You are allowed to remove any subsequence of elements of size exactly n from nums. The remaining 2 * n elements will be divided into two equal parts:

The first n elements belonging to the first part and their sum is sumfirst.
The next n elements belonging to the second part and their sum is sumsecond.
The difference in sums of the two parts is denoted as sumfirst - sumsecond.

For example, if sumfirst = 3 and sumsecond = 2, their difference is 1.
Similarly, if sumfirst = 2 and sumsecond = 3, their difference is -1.
Return the minimum difference possible between the sums of the two parts after the removal of n elements.

Link:
https://leetcode.com/problems/minimum-difference-in-sums-after-removal-of-elements/description/

Inspiration:
https://leetcode.com/problems/minimum-difference-in-sums-after-removal-of-elements/solutions/1747029/python-explanation-with-pictures-priority-queue/

User Who Gave Inspiration:
https://leetcode.com/Bakerston/
*/
func minimumDifference(nums []int) int64 {
	n := len(nums) / 3

	min_sum_left := int64(0)
	left_heap := heap.MaxHeap[int]{}
	for i := 0; i < n; i++ {
		left_heap.Insert(nums[i])
		min_sum_left += int64(nums[i])
	}

	max_sum_right := int64(0)
	right_heap := heap.MinHeap[int]{}
	for i := 2 * n; i < 3*n; i++ {
		right_heap.Insert(nums[i])
		max_sum_right += int64(nums[i])
	}

	left_sums := make([]int64, n+1) // left_sums[i] is the minimum sum of n numbers achievable in nums[0:n+i-1]
	left_sums[0] = min_sum_left
	right_sums := make([]int64, n+1) // right_sums[i] is the maximum sum of n numbers achievable in nums[n+i:3n-1]
	right_sums[n] = max_sum_right
	for i := 1; i <= n; i++ {
		// handle left
		next_left := nums[n+i-1]
		if left_heap.Peek() > next_left {
			v := left_heap.Extract()
			left_heap.Insert(next_left)
			min_sum_left -= int64(v)
			min_sum_left += int64(next_left)
		}
		left_sums[i] = min_sum_left

		// handle right
		next_right := nums[2*n-i]
		if right_heap.Peek() < next_right {
			v := right_heap.Extract()
			right_heap.Insert(next_right)
			max_sum_right -= int64(v)
			max_sum_right += int64(next_right)
		}
		right_sums[n-i] = max_sum_right
	}

	record := left_sums[0] - right_sums[0]
	for i := 1; i < len(left_sums); i++ {
		record = min(record, left_sums[i]-right_sums[i])
	}

	return record

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
*
Given a binary array nums, return the maximum length of a contiguous subarray with an equal number of 0 and 1.

Link:
https://leetcode.com/problems/contiguous-array/description/?envType=daily-question&envId=2024-03-16

Editorial:
https://leetcode.com/problems/contiguous-array/editorial/?envType=daily-question&envId=2024-03-16
*/
func findMaxLength(nums []int) int {
	total := 0
	counts := make(map[int]int)
	counts[0] = -1
	record := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] == 0 {
			total--
		} else {
			total++
		}
		earliest_occurence, key_present := counts[total]
		if key_present {
			length := i - earliest_occurence
			record = max(record, length)
		} else {
			counts[total] = i
		}
	}

	return record
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
There are some spherical balloons taped onto a flat wall that represents the XY-plane. The balloons are represented as a 2D integer array points where points[i] = [xstart, xend] denotes a balloon whose horizontal diameter stretches between xstart and xend. You do not know the exact y-coordinates of the balloons.

Arrows can be shot up directly vertically (in the positive y-direction) from different points along the x-axis. A balloon with xstart and xend is burst by an arrow shot at x if xstart <= x <= xend. There is no limit to the number of arrows that can be shot. A shot arrow keeps traveling up infinitely, bursting any balloons in its path.

Given the array points, return the minimum number of arrows that must be shot to burst all balloons.

Link:
https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/description/?envType=daily-question&envId=2024-03-18
*/
func findMinArrowShots(points [][]int) int {
	sort.SliceStable(points, func(idx_1, idx_2 int) bool {
		return points[idx_1][1] < points[idx_2][1]
	})
	// Essentially, shoot an arrow at the ends of all these balloons

	arrows := 0
	prev_end := math.MinInt
	i := 0
	for i < len(points) {
		for i < len(points) && points[i][0] <= prev_end {
			i++
		}
		if i < len(points) {
			arrows++
			prev_end = points[i][1]
		}
	}

	return arrows
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
A super ugly number is a positive integer whose prime factors are in the (sorted) array primes.

Given an integer n and an array of integers primes, return the nth super ugly number.

The nth super ugly number is guaranteed to fit in a 32-bit signed integer.

Link:
https://leetcode.com/problems/super-ugly-number/description/
*/
func nthSuperUglyNumber(n int, primes []int) int {

	// The 'index' of the most recent super ugly number we have reached
	k := 1
	// The value of the most recent super ugly number we have reached
	kth_super_ugly := 1

	prime_heaps := make([]heap.MinHeap[int], len(primes))
	for i := 0; i < len(primes); i++ {
		prime_heaps[i] = heap.MinHeap[int]{}
		prime_heaps[i].Insert(1)
	}

	for k < n {
		lowest := math.MaxInt
		idx := -1
		// Peek from each heap, and see how small multiplying said value by the heap's prime number would be
		for i := 0; i < len(prime_heaps); i++ {
			v := prime_heaps[i].Peek() * primes[i]
			if v < lowest {
				idx = i
				lowest = v
			}
		}
		// Pop whichever value corresponds to the lowest new value
		// That new value is the next super ugly number
		// Throw it into all your heaps
		k++
		kth_super_ugly = prime_heaps[idx].Extract() * primes[idx]
		for i := idx; i < len(prime_heaps); i++ {
			// Start at 'idx'; not zero, to avoid repeats
			prime_heaps[i].Insert(kth_super_ugly)
		}
	}

	return kth_super_ugly
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given an array of CPU tasks, each represented by letters A to Z, and a cooling time, n. Each cycle or interval allows the completion of one task. Tasks can be completed in any order, but there's a constraint: identical tasks must be separated by at least n intervals due to cooling time.

​Return the minimum number of intervals required to complete all tasks.

Link:
https://leetcode.com/problems/task-scheduler/description/?envType=daily-question&envId=2024-03-19
*/
func leastInterval(tasks []byte, n int) int {
	cooldown_left := make(map[byte]int)
	counts := make(map[byte]int)

	intervals := 0
	for i := 0; i < len(tasks); i++ {
		v, present := counts[tasks[i]]
		if !present {
			counts[tasks[i]] = 1
		} else {
			counts[tasks[i]] = v + 1
		}
	}
	for task := range counts {
		cooldown_left[task] = 0
	}

	tasks_scheduled := 0
	// At the beginning of this for loop, at least one task will be cooled down
	for tasks_scheduled < len(tasks) {
		intervals++
		schedule_this := byte(0)
		max_count := math.MinInt
		min_nonzero_cooldown := math.MaxInt // will be useful later
		// Find the task with zero cooldown of the highest count
		task_remaining_with_zero_cooldown := false
		for task, cooldown := range cooldown_left {
			if cooldown == 0 {
				if max_count > math.MinInt {
					// This is not the first zero cooldown task we have seen
					task_remaining_with_zero_cooldown = true
				}
				if counts[task] > max_count {
					schedule_this = task
					max_count = counts[task]
				}
			} else {
				cooldown_left[task]--
				if cooldown_left[task] == 0 {
					task_remaining_with_zero_cooldown = true
				} else {
					min_nonzero_cooldown = min(min_nonzero_cooldown, cooldown_left[task])
				}
			}
		}
		tasks_scheduled++
		counts[schedule_this]--
		if counts[schedule_this] == 0 {
			delete(counts, schedule_this)
			delete(cooldown_left, schedule_this)
		} else {
			cooldown_left[schedule_this] = n
			min_nonzero_cooldown = min(min_nonzero_cooldown, n)
		}

		if !task_remaining_with_zero_cooldown && tasks_scheduled < len(tasks) {
			for task := range cooldown_left {
				cooldown_left[task] -= min_nonzero_cooldown
			}
			intervals += min_nonzero_cooldown
		}
	}

	return intervals
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

Link:
https://leetcode.com/problems/search-in-rotated-sorted-array/description/
*/
func search(nums []int, target int) int {
	return searchPivot(nums, 0, len(nums), target)
}

/*
Helper method to search for an integer in nums, which is sorted except for a pivot
*/
func searchPivot(nums []int, left int, right int, target int) int {
	if right-left < 3 {
		if right-left < 2 {
			if nums[left] == target {
				return left
			} else {
				return -1
			}
		} else {
			if nums[left] == target {
				return left
			} else if nums[left+1] == target {
				return left + 1
			} else {
				return -1
			}
		}
	}

	search_left := left
	search_right := right
	pivot := -1
	// search for the pivot
	for search_left < search_right {
		mid := (search_left + search_right) / 2
		if mid == len(nums)-1 || nums[mid] > nums[mid+1] {
			pivot = mid
			break
		} else if mid > 0 && nums[mid-1] > nums[mid] {
			pivot = mid - 1
			break
		} else if nums[mid] > nums[search_right-1] {
			// pivot is right of mid
			search_left = mid + 1
		} else {
			// pivot is left of mid
			search_right = mid
		}
	}

	if pivot == -1 {
		// binary search  normally
		return binarySearch(nums, left, right, target)
	}

	// Otherwise
	if nums[pivot] == target {
		return pivot
	}

	// binary search from pivot + 1 to the end
	idx := binarySearch(nums, pivot+1, right, target)
	if idx != -1 {
		return idx
	}

	// binary search from start to pivot
	idx = binarySearch(nums, left, pivot, target)
	if idx != -1 {
		return idx
	}

	return -1
}

/*
There is an integer array nums sorted in non-decreasing order (not necessarily with distinct values).

Before being passed to your function, nums is rotated at an unknown pivot index k (0 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,4,4,5,6,6,7] might be rotated at pivot index 5 and become [4,5,6,6,7,0,1,2,4,4].

Given the array nums after the rotation and an integer target, return true if target is in nums, or false if it is not in nums.

You must decrease the overall operation steps as much as possible.

Link:
https://leetcode.com/problems/search-in-rotated-sorted-array-ii/description/
*/
func searchRepeats(nums []int, target int) bool {
	if nums[0] == nums[len(nums)-1] {
		// then the pivot is in the middle of a bunch of repeats
		if len(nums) < 3 {
			if len(nums) < 2 {
				return nums[0] == target
			} else {
				return nums[0] == target || nums[1] == target
			}
		}

		v := nums[0]
		if v == target {
			return true
		}
		left := 1
		right := len(nums) - 1
		for nums[left] == v && nums[right] == v && left < right {
			left++
			right--
		}
		if left >= right {
			return nums[left] == target || nums[right] == target
		} else {
			return searchPivot(nums, left, right+1, target) != -1
		}
	} else {
		// then nothing changes from the version of this problem where we did not have repeats
		if searchPivot(nums, 0, len(nums), target) != -1 {
			return true
		} else {
			return false
		}
	}
}

/*
Binary search a sorted integer array for a target value from left to right
NOTE - 'right' is exclusive
*/
func binarySearch(nums []int, left int, right int, target int) int {
	// binary search  normally
	for left < right {
		mid := (left + right) / 2
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid
		}
	}
	return -1
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.

There is only one repeated number in nums, return this repeated number.

You must solve the problem without modifying the array nums and uses only constant extra space.

Link:
https://leetcode.com/problems/find-the-duplicate-number/description/?envType=daily-question&envId=2024-03-24

Solution Link:
https://keithschwarz.com/interesting/code/?dir=find-duplicate
*/
func findDuplicate(nums []int) int {
	// This will form a Ro-shaped sequence - we need to find the beginning of the loop
	n := len(nums) - 1
	slow := n + 1
	fast := n + 1
	slow = nums[slow-1]
	fast = nums[nums[fast-1]-1]

	// Iterate slow and fast such that if slow = x_j, fast = x_{2j}
	for slow != fast {
		slow = nums[slow-1]
		fast = nums[nums[fast-1]-1]
	}

	// Now that slow = fast, slow = x_j, fast = x_{2j}
	// Let length of chain leading up to loop = c
	// Let loop length = l. j is the smallest multiple of l bigger than c
	// Proof: j > c because it must be in the loop
	//		  Also, since x_j=x_{2j}, is j iterations must mean we go around the loop a fixed number of times, so j is a multiple of l
	//		  j is the smallest such multiply of l because any smaller such multiple of l, our above iteration would have hit first

	// Now find the starting point of the loop
	finder := n + 1 // x_0
	// Also recall slow = x_j
	// Further, x_{c+j} is equivalent to iterating up to the start of the loop, and progressing around the loop an integer number of times
	// So you'll end up at the start of the loop after starting at x_0 and going through c+j iterations, and slow has already done j iterations
	// Therefore, after c iterations, finder will be at x_c - the start of the loop - by definition, and so will slow
	for slow != finder {
		finder = nums[finder-1]
		slow = nums[slow-1]
	}

	return finder
}

/*
Given an integer array nums of length n where all the integers of nums are in the range [1, n] and each integer appears once or twice, return an array of all the integers that appears twice.

You must write an algorithm that runs in O(n) time and uses only constant extra space.

Solution Discussion Credit:
https://leetcode.com/nextsde/

Link:
https://leetcode.com/problems/find-all-duplicates-in-an-array/description/?envType=daily-question&envId=2024-03-25
*/
func findDuplicates(nums []int) []int {
	duplicates := []int{}
	for _, v := range nums {
		v = int(math.Abs(float64(v)))
		if nums[v-1] < 0 {
			// Then that means v is a repeat, we ran into this value before because we have already flipped the OTHER value in array position THIS value minus 1
			duplicates = append(duplicates, v)
		} else {
			nums[v-1] = -nums[v-1]
		}
	}

	return duplicates
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an integer n, your task is to count how many strings of length n can be formed under the following rules:

Each character is a lower case vowel ('a', 'e', 'i', 'o', 'u')
Each vowel 'a' may only be followed by an 'e'.
Each vowel 'e' may only be followed by an 'a' or an 'i'.
Each vowel 'i' may not be followed by another 'i'.
Each vowel 'o' may only be followed by an 'i' or a 'u'.
Each vowel 'u' may only be followed by an 'a'.
Since the answer may be too large, return it modulo 10^9 + 7.

Link:
https://leetcode.com/problems/count-vowels-permutation/?envType=daily-question&envId=2024-03-24
*/
func countVowelPermutation(n int) int {
	chars := map[byte]int{
		'a': 0,
		'e': 1,
		'i': 2,
		'o': 3,
		'u': 4,
	}
	can_follow := map[byte][]byte{
		'a': {'e'},
		'e': {'a', 'i'},
		'i': {'a', 'e', 'o', 'u'},
		'o': {'i', 'u'},
		'u': {'a'},
	}
	sols := make([][]int, len(chars))
	for i := 0; i < len(sols); i++ {
		sols[i] = make([]int, n)
		sols[i][0] = 1
	}
	total := 0
	// we could start with any vowel
	for vowel := range chars {
		total = modulo.ModularAdd(total, topDownVowelPermutation(n, vowel, sols, chars, can_follow))
	}
	return total
}

/*
Top down helper method to find the number of vowel combinations we cna make of the given length and the vowel we are starting
*/
func topDownVowelPermutation(n int, vowel byte, sols [][]int, chars map[byte]int, can_follow map[byte][]byte) int {

	if sols[chars[vowel]][n-1] == 0 {
		// Need to solve this problem
		total := 0
		// Consider each character who can follow
		for _, character := range can_follow[vowel] {
			total = modulo.ModularAdd(total, topDownVowelPermutation(n-1, character, sols, chars, can_follow))
		}
		sols[chars[vowel]][n-1] = total
	}

	return sols[chars[vowel]][n-1]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
There are buckets buckets of liquid, where exactly one of the buckets is poisonous. To figure out which one is poisonous, you feed some number of (poor) pigs the liquid to see whether they will die or not. Unfortunately, you only have minutesToTest minutes to determine which bucket is poisonous.

You can feed the pigs according to these steps:

1) Choose some live pigs to feed.
2) For each pig, choose which buckets to feed it. The pig will consume all the chosen buckets simultaneously and will take no time. Each pig can feed from any number of buckets, and each bucket can be fed from by any number of pigs.
3) Wait for minutesToDie minutes. You may not feed any other pigs during this time.
4) After minutesToDie minutes have passed, any pigs that have been fed the poisonous bucket will die, and all others will survive.
5) Repeat this process until you run out of time.

Given buckets, minutesToDie, and minutesToTest, return the minimum number of pigs needed to figure out which bucket is poisonous within the allotted time.

Link:
https://leetcode.com/problems/poor-pigs/description/?envType=daily-question&envId=2024-03-24
*/
func poorPigs(buckets int, minutesToDie int, minutesToTest int) int {
	// According to the hints on LeetCode:
	// Say you have X pigs, and time enough for T test rounds.
	// How many different states does that generate?
	// Each pig could die on any of the rounds, or none of them, making T+1 total possibilities for each pig.
	// That's (T+1)^X possible states achieved, each corresponding to a different bucket being poisoned.
	// So pick X such that (T+1)^X >= buckets!
	toDie := float64(minutesToDie)
	toTest := float64(minutesToTest)
	T := math.Floor(toTest / toDie)
	return int(math.Ceil(math.Log2(float64(buckets)) / math.Log2(float64(T+1))))
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an array of integers nums and an integer k, return the number of contiguous subarrays where the product of all the elements in the subarray is strictly less than k.

Link:
https://leetcode.com/problems/subarray-product-less-than-k/description/?envType=daily-question&envId=2024-03-27
*/
func numSubarrayProductLessThanK(nums []int, k int) int {
	// Hint:
	// For each j, let opt(j) be the smallest i so that nums[i] * nums[i+1] * ... * nums[j] is less than k.
	// Note that opt is an increasing function.
	// Sliding window - initialize the window first
	left := 0
	prod := nums[left]
	for prod >= k {
		left++
		if left < len(nums) {
			prod = nums[left]
		} else {
			break
		}
	}
	if left == len(nums) {
		return 0
	}
	right := left
	count := 1
	for prod < k {
		right++
		if right < len(nums) && prod*nums[right] < k {
			prod *= nums[right]
			count += right - left + 1
		} else {
			right--
			break
		}
	}
	// Now that we have our biggest initial window possible, slide the window to the right
	for left < len(nums)-1 {
		left++
		prod /= nums[left-1]
		if right < left {
			right++
			prod *= nums[right]
			if prod < k {
				count++
			}
		}
		for prod >= k {
			left++
			if left >= len(nums) {
				break
			}
			prod /= nums[left-1]
			if right < left {
				right++
				prod *= nums[right]
				if prod < k {
					count++
				}
			}
		}
		// We have not found the next right window that achieves a product less than k
		for prod < k {
			right++
			if right < len(nums) && prod*nums[right] < k {
				prod *= nums[right]
				count += right - left + 1
			} else {
				right--
				break
			}
		}
	}

	return count
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
There are n cities connected by some number of flights. You are given an array flights where flights[i] = [fromi, toi, pricei] indicates that there is a flight from city fromi to city toi with cost pricei.

You are also given three integers src, dst, and k, return the cheapest price from src to dst with at most k stops. If there is no such route, return -1.

Link:
https://leetcode.com/problems/cheapest-flights-within-k-stops/description/?envType=daily-question&envId=2024-03-28
*/
func findCheapestPrice(n int, flights [][]int, src int, dst int, k int) int {
	// Use Breadth-First Search

	// Create an adjancency list
	adjacency_list := make([][][]int, n) // Each node has an array of int[2] arrays each of which record the next node and the cost to fly to it
	for _, flight := range flights {
		from := flight[0]
		to := flight[1]
		cost := flight[2]
		adjacency_list[from] = append(adjacency_list[from], []int{to, cost})
	}

	// Keep track of the cheapest cost to reach each node
	costs := make([][]int, k+1)
	for i := 0; i <= k; i++ {
		costs[i] = make([]int, n)
		for j := 0; j < n; j++ {
			if j != src {
				costs[i][j] = math.MaxInt
			}
		}
	}

	// Keep track of which nodes we have added to the queue for the given round of next stops
	added := make([][]bool, k+1)
	for i := 0; i <= k; i++ {
		added[i] = make([]bool, n)
	}

	// Now we simply apply breadth-first search from the starting node
	node_queue := linked_list.NewQueue[int]()
	node_queue.Enqueue(src)

	// Keep track of a record cost to get to the destination
	record := math.MaxInt

	// Perform depth-first search over the number of stops/iterations allowed
	for stops := 0; stops <= k; stops++ {
		// Everything in the queue, we dequeue
		num_to_add_node_from := node_queue.Length()
		for i := 0; i < num_to_add_node_from; i++ {
			from := node_queue.Dequeue()
			for _, connection := range adjacency_list[from] {
				to := connection[0]
				cost := connection[1]
				if stops == 0 {
					costs[stops][to] = cost
				} else {
					costs[stops][to] = min(costs[stops][to], costs[stops-1][from]+cost)
				}
				if to != dst && !added[stops][to] { // We will NEVER explore out from the destination.
					node_queue.Enqueue(to)
					added[stops][to] = true
				} else if to == dst {
					record = min(record, costs[stops][to])
				}
			}
		}
	}

	if record == math.MaxInt {
		return -1
	} else {
		return record
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given an integer n indicating there are n people numbered from 0 to n - 1. You are also given a 0-indexed 2D integer array meetings where meetings[i] = [xi, yi, timei] indicates that person xi and person yi have a meeting at timei. A person may attend multiple meetings at the same time. Finally, you are given an integer firstPerson.

Person 0 has a secret and initially shares the secret with a person firstPerson at time 0. This secret is then shared every time a meeting takes place with a person that has the secret. More formally, for every meeting, if a person xi has the secret at timei, then they will share the secret with person yi, and vice versa.

The secrets are shared instantaneously. That is, a person may receive the secret and share it with people in other meetings within the same time frame.

Return a list of all the people that have the secret after all the meetings have taken place. You may return the answer in any order.

Link:
https://leetcode.com/problems/find-all-people-with-secret/description/?envType=daily-question&envId=2024-03-29
*/
func findAllPeople(n int, meetings [][]int, firstPerson int) []int {
	// Nodes to implement disjoint sets
	people_nodes := make([]*disjointset.Node[int], n)
	for i := 0; i < len(people_nodes); i++ {
		people_nodes[i] = disjointset.NewNode[int](i)
	}

	// Sort all meetings by the time they occur
	sort.SliceStable(meetings, func(idx_1, idx_2 int) bool {
		return meetings[idx_1][2] < meetings[idx_2][2]
	})

	// Person 0 and Person firstPerson both know the secret
	people_nodes[0].Join(people_nodes[firstPerson])

	// Now consider all meetings
	meetings_considered := 0
	for meetings_considered < len(meetings) {
		time := meetings[meetings_considered][2]
		current := meetings_considered
		people_meeting := make(map[int]bool) // just so we can have a set
		for meetings[current][2] == time {
			person_a := meetings[current][0]
			person_b := meetings[current][1]
			people_meeting[person_a] = true
			people_meeting[person_b] = true
			people_nodes[person_a].Join(people_nodes[person_b])
			current++
			if current == len(meetings) {
				break
			}
		}
		meetings_considered = current
		for person := range people_meeting {
			if people_nodes[person].RootValue() != people_nodes[0].RootValue() {
				// To prevent time travelling - if someone this person has talked to later learns the secret, that does not mean THIS person will know
				people_nodes[person].Isolate()
			}
		}
	}

	// Now a linear sweep to see who all knows the secret
	people := []int{}
	for i := 0; i < n; i++ {
		if people_nodes[i].RootValue() == people_nodes[0].RootValue() {
			people = append(people, i)
		}
	}

	return people
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given a 0-indexed integer array nums, and you are allowed to traverse between its indices. You can traverse between index i and index j, i != j, if and only if gcd(nums[i], nums[j]) > 1, where gcd is the greatest common divisor.

Your task is to determine if for every pair of indices i and j in nums, where i < j, there exists a sequence of traversals that can take us from i to j.

Return true if it is possible to traverse between all such pairs of indices, or false otherwise.

Link:
https://leetcode.com/problems/greatest-common-divisor-traversal/?envType=daily-question&envId=2024-03-30
*/
func canTraverseAllPairs(nums []int) bool {
	s := disjointset.NewSetOfSets[int]()

	if len(nums) == 1 {
		return true
	}

	nodes := make([]*disjointset.Node[int], len(nums))
	for idx, n := range nums {
		if n == 1 {
			return false
		}
		s.MakeNode(n)
		nodes[idx] = s.GetNode(n)
	}

	for idx, n := range nums {
		prime_factors := prime_numbers.GetPrimeFactors(n)
		for _, p := range prime_factors {
			s.MakeNode(p)
			nodes[idx].Join(s.GetNode(p))
		}
	}

	first := nodes[0].RootValue()
	for i := 1; i < len(nodes); i++ {
		if nodes[i].RootValue() != first {
			return false
		}
	}
	return true
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given a 2D array of integers envelopes where envelopes[i] = [wi, hi] represents the width and the height of an envelope.

One envelope can fit into another if and only if both the width and height of one envelope are greater than the other envelope's width and height.

Return the maximum number of envelopes you can Russian doll (i.e., put one inside the other).

Note: You cannot rotate an envelope.

Link:
https://leetcode.com/problems/russian-doll-envelopes/description/

Source for Inspiration:
https://leetcode.com/problems/russian-doll-envelopes/solutions/2071477/c-java-python-best-explanation-with-pictures/
*/
func maxEnvelopes(envelopes [][]int) int {
	// First remove duplicates - sort by width then height, OR by height then width - doesn't matter
	sort.SliceStable(envelopes, func(i, j int) bool {
		if envelopes[i][0] < envelopes[j][0] {
			return true
		} else if envelopes[i][0] == envelopes[j][0] {
			return envelopes[i][1] < envelopes[j][1]
		} else {
			return false
		}
	})
	unique_envelopes := [][]int{}
	unique_widths := 1
	for idx, v := range envelopes {
		if idx > 0 {
			if envelopes[idx][0] != envelopes[idx-1][0] {
				unique_widths++
				unique_envelopes = append(unique_envelopes, v)
			} else if envelopes[idx][1] != envelopes[idx-1][1] {
				unique_envelopes = append(unique_envelopes, v)
			}
		} else {
			unique_envelopes = append(unique_envelopes, v)
		}
	}

	// Now we sort by INCREASING width, and for the same width DECREASING height
	sort.SliceStable(unique_envelopes, func(i, j int) bool {
		if unique_envelopes[i][0] < unique_envelopes[j][0] {
			return true
		} else if unique_envelopes[i][0] == unique_envelopes[j][0] {
			return unique_envelopes[i][1] > unique_envelopes[j][1]
		} else {
			return false
		}
	})

	// Copy the height values into an array
	heights := make([]int, len(unique_envelopes))
	for i := 0; i < len(unique_envelopes); i++ {
		heights[i] = unique_envelopes[i][1]
	}

	// Now apply longest increasing subsequence
	return algorithm.LongestIncreasingSubsequence(heights)
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an integer n, return the count of all numbers with unique digits, x, where 0 <= x < 10^n.

Link:
https://leetcode.com/problems/count-numbers-with-unique-digits/description/
*/
func countNumbersWithUniqueDigits(n int) int {
	// Suppose you know the number for n-2, and for n-1
	// Then for each of the numbers with EXACTLY n-1 unique digits (num(n-1)-num(n-2)), we have 10 - (n - 1) = 11 - n remaining digits to pick from
	// SO num(n) = num(n-1) + (num(n-1) - num(n-2))*(11-n)
	n_minus_2 := 1
	n_minus_1 := 10
	if n == 0 {
		return n_minus_2
	} else if n == 1 {
		return n_minus_1
	} else {
		for curr := 2; curr <= n; curr++ {
			val := n_minus_1 + (n_minus_1-n_minus_2)*(11-curr)
			n_minus_2 = n_minus_1
			n_minus_1 = val
		}
		return n_minus_1
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a string s of '(' , ')' and lowercase English characters.

Your task is to remove the minimum number of parentheses ( '(' or ')', in any positions ) so that the resulting parentheses string is valid and return any valid string.

Formally, a parentheses string is valid if and only if:

It is the empty string, contains only lowercase characters, or
It can be written as AB (A concatenated with B), where A and B are valid strings, or
It can be written as (A), where A is a valid string.

Link:
https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/description/?envType=daily-question&envId=2024-04-06
*/
func minRemoveToMakeValid(s string) string {
	to_remove := make(map[int]bool)
	idx_stack := linked_list.NewStack[int]()

	for idx, ch := range s {
		if ch == '(' {
			idx_stack.Push(idx)
		} else if ch == ')' && idx_stack.Empty() {
			to_remove[idx] = true
		} else if ch == ')' {
			idx_stack.Pop()
		}
	}

	for !idx_stack.Empty() {
		to_remove[idx_stack.Pop()] = true
	}

	byte_array := []byte{}
	for i := 0; i < len(s); i++ {
		_, ok := to_remove[i]
		if !ok {
			byte_array = append(byte_array, s[i])
		}
	}

	return string(byte_array)
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given a 0-indexed string s of even length n. The string consists of exactly n / 2 opening brackets '[' and n / 2 closing brackets ']'.

A string is called balanced if and only if:

It is the empty string, or
It can be written as AB, where both A and B are balanced strings, or
It can be written as [C], where C is a balanced string.
You may swap the brackets at any two indices any number of times.

Return the minimum number of swaps to make s balanced.

Link:
https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-string-balanced/description/
*/
func minSwaps(s string) int {
	idx_stack := linked_list.NewStack[int]()
	count := 0
	for idx, ch := range s {
		if ch == '[' {
			idx_stack.Push(idx)
		} else if ch == ']' && !idx_stack.Empty() {
			idx_stack.Pop()
		} else {
			count++
		}
	}

	// At this point, we have a certain, count where this count is the number of unmatched ']' AND the number of unmatched '['
	// So our string looks like this:
	// ...]...]...]...[...[...[...
	// Where the count of ']' and '[' is the same, and occur in this order.
	// Further, each '...' is a balanced substring of s.
	// Suppose you have ]...]...[...[
	// Note that you can swap the outer '[',']' to achieve:
	// [...]...[...], which is balanced.
	// This is accomplished in (count / 2) swaps
	// HOWEVER, suppose count is odd and you have:
	// ...]...]...]...[...[...[...
	// Then you have to swap twice, because one swap can achieve:
	// ...]...[...]...[...]...[...
	// But you obviously still need one more swap to achieve
	// ...[...[...]...[...]...]...
	// SO, in general, take the CEILING of (count / 2)

	return (count / 2) + (count % 2)
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a string s containing only three types of characters: '(', ')' and '*', return true if s is valid.

The following rules define a valid string:

Any left parenthesis '(' must have a corresponding right parenthesis ')'.
Any right parenthesis ')' must have a corresponding left parenthesis '('.
Left parenthesis '(' must go before the corresponding right parenthesis ')'.
'*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string "".

Link:
https://leetcode.com/problems/valid-parenthesis-string/description/?envType=daily-question&envId=2024-04-07
*/
func checkValidString(s string) bool {
	// This is classic dynamic programming - for a given '*', should we make it be "", ")", or "("
	least_unmatched_left := make([][]int, len(s))
	most_unmatched_left := make([][]int, len(s))
	least_unmatched_right := make([][]int, len(s))
	most_unmatched_right := make([][]int, len(s))
	for i := range len(s) {
		least_unmatched_left[i] = make([]int, len(s))
		least_unmatched_right[i] = make([]int, len(s))
		most_unmatched_left[i] = make([]int, len(s))
		most_unmatched_right[i] = make([]int, len(s))
	}

	// Base cases
	for i := 0; i < len(s); i++ {
		if s[i] == '*' {
			least_unmatched_left[i][i] = 0
			least_unmatched_right[i][i] = 0
			most_unmatched_left[i][i] = 1
			most_unmatched_right[i][i] = 1
		} else if s[i] == '(' {
			least_unmatched_left[i][i] = 1
			most_unmatched_left[i][i] = 1
		} else if s[i] == ')' {
			least_unmatched_right[i][i] = 1
			most_unmatched_right[i][i] = 1
		}
	}

	// Now we go bottom up
	for length := 2; length <= len(s); length++ {
		for start := 0; start <= len(s)-length; start++ {
			end := start + length - 1

			if s[start] == '*' {
				// Suppose we make this a ')'
				most_unmatched_right[start][end] = most_unmatched_right[start+1][end] + 1
				least_unmatched_left[start][end] = least_unmatched_left[start+1][end]
				// Suppose we make this a '('
				if least_unmatched_right[start+1][end] == 0 {
					// Then we CAN have an additional unmatched left '('
					most_unmatched_left[start][end] = most_unmatched_left[start+1][end] + 1
				} else {
					// We CANNOT have an additional unmatched left '('
					most_unmatched_left[start][end] = most_unmatched_left[start+1][end]
				}
				least_unmatched_right[start][end] = max(least_unmatched_right[start+1][end]-1, 0)
			} else if s[start] == '(' {
				least_unmatched_right[start][end] = max(least_unmatched_right[start+1][end]-1, 0)
				most_unmatched_right[start][end] = max(most_unmatched_right[start+1][end]-1, 0)
				if least_unmatched_right[start+1][end] == 0 {
					// Then we have CAN have an additional unmatched left
					most_unmatched_left[start][end] = most_unmatched_left[start+1][end] + 1
					if most_unmatched_right[start+1][end] == 0 { // We MUST have an additional unmatched left
						least_unmatched_left[start][end] = least_unmatched_left[start+1][end] + 1
					} else { // We don't HAVE to have an additional unmatched left
						least_unmatched_left[start][end] = least_unmatched_left[start+1][end]
					}
				} else {
					// Then we CANNOT have an additional unmatched left
					most_unmatched_left[start][end] = most_unmatched_left[start+1][end]
					least_unmatched_left[start][end] = least_unmatched_left[start+1][end]
				}
			} else { // ')' at the beginning - there is NO WAY we can match it
				most_unmatched_right[start][end] = most_unmatched_right[start+1][end] + 1
				least_unmatched_right[start][end] = least_unmatched_right[start+1][end] + 1
				most_unmatched_left[start][end] = most_unmatched_left[start+1][end]
				least_unmatched_left[start][end] = least_unmatched_left[start+1][end]
			}
		}
	}

	// Return the final answer
	return least_unmatched_left[0][len(s)-1] == 0 && least_unmatched_right[0][len(s)-1] == 0
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an integer n, count the total number of digit 1 appearing in all non-negative integers less than or equal to n.

Link:
https://leetcode.com/problems/number-of-digit-one/description/
*/
func countDigitOne(n int) int {
	if n == 0 {
		return 0
	}

	num_digits := int(math.Floor(math.Log10(float64(n)))) + 1
	if num_digits == 1 {
		return 1
	}

	// Let's say num_digits - 1 equals 5, for instance. Then we need to build up from 0 to 99999.
	sub_problems := make([]int, num_digits)
	sub_problems[0] = 0
	// Now we build up
	for i := 1; i < len(sub_problems); i++ {
		sub_problems[i] = 10*sub_problems[i-1] + int(math.Pow(10, float64(i-1)))
	}

	// Now we need to parse through our number
	total := 0
	one_count := 0
	for k := num_digits; k >= 1; k-- {
		digit := (n % int(math.Pow(10, float64(k)))) / int(math.Pow(10, float64(k-1)))
		total += digit * sub_problems[k-1] // For repeating hitting that many 1's 'digit' number of times
		if digit > 1 {
			total += int(math.Pow(10, float64(k-1))) // For having a '1' out front 10^{k-1} many times
		} else if digit == 1 {
			total += 1 // For the digits from the kth place onward hitting 10^{k-1}
		}
		total += one_count * digit * int(math.Pow(10, float64(k-1))) // For seeing all the 1's present from the nth place to the (k+1)th place and additional 'kth digit' number of times
		if digit == 1 {
			one_count++
		}
	}

	return total
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

Link:
https://leetcode.com/problems/coin-change/description/
*/
func coinChange(coins []int, amount int) int {
	if amount == 0 {
		return 0
	}

	sort.SliceStable(coins, func(i, j int) bool {
		return coins[i] < coins[j]
	})

	sols := make([][]int, len(coins))
	for i := 0; i < len(coins); i++ {
		sols[i] = make([]int, amount+1)
	}

	return topDownCoinChange(coins, len(coins)-1, amount, sols)
}

/*
Top down recursive helper method to use solve this problem
*/
func topDownCoinChange(coins []int, rightIdx int, amount int, sols [][]int) int {
	if sols[rightIdx][amount] == math.MaxInt {
		// We need to solve this problem
		// We are allowed to pick coins from 0 up to rightIdx
		best := math.MaxInt
		for i := 0; i <= rightIdx; i++ {
			value := coins[i]
			if value == amount {
				best = 1
				break
			} else if value < amount {
				if (amount - value) >= value {
					// We could still re-use this coin
					subsol := topDownCoinChange(coins, i, amount-value, sols)
					if subsol != -1 {
						best = min(best, 1+subsol)
					}
				} else if i > 0 {
					// We will not be able to re-use this coin
					subsol := topDownCoinChange(coins, i, amount-value, sols)
					if subsol != -1 {
						best = min(best, 1+subsol)
					}
				}
			}
		}
		if best == math.MaxInt {
			sols[rightIdx][amount] = -1
		} else {
			sols[rightIdx][amount] = best
		}
	}

	return sols[rightIdx][amount]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
There are n piles of coins on a table. Each pile consists of a positive number of coins of assorted denominations.

In one move, you can choose any coin on top of any pile, remove it, and add it to your wallet.

Given a list piles, where piles[i] is a list of integers denoting the composition of the ith pile from top to bottom, and a positive integer k, return the maximum total value of coins you can have in your wallet if you choose exactly k coins optimally.

Link:
https://leetcode.com/problems/maximum-value-of-k-coins-from-piles/description/
*/
func maxValueOfCoins(piles [][]int, k int) int {
	sols := make([][]int, len(piles))
	for i := 0; i < len(sols); i++ {
		sols[i] = make([]int, k+1)
	}
	return topDownMaxValueOfCoins(sols, piles, 0, k)
}

/*
Top-down recursive helper method to find the maximum value achievable given a certain set of piles of coins and the number of coins left you are allowed to pick
*/
func topDownMaxValueOfCoins(sols [][]int, piles [][]int, start int, coins_to_pick int) int {
	if coins_to_pick == 0 || start >= len(piles) {
		return 0
	} else if sols[start][coins_to_pick] != 0 {
		// We already solved this problem
		return sols[start][coins_to_pick]
	} else {
		// We have not yet solved this problem
		total := 0
		// First try not picking ANY coins from this pile
		record := topDownMaxValueOfCoins(sols, piles, start+1, coins_to_pick)
		// Now try picking all possible amounts of coins from this pile
		for i := 1; i <= min(len(piles[start]), coins_to_pick); i++ {
			total += piles[start][i-1]
			record = max(record, total+topDownMaxValueOfCoins(sols, piles, start+1, coins_to_pick-i))
		}
		sols[start][coins_to_pick] = record
		return sols[start][coins_to_pick]
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an array of strings words, return the smallest string that contains each string in words as a substring. If there are multiple valid strings of the smallest length, return any of them.

You may assume that no string in words is a substring of another string in words.

Link:
https://leetcode.com/problems/find-the-shortest-superstring/description/

Inspiration:
https://leetcode.com/problems/find-the-shortest-superstring/solutions/194932/travelling-salesman-problem/
*/
func shortestSuperstring(words []string) string {
	nodes := len(words)
	graph := make([][]int, nodes)
	for i := 0; i < len(graph); i++ {
		graph[i] = make([]int, nodes)
	}

	for i := 0; i < len(graph); i++ {
		for j := 0; j < len(graph); j++ {
			if j != i {
				graph[i][j] = findExtensionLength(words[i], words[j])
			}
		}
	}

	sols := make([][]*sol, nodes)
	for i := 0; i < nodes; i++ {
		sols[i] = make([]*sol, int(math.Pow(2, float64(nodes))))
		for j := 0; j < len(sols[i]); j++ {
			sols[i][j] = nil
		}
	}

	subset := int(math.Pow(2, float64(nodes))) - 1
	record := math.MaxInt
	best_path := []int{}
	for i := 0; i < nodes; i++ {
		sol := bestPath(words, sols, graph, subset, i)
		best, path := sol.cost, sol.path
		if record > best+len(words[i]) {
			best_path = path
			record = best + len(words[i])
		}
	}
	joined := words[best_path[0]]
	for i := 0; i < len(best_path)-1; i++ {
		next_word := words[best_path[i+1]]
		joined += next_word[len(next_word)-graph[best_path[i]][best_path[i+1]]:]
	}
	return joined
}

/*
This structure will be useful for the TSP
*/
type sol struct {
	path []int
	cost int
}

/*
Helper method to find the overlap between two strings given the first and the second
*/
func findExtensionLength(preceding string, following string) int {
	overlap := min(len(preceding), len(following))
	for overlap > 0 {
		if preceding[len(preceding)-overlap:] != following[:overlap] {
			overlap--
		} else {
			break
		}
	}

	return len(following) - overlap
}

/*
Helper method to implement the TSP to solve the problem in the context of joining strings together
*/
func bestPath(words []string, sols [][]*sol, graph [][]int, subset int, start int) *sol {
	if sols[start][subset] != nil {
		return sols[start][subset]
	} else {
		// Try every possible start
		path := []int{}
		best := math.MaxInt
		new_subset := subset ^ (int(math.Pow(2, float64(start))))
		for i := 0; i < len(sols); i++ {
			if (int(math.Pow(2, float64(i))))&new_subset != 0 { // automatically excludes the start due to XOR with subset yielding new_subset
				// This is a node to explore
				sub_sol_ptr := bestPath(words, sols, graph, new_subset, i)
				if (*sub_sol_ptr).cost+graph[start][i] < best {
					// Don't forget to add the additional connection
					best = sub_sol_ptr.cost + graph[start][i]
					path = sub_sol_ptr.path
					path = append([]int{start}, path...)
				}
			}
		}
		if best == math.MaxInt {
			// We had nobody to go to next
			sols[start][subset] = &sol{[]int{start}, 0}
		} else {
			sols[start][subset] = &sol{path, best}
		}

		return sols[start][subset]
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a set of distinct positive integers nums, return the largest subset answer such that every pair (answer[i], answer[j]) of elements in this subset satisfies:

	answer[i] % answer[j] == 0, or
	answer[j] % answer[i] == 0

If there are multiple solutions, return any of them.

Link:
https://leetcode.com/problems/largest-divisible-subset/
*/
func largestDivisibleSubset(nums []int) []int {
	sort.SliceStable(nums, func(i, j int) bool {
		return nums[i] < nums[j]
	})

	sols := make([][]int, len(nums))

	best := []int{}
	for i := 0; i < len(nums); i++ {
		largest_include_i := topDownLargestDivisibleSubset(sols, i, nums)
		if len(best) < len(largest_include_i) {
			best = largest_include_i
		}
	}

	return best
}

/*
Helper method to find the largest divisible subset that must include the indicated starting index
*/
func topDownLargestDivisibleSubset(sols [][]int, start int, nums []int) []int {
	if len(sols[start]) > 0 { // Have solved the problem already
		return sols[start]
	} else { // Need to solve the problem
		if start == len(nums)-1 {
			sols[start] = []int{nums[start]}
		} else {
			best := []int{nums[start]}
			for j := start + 1; j < len(nums); j++ {
				if nums[j]%nums[start] == 0 {
					include_j := topDownLargestDivisibleSubset(sols, j, nums)
					if len(best) < (len(include_j) + 1) {
						best = append([]int{nums[start]}, include_j...)
					}
				}
			}
			sols[start] = best
		}
	}
	return sols[start]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an integer n, you must transform it into 0 using the following operations any number of times:

	Change the rightmost (0th) bit in the binary representation of n.
	Change the ith bit in the binary representation of n if the (i-1)th bit is set to 1 and the (i-2)th through 0th bits are set to 0.

Return the minimum number of operations to transform n into 0.

Link:
https://leetcode.com/problems/minimum-one-bit-operations-to-make-integers-zero/description/?envType=daily-question&envId=2024-04-13
*/
func minimumOneBitOperations(n int) int {
	if n == 0 {
		return 0
	}
	left_most_bit_posn := int(math.Log2(float64(n)))
	if left_most_bit_posn == 0 {
		return 1
	} else {
		// Find the next 1
		new_n := n ^ (1 << left_most_bit_posn) // Mask out left-most 1
		return (1 << (left_most_bit_posn + 1)) - 1 - minimumOneBitOperations(new_n)
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given an integer array nums. In one operation, you can replace any element in nums with any integer.

nums is considered continuous if both of the following conditions are fulfilled:

All elements in nums are unique.
The difference between the maximum element and the minimum element in nums equals nums.length - 1.
For example, nums = [4, 2, 5, 3] is continuous, but nums = [1, 2, 3, 5, 6] is not continuous.

Return the minimum number of operations to make nums continuous.

Source for Inspiration:
https://leetcode.com/problems/minimum-number-of-operations-to-make-array-continuous/solutions/4152936/simple-c-solution-beginner-friendly-sorting-binary-search/

Link:
https://leetcode.com/problems/minimum-number-of-operations-to-make-array-continuous/description/
*/
func minOperations(nums []int) int {
	sort.SliceStable(nums, func(i, j int) bool {
		return nums[i] < nums[j]
	})

	// Surely ALL duplicates will have to change
	duplicates := 0
	uniques := []int{nums[0]}
	for i := 1; i < len(nums); i++ {
		if nums[i] == nums[i-1] {
			duplicates++
		} else {
			uniques = append(uniques, nums[i])
		}
	}

	// The amount of switches we will need to do is ALWAYS in {0, ...len-1}
	diff := len(nums) - 1
	best := len(uniques) - 1
	for i := 0; i < len(uniques); i++ {
		// Try making this the smallest value in our continuous array
		last := uniques[i] + diff
		last_posn := algorithm.BinarySearchMeetOrLower(uniques, last)
		if last_posn != -1 {
			// Then all values from i to last_posn are PRODUCTIVE values to having a continuous array - all others within 'unique' must be changed
			best = min(best, i+len(uniques)-last_posn-1)
		}
	}

	// Duplicates will be changed optimally to fill in whatever gaps are necessary to choose the above best starting value for our continuous array
	return best + duplicates
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
There is an undirected connected tree with n nodes labeled from 0 to n - 1 and n - 1 edges.

You are given the integer n and the array edges where edges[i] = [a_i, b_i] indicates that there is an edge between nodes a_i and b_i in the tree.

Return an array answer of length n where answer[i] is the sum of the distances between the ith node in the tree and all other nodes.

Link:
https://leetcode.com/problems/sum-of-distances-in-tree/description/
*/
func sumOfDistancesInTree(n int, edges [][]int) []int {
	// Create our array of distance counts, which we will be returning
	distance_sums := make([]int, n)

	// First create a connectivity list
	connections := make([]map[int]bool, n)
	for _, edge := range edges {
		first := edge[0]
		second := edge[1]
		if connections[first] == nil {
			connections[first] = make(map[int]bool)
		}
		if connections[second] == nil {
			connections[second] = make(map[int]bool)
		}
		connections[first][second] = true
		connections[second][first] = true
	}

	// Use BFS to find the total sum of all node distances from a start node (0 works)
	sum := 0
	num_edges := 0
	visited := make([]bool, n)
	visited[0] = true
	q := linked_list.NewQueue[int]()
	q.Enqueue(0)
	for !q.Empty() {
		dequeue := q.Length()
		for i := 0; i < dequeue; i++ {
			sum += num_edges
			next := q.Dequeue()
			for neigher := range connections[next] {
				if !visited[neigher] {
					visited[neigher] = true
					q.Enqueue(neigher)
				}
			}
		}
		num_edges++
	}
	distance_sums[0] = sum

	// Collapse all nodes which only have one edge connected to them AND whose neighbor has more than one edge connected to it (and keep track of them for later)
	single_edge_nodes := []int{}
	weights := make([]int, n)
	single := make([]bool, n)
	for node, edges := range connections {
		if len(edges) == 1 {
			// Then node 'idx' can be collapsed into its neighbor IFF that neighbor has more than one edge connected to it
			for neighbor := range edges { // (There will only be one neighbor)
				if len(connections[neighbor]) > 1 {
					// It is safe to mark this node as a single edge node
					single_edge_nodes = append(single_edge_nodes, node)
					single[node] = true
				}
			}
		}
	}
	for _, node := range single_edge_nodes {
		for neighbor := range connections[node] { // (There will only be one neighbor)
			delete(connections[neighbor], node)
			weights[neighbor]++
		}
	}

	// Now starting at every singular node, we are ready to traverse the rest of our graph, changing the sum as needed
	// This will be DFS-esque
	sols := make(map[int]map[int]int)
	visited = make([]bool, n)
	st := linked_list.NewStack[int]()
	visited[0] = true
	st.Push(0)
	for !st.Empty() {
		from := st.Pop()
		curr_sum := distance_sums[from]
		for next := range connections[from] {
			if !visited[next] {
				// Some nodes just got farther progressing to next, and some got closer - change the current sum accordingly
				distance_sums[next] = curr_sum - countNewFromHere(from, next, connections, sols, weights, single) + countNewFromHere(next, from, connections, sols, weights, single)
				visited[next] = true
				st.Push(next)
			}
		}
	}

	// The above algorithm did not necessarily include the weights for all the single-connection nodes - since those were deleted - luckily we can fix that easily
	for _, node := range single_edge_nodes {
		// Record this node's path length sum
		path_length_sum := n - 1                  // Start with number of nodes minus 1
		for neighbor := range connections[node] { // (There will only be one neighbor)
			path_length_sum += distance_sums[neighbor] - 1
		}
		distance_sums[node] = path_length_sum
	}

	return distance_sums
}

/*
Helper method to count the number of nodes that just got closer should we traverse along this edge
*/
func countNewFromHere(from int, to int, connections []map[int]bool, sols map[int]map[int]int, weights []int, single []bool) int {
	// Check and see if we have solved this problem
	from_map, ok := sols[from]
	if !ok {
		from_map = make(map[int]int)
		sols[from] = from_map
	}
	sum, ok := from_map[to]
	if !ok {
		// We have not solved this problem - run our algorithm
		sum = weights[to]
		if single[from] {
			sum--
		}
		for next := range connections[to] {
			if next != from {
				sum += 1 + countNewFromHere(to, next, connections, sols, weights, single)
			}
		}
		from_map[to] = sum
	}
	return sum
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

Link:
https://leetcode.com/problems/number-of-islands/description/?envType=daily-question&envId=2024-04-19
*/
func numIslands(grid [][]byte) int {
	// Join all of the '1' cells together through disjoint sets, and the number of disjoint sets is the number of islands
	type coordinate struct {
		row int
		col int
	}

	rows := len(grid)
	cols := len(grid[0])

	nodes := make([][]*disjointset.Node[coordinate], rows)
	for j := 0; j < rows; j++ {
		nodes[j] = make([]*disjointset.Node[coordinate], cols)
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if grid[i][j] == '0' {
				continue
			}
			nodes[i][j] = disjointset.NewNode(coordinate{i, j})
			if i > 0 { // Look up
				if grid[i-1][j] == '1' {
					nodes[i][j].Join(nodes[i-1][j])
				}
			}
			if j > 0 { // Look left
				if grid[i][j-1] == '1' {
					nodes[i][j].Join(nodes[i][j-1])
				}
			}
		}
	}

	islands := 0
	seen := make([][]bool, rows)
	for i := 0; i < rows; i++ {
		seen[i] = make([]bool, cols)
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if nodes[i][j] != nil {
				node_ptr := nodes[i][j]
				island_row := node_ptr.RootValue().row
				island_col := node_ptr.RootValue().col
				if !seen[island_row][island_col] {
					islands++
					seen[island_row][island_col] = true
				}
			}
		}
	}
	return islands

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given a 0-indexed m x n binary matrix land where a 0 represents a hectare of forested land and a 1 represents a hectare of farmland.

To keep the land organized, there are designated rectangular areas of hectares that consist entirely of farmland. These rectangular areas are called groups. No two groups are adjacent, meaning farmland in one group is not four-directionally adjacent to another farmland in a different group.

land can be represented by a coordinate system where the top left corner of land is (0, 0) and the bottom right corner of land is (m-1, n-1). Find the coordinates of the top left and bottom right corner of each group of farmland. A group of farmland with a top left corner at (r1, c1) and a bottom right corner at (r2, c2) is represented by the 4-length array [r1, c1, r2, c2].

Return a 2D array containing the 4-length arrays described above for each group of farmland in land. If there are no groups of farmland, return an empty array. You may return the answer in any order.

Link:
https://leetcode.com/problems/find-all-groups-of-farmland/description/?envType=daily-question&envId=2024-04-20
*/
func findFarmland(land [][]int) [][]int {
	// Join all of the '1' cells together through disjoint sets, and the number of disjoint sets is the number of islands
	type coordinate struct {
		row int
		col int
	}

	rows := len(land)
	cols := len(land[0])

	nodes := make([][]*disjointset.Node[coordinate], rows)
	for j := 0; j < rows; j++ {
		nodes[j] = make([]*disjointset.Node[coordinate], cols)
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if land[i][j] == 0 {
				continue
			}
			nodes[i][j] = disjointset.NewNode(coordinate{i, j})
			if i > 0 { // Look up
				if land[i-1][j] == 1 {
					nodes[i][j].Join(nodes[i-1][j])
				}
			}
			if j > 0 { // Look left
				if land[i][j-1] == 1 {
					nodes[i][j].Join(nodes[i][j-1])
				}
			}
		}
	}

	corners := [][]int{}
	seen := make([][]bool, rows)
	for i := 0; i < rows; i++ {
		seen[i] = make([]bool, cols)
	}
	// We need to remember the "index" of each plot of land
	plots := make(map[int]map[int]int)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if nodes[i][j] != nil {
				node_ptr := nodes[i][j]
				island_row := node_ptr.RootValue().row
				island_col := node_ptr.RootValue().col
				if !seen[island_row][island_col] {
					// We just ran into the top left corner of this plot of farmland
					plot := []int{i, j}
					plots[island_row] = make(map[int]int)
					plots[island_row][island_col] = len(corners) // Storing the right index
					corners = append(corners, plot)
					seen[island_row][island_col] = true
				} else {
					// We can set the new record for the bottom right corner of this plot of farmland
					plot_index := plots[island_row][island_col]
					// Now modify this plot
					if len(corners[plot_index]) == 2 {
						corners[plot_index] = append(corners[plot_index], island_row)
						corners[plot_index] = append(corners[plot_index], island_col)
					} else {
						corners[plot_index][2] = island_row
						corners[plot_index][3] = island_col
					}
				}
			}
		}
	}

	for idx, plot := range corners {
		if len(plot) == 2 { // Single cell
			// Technically plot[0] equals plot[1], but you get the idea
			plot = append(plot, plot[0])
			plot = append(plot, plot[1])
			corners[idx] = plot
		}
	}

	return corners
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
We are playing the Guessing Game. The game will work as follows:

1) I pick a number between 1 and n.
2) You guess a number.
3) If you guess the right number, you win the game.
4) If you guess the wrong number, then I will tell you whether the number I picked is higher or lower, and you will continue guessing.
5) Every time you guess a wrong number x, you will pay x dollars. If you run out of money, you lose the game.

Given a particular n, return the minimum amount of money you need to guarantee a win regardless of what number I pick.

Link:
https://leetcode.com/problems/guess-number-higher-or-lower-ii/description/

Source for Inspiration:
https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-1-introduction/#
*/
func getMoneyAmount(n int) int {
	min_sols := make([][]int, n+1)
	for i := 0; i <= n; i++ {
		min_sols[i] = make([]int, n+1)
		for j := 0; j <= n; j++ {
		}
	}
	return topDownMinMoney(1, n, min_sols)
}

/*
This is the part of the minimizer - see which choice will give the maximizer the lowest possible max result
*/
func topDownMinMoney(lower_bound int, upper_bound int, min_sols [][]int) int {
	if lower_bound == upper_bound {
		return 0 // You're not gonna need any money to guess right
	} else if min_sols[lower_bound][upper_bound] != 0 {
		return min_sols[lower_bound][upper_bound]
	} else {
		// Try every possible first guess, and see what the worst result could be
		// Go with the lowest possible worst result
		best := math.MaxInt
		for choice := lower_bound; choice <= upper_bound; choice++ {
			best = min(best, choice+topDownMaxMoney(lower_bound, upper_bound, choice, min_sols))
		}
		// Record and return
		min_sols[lower_bound][upper_bound] = best
		return best
	}
}

/*
This is the part of the maximizer - see which choice will give the minimizer the highest possible min result
*/
func topDownMaxMoney(lower_bound int, upper_bound int, guess int, min_sols [][]int) int {
	if lower_bound == upper_bound {
		return 0
	} else if guess == lower_bound {
		return topDownMinMoney(guess+1, upper_bound, min_sols)
	} else if guess == upper_bound {
		return topDownMinMoney(lower_bound, guess-1, min_sols)
	} else {
		return max(topDownMinMoney(lower_bound, guess-1, min_sols), topDownMinMoney(guess+1, upper_bound, min_sols))
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You have a lock in front of you with 4 circular wheels. Each wheel has 10 slots: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'. The wheels can rotate freely and wrap around: for example we can turn '9' to be '0', or '0' to be '9'. Each move consists of turning one wheel one slot.

The lock initially starts at '0000', a string representing the state of the 4 wheels.

You are given a list of deadends dead ends, meaning if the lock displays any of these codes, the wheels of the lock will stop turning and you will be unable to open it.

Given a target representing the value of the wheels that will unlock the lock, return the minimum total number of turns required to open the lock, or -1 if it is impossible.

Link:
https://leetcode.com/problems/open-the-lock/description/?envType=daily-question&envId=2024-04-22
*/
func openLock(deadends []string, target string) int {
	// According to the hint, this is a graph problem with 10,000 different nodes - 0000 through 9999
	// A node exists IF AND ONLY IF it is not in deadends
	// Two nodes are connected IF AND ONLY IF exactly one of the digits between the two of them differs, and the difference is either of distance 1 (e.g. 7-8) or distance 9 (e.g. 0-9)
	// We want the shortest path from 0000 to target
	num_nodes := 10000
	target_num, _ := strconv.Atoi(target)

	type lock_node struct {
		underlying_node *graph.GraphNode
		lock_values     []int
	}

	deadends_set := make(map[int]bool)
	for _, deadend := range deadends {
		deadend_int, _ := strconv.Atoi(deadend)
		if deadend_int == 0 || deadend_int == target_num {
			return -1
		}
		deadends_set[deadend_int] = true
	}

	lock_nodes := make([]*lock_node, num_nodes)
	for i := 0; i < num_nodes; i++ {
		_, ok := deadends_set[i]
		if !ok {
			lock_nodes[i] = &lock_node{
				underlying_node: &graph.GraphNode{
					Id:          i,
					Connections: []*graph.Edge{},
					IsVisited:   false,
					Cost:        math.MaxInt,
				},
				lock_values: []int{(i % 10) / 1, (i % 100) / 10, (i % 1000) / 100, (i % 10000) / 1000},
			}
		}
	}

	// Add all of each node's connections
	for i := 0; i < num_nodes; i++ {
		if lock_nodes[i] == nil {
			continue
		}
		current := lock_nodes[i]
		// Generate the neighbors
		lock_nums := current.lock_values
		for i := 0; i < 4; i++ {
			// Find all the neighbors by changing each lock value
			lock := lock_nums[i]
			new_lock_nums := [][]int{
				{lock_nums[0], lock_nums[1], lock_nums[2], lock_nums[3]},
				{lock_nums[0], lock_nums[1], lock_nums[2], lock_nums[3]},
			}
			switch lock {
			case 0: // 9 and 1
				new_lock_nums[0][i] = 9
				new_lock_nums[1][i] = 1
			case 1: // 0 and 2
				new_lock_nums[0][i] = 0
				new_lock_nums[1][i] = 2
			case 2: // 1 and 3
				new_lock_nums[0][i] = 1
				new_lock_nums[1][i] = 3
			case 3: // 2 and 4
				new_lock_nums[0][i] = 2
				new_lock_nums[1][i] = 4
			case 4: // 3 and 5
				new_lock_nums[0][i] = 3
				new_lock_nums[1][i] = 5
			case 5: // 4 and 6
				new_lock_nums[0][i] = 4
				new_lock_nums[1][i] = 6
			case 6: // 5 and 7
				new_lock_nums[0][i] = 5
				new_lock_nums[1][i] = 7
			case 7: // 6 and 7
				new_lock_nums[0][i] = 6
				new_lock_nums[1][i] = 7
			case 8: // 7 and 9
				new_lock_nums[0][i] = 7
				new_lock_nums[1][i] = 9
			case 9: // 8 and 0
				new_lock_nums[0][i] = 8
				new_lock_nums[1][i] = 0
			}
			id_1 := new_lock_nums[0][0] + new_lock_nums[0][1]*10 + new_lock_nums[0][2]*100 + new_lock_nums[0][3]*1000
			id_2 := new_lock_nums[1][0] + new_lock_nums[1][1]*10 + new_lock_nums[1][2]*100 + new_lock_nums[1][3]*1000
			if lock_nodes[id_1] != nil {
				// Not a dead end
				current.underlying_node.Connections = append(current.underlying_node.Connections, &graph.Edge{To: lock_nodes[id_1].underlying_node, Weight: 1})
			}
			if lock_nodes[id_2] != nil {
				// Not a dead end
				current.underlying_node.Connections = append(current.underlying_node.Connections, &graph.Edge{To: lock_nodes[id_2].underlying_node, Weight: 1})
			}
		}
	}

	underlying_nodes := make([]*graph.GraphNode, len(lock_nodes))
	for idx, lock_node := range lock_nodes {
		if lock_node != nil {
			underlying_nodes[idx] = lock_node.underlying_node
		}
	}

	// No need for Djikstra's algorithm here as all edge weights are 1 - BFS will do
	q := linked_list.NewQueue[*graph.GraphNode]()
	lock_nodes[0].underlying_node.IsVisited = true
	q.Enqueue(lock_nodes[0].underlying_node)
	length := 0
	for !q.Empty() {
		empty_this_many := q.Length()
		for i := 0; i < empty_this_many; i++ {
			next := q.Dequeue()
			if next.Id == target_num {
				return length
			} else {
				for _, edge := range next.Connections {
					if !edge.To.IsVisited {
						edge.To.IsVisited = true
						q.Enqueue(edge.To)
					}
				}
			}
		}
		length++
	}

	return -1
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
A tree is an undirected graph in which any two vertices are connected by exactly one path. In other words, any connected graph without simple cycles is a tree.

Given a tree of n nodes labelled from 0 to n - 1, and an array of n - 1 edges where edges[i] = [ai, bi] indicates that there is an undirected edge between the two nodes ai and bi in the tree, you can choose any node of the tree as the root. When you select a node x as the root, the result tree has height h. Among all possible rooted trees, those with minimum height (i.e. min(h))  are called minimum height trees (MHTs).

Return a list of all MHTs' root labels. You can return the answer in any order.

The height of a rooted tree is the number of edges on the longest downward path between the root and a leaf.

Link:
https://leetcode.com/problems/minimum-height-trees/description/?envType=daily-question&envId=2024-04-23

Inspiration:
https://leetcode.com/problems/minimum-height-trees/solutions/5060930/full-explanation-bfs-remove-leaf-nodes/?envType=daily-question&envId=2024-04-23
*/
func findMinHeightTrees(n int, edges [][]int) []int {
	if n == 1 {
		return []int{0}
	} else if n == 2 {
		return []int{0, 1}
	}

	// Turns out, all we need to do is repeatedly remove vertices
	connectivity_list := make(map[int]map[int]bool)

	for _, edge := range edges {
		first := edge[0]
		second := edge[1]

		// First node POV
		neighbors, ok := connectivity_list[first]
		if !ok {
			// node not seen yet
			neighbors = make(map[int]bool)
		}
		neighbors[second] = true
		connectivity_list[first] = neighbors

		// Second node POV
		neighbors, ok = connectivity_list[second]
		if !ok {
			// node not seen yet
			neighbors = make(map[int]bool)
		}
		neighbors[first] = true
		connectivity_list[second] = neighbors
	}

	// Remove leaf nodes until there is nothing left
	for len(connectivity_list) > 2 {
		// Find your leaves
		leaves := []int{}
		for id, neighbors := range connectivity_list {
			if len(neighbors) == 1 {
				// MAY be a leaf node
				for neighbor_id := range neighbors {
					if len(connectivity_list[neighbor_id]) > 1 {
						// Then id IS a leaf node
						leaves = append(leaves, id)
					}
				}
			}
		}
		for _, id := range leaves {
			neighbors := connectivity_list[id]
			for neighbor_id := range neighbors { // There will only be one neighbor
				delete(connectivity_list, id)
				delete(connectivity_list[neighbor_id], id)
			}
		}
	}

	// Return all the nodes left over
	remaining := []int{}
	for id := range connectivity_list {
		remaining = append(remaining, id)
	}

	return remaining
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
A wiggle sequence is a sequence where the differences between successive numbers strictly alternate between positive and negative. The first difference (if one exists) may be either positive or negative. A sequence with one element and a sequence with two non-equal elements are trivially wiggle sequences.

For example, [1, 7, 4, 9, 2, 5] is a wiggle sequence because the differences (6, -3, 5, -7, 3) alternate between positive and negative.
In contrast, [1, 4, 7, 2, 5] and [1, 7, 4, 5, 5] are not wiggle sequences. The first is not because its first two differences are positive, and the second is not because its last difference is zero.
A subsequence is obtained by deleting some elements (possibly zero) from the original sequence, leaving the remaining elements in their original order.

Given an integer array nums, return the length of the longest wiggle subsequence of nums.

Link:
https://leetcode.com/problems/wiggle-subsequence/
*/
func wiggleMaxLength(nums []int) int {
	// Firstly, any consecutive identical values are not helpful in increasing a wiggle subsequence length - remove
	new_nums := []int{nums[0]}
	for i := 1; i < len(nums); i++ {
		if nums[i] != nums[i-1] {
			new_nums = append(new_nums, nums[i])
		}
	}

	// Edge case
	if len(new_nums) == 1 {
		return 1
	}

	// Bottom-up dynamic programming will accomplish this - we need to keep track of the maximum wiggle subsequence from start->end
	// The maximum wiggle lengths from start->end being allowed
	max_wiggle_lengths := make([][]int, len(new_nums))
	for i := 0; i < len(new_nums); i++ {
		max_wiggle_lengths[i] = make([]int, len(new_nums))
	}
	// Was the last addition an increase or a decrease? Was it allowed to be both?
	last_change_allowed := make([][]int, len(new_nums)) // 0 - record could be accomplished by both, 1 - only accomplished by increase, 2 - only accomplished by decrease
	for i := 0; i < len(new_nums); i++ {
		last_change_allowed[i] = make([]int, len(new_nums))
	}
	// What was the last value used in the longest possible wiggle length subsequence from start to end?
	// We do not need to keep track of this - it will ALWAYS be new_nums[end]
	// Proof:
	// If new_nums[end] can extend a previous subsequence to give you a new record, obviously the new end is nums[end]
	// Otherwise,
	// If new_nums[end] is greater than the ending of a subsequence with an addition, then it's better to use nums[end] as the new ending since it'll have to be followed by a decrease
	// Otherwise, new_nums[end] is smaller than the ending of a subsequence with a subtraction, so it's better to use nums[end] as the new ending since it'll have to be followed by an increase

	// Base cases - lengths 2 and 1
	for i := 1; i < len(new_nums); i++ {
		max_wiggle_lengths[i-1][i] = 2
		if new_nums[i] > new_nums[i-1] {
			last_change_allowed[i-1][i] = 1 // Accomplished only by an increase
		} else {
			last_change_allowed[i-1][i] = 2 // Accomplished only by a decrease
		}
	}

	// Now use bottom-up dynamic programming to solve our problem
	for length := 3; length <= len(new_nums); length++ {
		for start := 0; start <= len(new_nums)-length; start++ {
			end := start + length - 1

			// We have two subproblems to look at:

			// First subproblem: start -> end-1 where we DO allow start
			prev_end := new_nums[end-1] // Because remember from above, the best subsequence from (i,j) should ALWAYS end in nums[j]
			first_record := max_wiggle_lengths[start][end-1]
			first_decrease_record := first_record
			if last_change_allowed[start][end-1] == 0 || last_change_allowed[start][end-1] == 1 {
				// Then one possible extension is being less than the previous end
				if new_nums[end] < prev_end {
					first_decrease_record++
				}
			}
			first_increase_record := first_record
			if last_change_allowed[start][end-1] == 0 || last_change_allowed[start][end-1] == 2 {
				// Then one possible extension is being greater than the previous end
				if new_nums[end] > prev_end {
					first_increase_record++
				}
			}
			first_record_last_change_allowed := last_change_allowed[start][end-1]
			if first_increase_record > first_decrease_record {
				// Record ONLY accomplished by increase
				first_record_last_change_allowed = 1
			} else if first_increase_record < first_decrease_record {
				// Record ONLY accomplished by decrease
				first_record_last_change_allowed = 2
			}
			first_record = max(first_decrease_record, first_increase_record)

			second_record := max_wiggle_lengths[start+1][end]
			second_record_last_change_allowed := last_change_allowed[start+1][end]

			new_record_last_change_allowed := 0
			if first_record == second_record {
				if first_record_last_change_allowed == 1 && second_record_last_change_allowed == 1 {
					// No choice
					new_record_last_change_allowed = 1
				} else if first_record_last_change_allowed == 2 && second_record_last_change_allowed == 2 {
					// No choice
					new_record_last_change_allowed = 2
				}
			} else if first_record > second_record { // first subproblem won
				new_record_last_change_allowed = first_record_last_change_allowed
			} else { // second won
				new_record_last_change_allowed = second_record_last_change_allowed
			}

			new_record := max(first_record, second_record)
			max_wiggle_lengths[start][end] = new_record
			last_change_allowed[start][end] = new_record_last_change_allowed
		}
	}

	return max_wiggle_lengths[0][len(new_nums)-1]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given a string s consisting of lowercase letters and an integer k. We call a string t ideal if the following conditions are satisfied:

- t is a subsequence of the string s.
- The absolute difference in the alphabet order of every two adjacent letters in t is less than or equal to k.
- Return the length of the longest ideal string.

A subsequence is a string that can be derived from another string by deleting some or no characters without changing the order of the remaining characters.

Note that the alphabet order is not cyclic. For example, the absolute difference in the alphabet order of 'a' and 'z' is 25, not 1.

Link: (The hint was helpful)
https://leetcode.com/problems/longest-ideal-subsequence/description/?envType=daily-question&envId=2024-04-25
*/
func longestIdealString(s string, k int) int {
	asciis := make([]int, len(s))
	for i := 0; i < len(s); i++ {
		asciis[i] = int(s[i])
	}

	// Now we need to ask ourselves the question - at a particular index i, what is the longest possible ideal subsequence that MUST end at index i?
	record := 1
	records := make([]int, len(asciis))
	// Keep track of a map of value versus earliest index occurrence (seen thus far as we iterate)
	value_occurences := make(map[int]int)
	for i := 0; i < len(asciis); i++ {
		current_val := asciis[i]
		current_record_for_i := 1
		for other_val := current_val - k; other_val <= current_val+k; other_val++ {
			idx, ok := value_occurences[other_val]
			if ok {
				current_record_for_i = max(current_record_for_i, 1+records[idx])
			}
		}
		// Update the latest occurrence for our current value
		// Think about it - if we end on a certain SAME value, we'd rather force ourselves to end later rather than earlier - that's only going to give us a longer ideal subsequence
		value_occurences[current_val] = i
		records[i] = current_record_for_i
		record = max(record, current_record_for_i)
	}

	return record
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an n x n integer matrix grid, return the minimum sum of a falling path with non-zero shifts.

A falling path with non-zero shifts is a choice of exactly one element from each row of grid such that no two elements chosen in adjacent rows are in the same column.

Link:
https://leetcode.com/problems/minimum-falling-path-sum-ii/description/?envType=daily-question&envId=2024-04-26
*/
func minFallingPathSum(grid [][]int) int {
	if len(grid) == 1 {
		return grid[0][0]
	}

	sums := make([][]int, len(grid))
	// We will have a square grid
	for i := 0; i < len(sums); i++ {
		sums[i] = make([]int, len(grid[0]))
	}
	copy(sums[0], grid[0])

	// Keep track of the two best path sums from the previous row - they're the only one's we're going to be using
	best_col := -1
	second_best_col := -1
	best_val := math.MaxInt
	second_best_val := math.MaxInt
	for i := 0; i < len(grid[0]); i++ {
		if best_val > sums[0][i] {
			if best_val < math.MaxInt {
				// We have already found a candidate for lowest on this row
				second_best_val = best_val
				second_best_col = best_col
			}
			// But either way, we still update the best record
			best_val = sums[0][i]
			best_col = i
		} else if second_best_val > sums[0][i] {
			second_best_val = sums[0][i]
			second_best_col = i
		}
	}

	for row := 1; row < len(grid); row++ {
		best_val := math.MaxInt
		second_best_val := math.MaxInt
		new_best_col := -1
		new_second_best_col := -1
		for col := 0; col < len(grid[row]); col++ {
			if col == best_col { // We can't take the best previous value because it was in the same column, so take second best
				sums[row][col] = grid[row][col] + sums[row-1][second_best_col]
			} else { // We CAN take the best previous value
				sums[row][col] = grid[row][col] + sums[row-1][best_col]
			}
			if best_val > sums[row][col] {
				if best_val < math.MaxInt {
					// We have already found a candidate for lowest on this row
					second_best_val = best_val
					new_second_best_col = new_best_col
				}
				// But either way, we still update the best record
				best_val = sums[row][col]
				new_best_col = col
			} else if second_best_val > sums[row][col] {
				second_best_val = sums[row][col]
				new_second_best_col = col
			}
		}
		best_col = new_best_col
		second_best_col = new_second_best_col
	}

	min_path_sum := math.MaxInt
	for _, path_sum := range sums[len(sums)-1] {
		min_path_sum = min(min_path_sum, path_sum)
	}

	return min_path_sum
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
In the video game Fallout 4, the quest "Road to Freedom" requires players to reach a metal dial called the "Freedom Trail Ring" and use the dial to spell a specific keyword to open the door.

Given a string ring that represents the code engraved on the outer ring and another string key that represents the keyword that needs to be spelled, return the minimum number of steps to spell all the characters in the keyword.

Initially, the first character of the ring is aligned at the "12:00" direction. You should spell all the characters in key one by one by rotating ring clockwise or anticlockwise to make each character of the string key aligned at the "12:00" direction and then by pressing the center button.

At the stage of rotating the ring to spell the key character key[i]:

- You can rotate the ring clockwise or anticlockwise by one place, which counts as one step. The final purpose of the rotation is to align one of ring's characters at the "12:00" direction, where this character must equal key[i].
- If the character key[i] has been aligned at the "12:00" direction, press the center button to spell, which also counts as one step. After the pressing, you could begin to spell the next character in the key (next stage). Otherwise, you have finished all the spelling.

Link:
https://leetcode.com/problems/freedom-trail/description/?envType=daily-question&envId=2024-04-27
*/
func findRotateSteps(ring string, key string) int {
	// Keep track of each character's locations
	char_locations := make(map[rune][]int)
	for idx, r := range ring {
		_, ok := char_locations[r]
		if !ok {
			char_locations[r] = []int{idx}
		} else {
			char_locations[r] = append(char_locations[r], idx)
		}
	}

	// For a given position in 'ring', and position in 'key', we need to know how many moves it will take to enter in the rest of 'key', assuming the first character we need is our current position in 'ring'
	sols := make([][]int, len(ring))
	for i := 0; i < len(sols); i++ {
		sols[i] = make([]int, len(key))
	}

	moves := math.MaxInt
	// Move to all possible positions of the first character we need, and take the option which gives us the best result
	if ring[0] == key[0] {
		return topDownRotateSteps(ring, key, 0, len(key), sols, char_locations)
	} else {
		for _, posn := range char_locations[rune(key[0])] {
			steps := min(posn, len(ring)-posn)
			moves = min(moves, steps+topDownRotateSteps(ring, key, posn, len(key), sols, char_locations))
		}
		return moves
	}
}

/*
Top-down recursive helper method to find the minimum number of steps required to match the rest of the characters in key given our current position
*/
func topDownRotateSteps(ring string, key string, posn int, chars_left int, sols [][]int, char_locations map[rune][]int) int {
	if chars_left == 1 {
		return 1
	} else {
		if sols[posn][chars_left-1] == 0 {
			// We need to solve this problem
			next_char := rune(key[len(key)-chars_left+1])
			next_moves := math.MaxInt
			for _, next_posn := range char_locations[next_char] {
				// Take the minimum amount of steps you need to get from posn to next_posn
				steps := min(max(next_posn, posn)-min(next_posn, posn), min(next_posn, posn)+len(ring)-max(next_posn, posn))
				next_moves = min(next_moves, steps+topDownRotateSteps(ring, key, next_posn, chars_left-1, sols, char_locations))
			}
			sols[posn][chars_left-1] = 1 + next_moves
		}
		return sols[posn][chars_left-1]
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given the root of a binary tree with n nodes where each node in the tree has node.val coins. There are n coins in total throughout the whole tree.

In one move, we may choose two adjacent nodes and move one coin from one node to another. A move may be from parent to child, or from child to parent.

Return the minimum number of moves required to make every node have exactly one coin.

Link:
https://leetcode.com/problems/distribute-coins-in-binary-tree/
*/
func distributeCoins(root *binary_tree.TreeNode) int {
	moves := []int{0}
	coinsFromAbove(root, moves)
	return moves[0]
}

/*
Recursive helper method
*/
func coinsFromAbove(current *binary_tree.TreeNode, moves []int) int {
	num := 1 - current.Val
	if current.Left != nil {
		num += coinsFromAbove(current.Left, moves)
	}
	if current.Right != nil {
		num += coinsFromAbove(current.Right, moves)
	}
	// We either need coins from above, to travel through me, or coins from below to travel through me
	moves[0] += int(math.Abs(float64(num)))
	return num
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given a 0-indexed integer array nums and a positive integer k.

You can apply the following operation on the array any number of times:
- Choose any element of the array and flip a bit in its binary representation. Flipping a bit means changing a 0 to 1 or vice versa.

Return the minimum number of operations required to make the bitwise XOR of all elements of the final array equal to k.

Note that you can flip leading zero bits in the binary representation of elements. For example, for the number (101) (base 2) you can flip the fourth bit and obtain (1101) (base 2).

Link:
https://leetcode.com/problems/minimum-number-of-operations-to-make-array-xor-equal-to-k/?envType=daily-question&envId=2024-04-29
*/
func minBitOperations(nums []int, k int) int {
	// For every single 1 in the binary representation of k, we need an ODD number of elements in nums to have a one in that position

	// So for each bit that's a 1 in k, linear search through all numbers in nums, and the number of operations needed is the number of said elements that have a 1 in that position MINUS 1
	// If NO elements have a 1 in that position, add an additional move because we'll just need to flip some number's bit at that position to a 1 - any number will do

	// For each bit that's a 0 in k, linear search through all numbers in nums, and if an odd number of elements have a 1 in that position, we need to flip it to a zero, which is an additional move
	// Otherwise we need no additional move

	moves := 0
	for i := 0; i < 32; i++ {
		if (1<<i)&k == (1 << i) {
			// k has a 1 in this position
			count := 0
			for _, n := range nums {
				if (1<<i)&n == (1 << i) {
					count++
				}
			}
			if count%2 == 0 {
				moves++
			}
		} else {
			// k has a 0 in this position
			// k has a 1 in this position
			count := 0
			for _, n := range nums {
				if (1<<i)&n == (1 << i) {
					count++
				}
			}
			moves += count % 2
		}
	}

	return moves
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
In the "100 game" two players take turns adding, to a running total, any integer from 1 to 10. The player who first causes the running total to reach or exceed 100 wins.

What if we change the game so that players cannot re-use integers?

For example, two players might take turns drawing from a common pool of numbers from 1 to 15 without replacement until they reach a total >= 100.

Given two integers maxChoosableInteger and desiredTotal, return true if the first player to move can force a win, otherwise, return false. Assume both players play optimally.

Link:
https://leetcode.com/problems/can-i-win/
*/
func canIWin(maxChoosableInteger int, desiredTotal int) bool {
	if maxChoosableInteger*(maxChoosableInteger+1)/2 < desiredTotal {
		// The total number of all available integers does not meet desiredTotal - NOBODY can win
		return false
	}

	sols := make(map[int]map[int]bool)
	available := 0 // We will represent our available numbers with a bit string
	for i := 0; i < maxChoosableInteger; i++ {
		available += (1 << i)
	}

	return topDownCanIWin(available, maxChoosableInteger, desiredTotal, sols)
}

/*
Top down recursive helper method to solve this problem
*/
func topDownCanIWin(available int, maxChoosableInteger int, desiredTotal int, sols map[int]map[int]bool) bool {
	if desiredTotal < 0 {
		return false
	} else {
		_, ok := sols[available][desiredTotal]
		if !ok {
			// We need to solve this problem - try picking each available number
			can_win := false
			for i := 0; i < maxChoosableInteger; i++ {
				pick := 1 << i
				if available&pick == pick { // We can pick this option
					value_picked := i + 1
					if value_picked >= desiredTotal {
						can_win = true
						break
					}
					new_available := available ^ pick
					if !topDownCanIWin(new_available, maxChoosableInteger, desiredTotal-value_picked, sols) {
						can_win = true
						break
					}
				}
			}
			_, ok := sols[available]
			if !ok {
				sols[available] = make(map[int]bool)
			}
			sols[available][desiredTotal] = can_win
		}
		return sols[available][desiredTotal]
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an array of distinct integers nums and a target integer target, return the number of possible combinations that add up to target.

The test cases are generated so that the answer can fit in a 32-bit integer.

Link:
https://leetcode.com/problems/combination-sum-iv/description/
*/
func combinationSum(nums []int, target int) int {

	sort.SliceStable(nums, func(i, j int) bool {
		return nums[i] < nums[j]
	})

	if target < nums[0] {
		return 0
	}

	dp := make([][]int, target+1)
	for i := 0; i <= target; i++ {
		dp[i] = make([]int, target+1)
	}
	// dp[i][j] is how many orderings of i numbers are there that produce a total of j?
	for _, val := range nums {
		if val <= target {
			dp[1][val] = 1
		} else {
			break
		}
	}

	for i := 2; i <= target; i++ {
		// i is the number of values was are letting ourselves work with
		for j := nums[0] * i; j <= target; j++ {
			// j is the target sum we want to reach - clearly cannot be lower than the lowest value in nums times i
			for _, val := range nums {
				if val < j {
					// we pick this value, which sends us to a subproblem of lower target value and one less allowed number
					dp[i][j] += dp[i-1][j-val]
				} else {
					break
				}
			}
		}
	}

	total := 0
	for i := 0; i <= target; i++ {
		total += dp[i][target]
	}

	return total
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a binary tree root, return the maximum sum of all keys of any sub-tree which is also a Binary Search Tree (BST).

Assume a BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.

Link:
https://leetcode.com/problems/maximum-sum-bst-in-binary-tree/description/
*/
func maxSumBST(root *binary_tree.TreeNode) int {
	sums := make(map[*binary_tree.TreeNode]int)
	findSums(root, sums)

	bsts := &[]*binary_tree.TreeNode{}
	isBST(root, bsts)

	max_bst_sum := 0
	for _, node := range *bsts {
		max_bst_sum = max(max_bst_sum, sums[node])
	}

	return max_bst_sum
}

// No need for DP - a tree has no cycles so we have no overlapping subproblems
func findSums(root *binary_tree.TreeNode, sums map[*binary_tree.TreeNode]int) {
	sum := root.Val
	if root.Left != nil {
		findSums(root.Left, sums)
		sum += sums[root.Left]
	}
	if root.Right != nil {
		findSums(root.Right, sums)
		sum += sums[root.Right]
	}
	sums[root] = sum
}

// Find all the BST's - O(n^2) is too slow!
/*
Returns whether this root is a binary search tree, and if so gives the lowest and highest values present
*/
func isBST(root *binary_tree.TreeNode, bsts *[]*binary_tree.TreeNode) (int, int, bool) {
	// Leaf nodes by default are trivial BSTs
	if root.Left == nil && root.Right == nil {
		*bsts = append(*bsts, root)
		return root.Val, root.Val, true
	}

	// These may change
	min_val := root.Val
	max_val := root.Val

	right_works := true
	if root.Right != nil {
		max_value_in_right, min_value_in_right, right_is_bst := isBST(root.Right, bsts)
		if !right_is_bst || (min_value_in_right <= root.Val) {
			right_works = false
		} else {
			max_val = max_value_in_right
		}
	}
	left_works := true
	if root.Left != nil {
		max_value_in_left, min_value_in_left, left_is_bst := isBST(root.Left, bsts)
		if !left_is_bst || (max_value_in_left >= root.Val) {
			left_works = false
		} else {
			min_val = min_value_in_left
		}
	}

	bst := (left_works && right_works)
	if bst {
		*bsts = append(*bsts, root)
	}

	return max_val, min_val, bst
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a positive integer n, you can apply one of the following operations:

If n is even, replace n with n / 2.
If n is odd, replace n with either n + 1 or n - 1.
Return the minimum number of operations needed for n to become 1.

Link:
https://leetcode.com/problems/integer-replacement/description/
*/
func integerReplacement(n int) int {
	sols := make(map[int]int)
	return topDownIntegerReplacement(n, sols)
}

// Recursive helper method
func topDownIntegerReplacement(n int, sols map[int]int) int {
	if n == 1 {
		return 0
	} else {
		_, ok := sols[n]
		if !ok {
			// We need to solve this problem
			if n%2 == 0 {
				sols[n] = 1 + topDownIntegerReplacement(n/2, sols)
			} else {
				sols[n] = 1 + min(topDownIntegerReplacement(n+1, sols), topDownIntegerReplacement(n-1, sols))
			}
		}
		return sols[n]
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Alice and Bob continue their games with piles of stones.  There are a number of piles arranged in a row, and each pile has a positive integer number of stones piles[i].  The objective of the game is to end with the most stones.

Alice and Bob take turns, with Alice starting first.  Initially, M = 1.

On each player's turn, that player can take all the stones in the first X remaining piles, where 1 <= X <= 2M.  Then, we set M = max(M, X).

The game continues until all the stones have been taken.

Assuming Alice and Bob play optimally, return the maximum number of stones Alice can get.

Link:
https://leetcode.com/problems/stone-game-ii/description/
*/
func stoneGameII(piles []int) int {
	sols := make([][]int, len(piles))
	sums := make([][]int, len(piles))
	for i := 0; i < len(sols); i++ {
		sols[i] = make([]int, len(sols)+1)
		sums[i] = make([]int, len(sols))
		sums[i][i] = piles[i]
	}
	for row := 0; row < len(sums); row++ {
		for col := row + 1; col < len(sums[row]); col++ {
			sums[row][col] = sums[row][col-1] + piles[col]
		}
	}
	return topDownStoneGameII(0, 1, piles, sols, sums)
}

/*
Recursive helper method to solve stoneGameII
*/
func topDownStoneGameII(first_unpicked_idx int, M int, piles []int, sols [][]int, sums [][]int) int {
	if first_unpicked_idx >= len(piles) {
		// We ran out of stones to pick
		return 0
	}
	if sols[first_unpicked_idx][M] == 0 {
		// We need to solve this problem
		if first_unpicked_idx == len(piles)-1 {
			sols[first_unpicked_idx][M] = piles[first_unpicked_idx]
		} else {
			// We can take all stones in the first X remaining piles, where 1 <= X <= 2M
			min_num_piles := 1
			max_num_piles := min(2*M, len(piles)-first_unpicked_idx)
			if max_num_piles == 0 {
				sols[first_unpicked_idx][M] = 0
			} else {
				record := math.MinInt
				for num_pick := min_num_piles; num_pick <= max_num_piles; num_pick++ {
					// If we pick these many piles, the total we get is the piles' sum, PLUS all the remaining piles, MINUS the maximum number of piles our opponent can pick up if we do this
					new_M := max(M, num_pick)
					pick_total := sums[first_unpicked_idx][first_unpicked_idx+num_pick-1]
					rest_stones_total := 0
					if first_unpicked_idx+num_pick < len(piles) {
						rest_stones_total += sums[first_unpicked_idx+num_pick][len(piles)-1]
					}
					record = max(record, pick_total+rest_stones_total-topDownStoneGameII(first_unpicked_idx+num_pick, new_M, piles, sols, sums))
				}
				sols[first_unpicked_idx][M] = record
			}
		}
	}
	return sols[first_unpicked_idx][M]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Alice and Bob continue their games with piles of stones. There are several stones arranged in a row, and each stone has an associated value which is an integer given in the array stoneValue.

Alice and Bob take turns, with Alice starting first. On each player's turn, that player can take 1, 2, or 3 stones from the first remaining stones in the row.

The score of each player is the sum of the values of the stones taken. The score of each player is 0 initially.

The objective of the game is to end with the highest score, and the winner is the player with the highest score and there could be a tie. The game continues until all the stones have been taken.

Assume Alice and Bob play optimally.

Return "Alice" if Alice will win, "Bob" if Bob will win, or "Tie" if they will end the game with the same score.

Link:
https://leetcode.com/problems/stone-game-iii/description/
*/
func stoneGameIII(stoneValue []int) string {
	sols := make([]int, len(stoneValue)+1)
	for i := 0; i < len(stoneValue); i++ {
		sols[i] = math.MinInt
	}
	res := topDownStoneGameIII(stoneValue, 0, sols)
	if res == 0 {
		return "Tie"
	} else if res > 0 {
		return "Alice"
	} else {
		return "Bob"
	}
}

/*
Top-down recursive helper method to solve the stone game problem
*/
func topDownStoneGameIII(stoneValue []int, start int, sols []int) int {
	if sols[start] == math.MinInt { // We need to solve this problem
		// Try all possible stone pickings
		record := math.MinInt
		if start < len(stoneValue)-2 {
			// Try picking 3 stones, get their total, and SUBTRACT whatever the opponent can win by given the rest
			record = max(record, stoneValue[start]+stoneValue[start+1]+stoneValue[start+2]-topDownStoneGameIII(stoneValue, start+3, sols))
		}
		if start < len(stoneValue)-1 {
			// Try picking 2 stones, and do the same thing
			record = max(record, stoneValue[start]+stoneValue[start+1]-topDownStoneGameIII(stoneValue, start+2, sols))
		}
		// We KNOW we can pick 1 stone, because in the above function we made sols[len(sols)-1] = 0, and so a starting point of len(sols)-1 would NEVER end up in this conditional
		record = max(record, stoneValue[start]-topDownStoneGameIII(stoneValue, start+1, sols))
		sols[start] = record
	}
	return sols[start]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Alice and Bob take turns playing a game, with Alice starting first.

Initially, there are n stones in a pile. On each player's turn, that player makes a move consisting of removing any non-zero square number of stones in the pile.

Also, if a player cannot make a move, he/she loses the game.

Given a positive integer n, return true if and only if Alice wins the game otherwise return false, assuming both players play optimally.

Link:
https://leetcode.com/problems/stone-game-iv/description/
*/
func winnerSquareGame(n int) bool {
	// First we need to generate a list of perfect squares less than or equal to n
	squares := []int{}
	for i := 1; i <= n; i++ {
		if i*i <= n {
			squares = append(squares, i*i)
		} else {
			break
		}
	}

	can_win := make(map[int]bool)
	for _, square := range squares {
		can_win[square] = true
	}
	return canWinSquareGame(n, can_win, squares)
}

func canWinSquareGame(n int, can_win map[int]bool, squares []int) bool {
	_, ok := can_win[n]
	if !ok {
		// Need to solve this subproblem
		// Try picking every possible perfect square less than n (n is clearly not a perfect square or it would already be in our map)
		able_to_win := false
		for _, square := range squares {
			if square > n {
				break
			}
			if !canWinSquareGame(n-square, can_win, squares) {
				able_to_win = true
				break
			}
		}
		can_win[n] = able_to_win
	}

	return can_win[n]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
There are several stones arranged in a row, and each stone has an associated value which is an integer given in the array stoneValue.

In each round of the game, Alice divides the row into two non-empty rows (i.e. left row and right row), then Bob calculates the value of each row which is the sum of the values of all the stones in this row. Bob throws away the row which has the maximum value, and Alice's score increases by the value of the remaining row. If the value of the two rows are equal, Bob lets Alice decide which row will be thrown away. The next round starts with the remaining row.

The game ends when there is only one stone remaining. Alice's is initially zero.

Return the maximum score that Alice can obtain.

Link:
https://leetcode.com/problems/stone-game-v/description/
*/
func stoneGameV(stoneValue []int) int {
	if len(stoneValue) == 1 {
		return 0
	}

	sums := make([][]int, len(stoneValue))
	sols := make([][]int, len(stoneValue))
	for i := 0; i < len(sums); i++ {
		sums[i] = make([]int, len(stoneValue))
		sums[i][i] = stoneValue[i]
		sols[i] = make([]int, len(sols))
		if i < len(sums)-1 {
			sols[i][i+1] = min(stoneValue[i], stoneValue[i+1])
		}
	}
	for row := 0; row < len(sums); row++ {
		for col := row + 1; col < len(sums); col++ {
			sums[row][col] = sums[row][col-1] + stoneValue[col]
		}
	}

	return topDownStoneGameV(0, len(stoneValue)-1, sols, sums)
}

/*
Recursive helper method to solve the stone game v
*/
func topDownStoneGameV(start int, end int, sols [][]int, sums [][]int) int {
	if sols[start][end] == 0 {
		// We need to solve this problem
		record := 0
		// Try each possible split
		for split := start; split < end; split++ {
			left_sum := sums[start][split]
			right_sum := sums[split+1][end]
			if left_sum < right_sum {
				// We get left half
				record = max(record, left_sum+topDownStoneGameV(start, split, sols, sums))
			} else if left_sum > right_sum {
				// We get right half
				record = max(record, right_sum+topDownStoneGameV(split+1, end, sols, sums))
			} else {
				// Equal sums - we get to pick
				record = max(record, left_sum+max(topDownStoneGameV(start, split, sols, sums), topDownStoneGameV(split+1, end, sols, sums)))
			}
		}
		sols[start][end] = record
	}

	return sols[start][end]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Alice and Bob take turns playing a game, with Alice starting first.

There are n stones in a pile. On each player's turn, they can remove a stone from the pile and receive points based on the stone's value. Alice and Bob may value the stones differently.

You are given two integer arrays of length n, aliceValues and bobValues. Each aliceValues[i] and bobValues[i] represents how Alice and Bob, respectively, value the ith stone.

The winner is the person with the most points after all the stones are chosen. If both players have the same amount of points, the game results in a draw. Both players will play optimally. Both players know the other's values.

Determine the result of the game, and:

If Alice wins, return 1.
If Bob wins, return -1.
If the game results in a draw, return 0.

Link:
https://leetcode.com/problems/stone-game-vi/description/
*/
func stoneGameVI(aliceValues []int, bobValues []int) int {
	// FROM THE HINT - we want to take greedily based on the sum of Alice's and Bob's values of a stone
	sums := make([][]int, len(aliceValues))
	for i := 0; i < len(sums); i++ {
		// Remember Alice and Bob's individual values as well
		sums[i] = []int{aliceValues[i] + bobValues[i], aliceValues[i], bobValues[i]}
	}
	sort.SliceStable(sums, func(i, j int) bool {
		return sums[i][0] > sums[j][0]
	})

	alice_sum := 0
	bob_sum := 0
	for i := 0; i < len(sums); i += 2 {
		alice_sum += sums[i][1]
		if i < len(sums)-1 {
			bob_sum += sums[i+1][2]
		}
	}

	if alice_sum > bob_sum {
		return 1
	} else if alice_sum < bob_sum {
		return -1
	} else {
		return 0
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Alice and Bob take turns playing a game, with Alice starting first.

There are n stones arranged in a row.
On each player's turn, they can remove either the leftmost stone or the rightmost stone from the row and receive points equal to the sum of the remaining stones' values in the row.
The winner is the one with the higher score when there are no stones left to remove.

Bob found that he will always lose this game (poor Bob, he always loses), so he decided to minimize the score's difference.
Alice's goal is to maximize the difference in the score.

Given an array of integers stones where stones[i] represents the value of the ith stone from the left, return the difference in Alice and Bob's score if they both play optimally.

Link:
https://leetcode.com/problems/stone-game-vii/description/
*/
func stoneGameVII(stones []int) int {
	sums := make([][]int, len(stones))
	for i := 0; i < len(sums); i++ {
		sums[i] = make([]int, len(stones))
		sums[i][i] = stones[i]
	}
	for row := 0; row < len(stones); row++ {
		for col := row + 1; col < len(stones); col++ {
			sums[row][col] = sums[row][col-1] + stones[col]
		}
	}

	alice_sols := make(map[int]map[int]int)
	bob_sols := make(map[int]map[int]int)
	return topDownStoneGameVIIAlice(0, len(stones)-1, stones, sums, alice_sols, bob_sols)
}

/*
Maximizer top-down helper
*/
func topDownStoneGameVIIAlice(left int, right int, stones []int, sums [][]int, alice_sols map[int]map[int]int, bob_sols map[int]map[int]int) int {
	_, ok := alice_sols[left]
	if !ok {
		alice_sols[left] = make(map[int]int)
	}
	_, ok = alice_sols[left][right]
	if !ok {
		// We need to solve this problem
		if left == right {
			// Base case - if left == right, the answer is just zero - remove that value and nothing is left
			alice_sols[left][right] = 0
		} else {
			// Try picking the left value, and try picking the right value, and the difference in scores INCREASES by whatever sum Alice picks since she goes first
			pick_left_difference := sums[left+1][right] + topDownStoneGameVIIBob(left+1, right, stones, sums, alice_sols, bob_sols)
			pick_right_difference := sums[left][right-1] + topDownStoneGameVIIBob(left, right-1, stones, sums, alice_sols, bob_sols)
			alice_sols[left][right] = max(pick_left_difference, pick_right_difference)
		}
	}

	return alice_sols[left][right]
}

/*
Minimizer top-down helper
*/
func topDownStoneGameVIIBob(left int, right int, stones []int, sums [][]int, alice_sols map[int]map[int]int, bob_sols map[int]map[int]int) int {
	_, ok := bob_sols[left]
	if !ok {
		bob_sols[left] = make(map[int]int)
	}
	_, ok = bob_sols[left][right]
	if !ok {
		// We need to solve this problem
		if left == right {
			// Base case - if left == right, the answer is just zero - remove that value and nothing is left
			bob_sols[left][right] = 0
		} else {
			// Try picking the left value, and try picking the right value, and the difference in scores DECREASES by whatever sum Bob picks since he goes second
			pick_left_difference := topDownStoneGameVIIAlice(left+1, right, stones, sums, alice_sols, bob_sols) - sums[left+1][right]
			pick_right_difference := topDownStoneGameVIIAlice(left, right-1, stones, sums, alice_sols, bob_sols) - sums[left][right-1]
			bob_sols[left][right] = min(pick_left_difference, pick_right_difference)
		}
	}

	return bob_sols[left][right]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Alice and Bob take turns playing a game, with Alice starting first.

There are n stones arranged in a row. On each player's turn, while the number of stones is more than one, they will do the following:

- Choose an integer x > 1, and remove the leftmost x stones from the row.
- Add the sum of the removed stones' values to the player's score.
- Place a new stone, whose value is equal to that sum, on the left side of the row.
- The game stops when only one stone is left in the row.

The score difference between Alice and Bob is (Alice's score - Bob's score). Alice's goal is to maximize the score difference, and Bob's goal is the minimize the score difference.

Given an integer array stones of length n where stones[i] represents the value of the ith stone from the left, return the score difference between Alice and Bob if they both play optimally.

Link:
https://leetcode.com/problems/stone-game-viii/description/

Inspiration:
https://leetcode.com/problems/stone-game-viii/solutions/1224658/python-cumulative-sums-oneliner-explained/
*/
func stoneGameVIII(stones []int) int {
	sums := make([]int, len(stones)-1)
	sums[0] = stones[0] + stones[1]
	// Given a,b,c,d,e
	// We need a+b, a+b+c, a+b+c+d, a+b+c+d+e
	for i := 1; i < len(sums); i++ {
		sums[i] = sums[i-1] + stones[i+1]
	}
	// Now we play the optimization game - if we pick a certain a certain pile sum, then our opponent can follow that up with any later pile sum.

	// If we take the sum with index i, how much can we win by?
	record := sums[len(sums)-1] // If we are ONLY allowed to pick up the last sum, that means all but one stone has been removed (concatenated into the left stone), and we just have to grab both of those stones - no choice.
	for i := len(sums) - 2; i >= 0; i-- {
		// Say we don't immediately pick up all stones - we only pick up the first (i+2) stones
		// What's the best we can achieve if we do that?
		// Well, that's just going to be that sum of stone values, MINUS the previous record which is what our opponent will be stuck with doing.
		record = max(record, sums[i]-record)
	}
	return record
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Alice and Bob continue their games with stones.
There is a row of n stones, and each stone has an associated value.
You are given an integer array stones, where stones[i] is the value of the ith stone.

Alice and Bob take turns, with Alice starting first.
On each turn, the player may remove any stone from stones.
The player who removes a stone loses if the sum of the values of all removed stones is divisible by 3.
Bob will win automatically if there are no remaining stones (even if it is Alice's turn).

Assuming both players play optimally, return true if Alice wins and false if Bob wins.

Link:
https://leetcode.com/problems/stone-game-ix/description/
*/
func stoneGameIX(stones []int) bool {
	num_0 := 0
	num_1 := 0
	num_2 := 0
	for _, stone := range stones {
		mod := stone % 3
		if mod == 0 {
			num_0++
		} else if mod == 1 {
			num_1++
		} else {
			num_2++
		}
	}
	// If at any point a person removes a stone, and that leaves (removed_1 + 2*removed_2) % 3 == 0, then that person loses
	// What's Alice going to start with? She could try removing a 1 or she could try removing a 2
	if (num_1 == 0 && num_2 == 0) || (num_1+num_2 < 2) || (num_1+2*num_2 < 3) {
		// Alice is screwed
		return false
	} else if num_1 == 1 && num_2 == 1 {
		// Bob is screwed unless num_0 saves him
		return num_0%2 == 0
	} else {
		// Can each player avoid making (removed_1 + 2*removed_2) % 3 == 0?
		// Assuming we were at 1, we CAN'T pick 2. We must pick a 3 or a 1.
		// Assuming we were at 2, we CAN'T pick 1. We must pick a 3 or a 2.
		// If we don't think about the 3's yet, that's this kind of picking sequence:
		// 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, ...
		// If we end in a 1, that means num_1 = num_2 + 2
		// If we end in a 2, that means num_1 = num_2 + 1
		// OR
		// 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, ...
		// If we end in a 2, that means num_2 = num_1 + 2
		// If we end in a 1, that means num_2 = num_1 + 1
		if !(num_1 == num_2+2 || num_1 == num_2+1 || num_2 == num_1+2 || num_2 == num_1+1) {
			// Then it won't come down to 3's - the 1's and 2's will eventually total something divisible by 3
			// Who's going to be the unlucky one?
			// Play as if there are no num_0's.
			// Then if there are an even number of num_0's the result holds.
			// If there are an odd number of num_0's the result switches.
			// SOMEONE is going to have to pick up a stone that will break the pattern - who?
			if num_1 >= 1 {
				// Alice can try starting a 1,1,2,1,2,1,...
				if num_2 >= num_1 {
					// Ends in a 2,2 - an even number of stones at that point, so Bob loses
					// UNLESS num_0 is odd
					if num_0%2 == 0 {
						return true
					}
				} else {
					// Ends in a 1,1 - that implies an odd number of stones
					// Alice loses unless there are an odd number of num_0's
					if num_0%2 == 1 {
						return true
					}
				}
			}
			if num_2 >= 1 {
				// Alice can try starting a 2,2,1,2,1,2,...
				if num_1 >= num_2 {
					// Ends in a 1,1 - an even number of stones at that point, so Bob loses
					// UNLESS num_0 is odd
					if num_0%2 == 0 {
						return true
					}
				} else {
					// Ends in a 2,2 - that implies an odd number of stones
					// Alice loses unless there are an odd number of num_0's
					if num_0%2 == 1 {
						return true
					}
				}
			}
			// Bob wins
			return false
		} else {
			// Then it COULD come down to 3's and eventually the stones will run out, so Bob would win
			// However, maybe that's if we have a 2,2,1,2,1,2,1,... pattern
			// Maybe Alice can mess that up with a 1,1,2,1,... starter pattern and screw Bob over
			if num_1 > num_2 {
				// Then Alice should see what she can do with a 2,2,1,2,1,2,1,... pattern
				// Someone will be forced to hit a 1,1, which happens on an even number, screwing over Bob
				if num_0%2 == 0 {
					// Bob can't counter
					return true
				}
			} else if num_2 > num_1 {
				// Similarly, Alice should see if she can win with a 1,1,2,1,2,1,... pattern
				if num_0%2 == 0 {
					// Again, Bob can't counter
					return true
				}
			}
			return false
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given an array people where people[i] is the weight of the ith person, and an infinite number of boats where each boat can carry a maximum weight of limit.
Each boat carries at most two people at the same time, provided the sum of the weight of those people is at most limit.

Return the minimum number of boats to carry every given person.

Link:
https://leetcode.com/problems/boats-to-save-people/description/?envType=daily-question&envId=2024-05-04
*/
func numRescueBoats(people []int, limit int) int {
	if len(people) == 1 {
		return 1
	}
	sort.SliceStable(people, func(i, j int) bool {
		return people[i] < people[j]
	})
	// Combine the people into doubles for as long as we can
	// Find the heavieset person who weighs less than the limit of the boat
	heaviest_person_less := algorithm.BinarySearchMeetOrLower(people, limit-1)
	if heaviest_person_less == -1 {
		// NO ONE is lighter than the boat capacity
		return len(people)
	} else {
		// All the people who need their own boat first of all
		boats := len(people) - heaviest_person_less - 1

		// Now for everyone else, greedily combine heaviest people with lighest people if we can
		heavier := heaviest_person_less
		lighter := 0
		for heavier > lighter {
			if people[heavier]+people[lighter] <= limit {
				// They can go on their own boat together
				heavier--
				lighter++
			} else {
				// The heavier person will DEFINITELY need their own boat
				heavier--
			}
			boats++
		}
		if heavier == lighter {
			// One last person
			boats++
		}

		return boats
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given the head of a non-empty linked list representing a non-negative integer without leading zeroes.

Return the head of the linked list after doubling it.

Link:
https://leetcode.com/problems/double-a-number-represented-as-a-linked-list/description/?envType=daily-question&envId=2024-05-07
*/
func doubleIt(head *list_node.ListNode) *list_node.ListNode {
	node_values := []int{}
	current := head
	for current != nil {
		node_values = append(node_values, current.Val)
		current = current.Next
	}

	carry := 0
	for i := len(node_values) - 1; i >= 0; i-- {
		sum := 2*node_values[i] + carry
		carry = sum / 10
		val := sum % 10
		node_values[i] = val
	}
	to_return := head
	if carry > 0 {
		to_return = &list_node.ListNode{Val: carry}
		to_return.Next = head
	}
	current = head
	for i := 0; i < len(node_values); i++ {
		current.Val = node_values[i]
		current = current.Next
	}

	return to_return
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given an array happiness of length n, and a positive integer k.

There are n children standing in a queue, where the ith child has happiness value happiness[i]. You want to select k children from these n children in k turns.

In each turn, when you select a child, the happiness value of all the children that have not been selected till now decreases by 1. Note that the happiness value cannot become negative and gets decremented only if it is positive.

Return the maximum sum of the happiness values of the selected children you can achieve by selecting k children.

Link:
https://leetcode.com/problems/maximize-happiness-of-selected-children/description/?envType=daily-question&envId=2024-05-09
*/
func maximumHappinessSum(happiness []int, k int) int64 {
	sort.SliceStable(happiness, func(i, j int) bool {
		return happiness[i] > happiness[j]
	})

	total := int64(0)
	for round := 0; round < k; round++ {
		total += max(int64(0), int64(happiness[round]-round))
	}

	return int64(total)
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given a sorted integer array arr containing 1 and prime numbers, where all the integers of arr are unique. You are also given an integer k.

For every i and j where 0 <= i < j < arr.length, we consider the fraction arr[i] / arr[j].

Return the kth smallest fraction considered. Return your answer as an array of integers of size 2, where answer[0] == arr[i] and answer[1] == arr[j].

Link:
https://leetcode.com/problems/k-th-smallest-prime-fraction/description/?envType=daily-question&envId=2024-05-10
*/
func kthSmallestPrimeFraction(arr []int, k int) []int {
	fractions := make(map[float64][]int)
	floats := heap.NewMinHeap[float64]()
	for i := 0; i < len(arr)-1; i++ {
		// By the constraints of k (<= len(arr) choose 2), we know the numerator will ALWAYS be less than the denominator for fractions 1 through k
		for j := i; j < len(arr); j++ {
			val := float64(float64(arr[i]) / float64(arr[j]))
			floats.Insert(val)
			fractions[val] = []int{arr[i], arr[j]}
		}
	}

	extracted := 0
	for extracted < k-1 {
		floats.Extract()
		extracted++
	}
	// Now for the kth fraction
	return fractions[floats.Extract()]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
There are n workers. You are given two integer arrays quality and wage where quality[i] is the quality of the ith worker and wage[i] is the minimum wage expectation for the ith worker.

We want to hire exactly k workers to form a paid group. To hire a group of k workers, we must pay them according to the following rules:

Every worker in the paid group must be paid at least their minimum wage expectation.
In the group, each worker's pay must be directly proportional to their quality. This means if a worker’s quality is double that of another worker in the group, then they must be paid twice as much as the other worker.
Given the integer k, return the least amount of money needed to form a paid group satisfying the above conditions. Answers within 10-5 of the actual answer will be accepted.

Link:
https://leetcode.com/problems/minimum-cost-to-hire-k-workers/description/?envType=daily-question&envId=2024-05-11
*/
func mincostToHireWorkers(quality []int, wage []int, k int) float64 {
	sort_by_wage_over_quality := make([][]float64, len(quality))
	for i := 0; i < len(quality); i++ {
		sort_by_wage_over_quality[i] = []float64{float64(wage[i]), float64(quality[i])}
	}
	sort.SliceStable(sort_by_wage_over_quality, func(i, j int) bool {
		return (sort_by_wage_over_quality[i][0] / sort_by_wage_over_quality[i][1]) < (sort_by_wage_over_quality[j][0] / sort_by_wage_over_quality[j][1])
	})

	lowest_quality := heap.NewCustomMaxHeap[[]float64](
		func(first, second []float64) bool {
			return first[1] > second[1]
		},
	)

	quality_sum := float64(0)
	for i := 0; i < k; i++ {
		lowest_quality.Insert(sort_by_wage_over_quality[i])
		quality_sum += sort_by_wage_over_quality[i][1]
	}
	ratio := sort_by_wage_over_quality[k-1][0] / sort_by_wage_over_quality[k-1][1]
	record := quality_sum * ratio
	for i := k; i < len(sort_by_wage_over_quality); i++ {
		ratio = sort_by_wage_over_quality[i][0] / sort_by_wage_over_quality[i][1]
		quality_sum -= lowest_quality.Extract()[1]
		lowest_quality.Insert(sort_by_wage_over_quality[i])
		quality_sum += sort_by_wage_over_quality[i][1]
		price := quality_sum * ratio
		record = math.Min(record, price)
	}

	return record
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given an m x n binary matrix grid.

A move consists of choosing any row or column and toggling each value in that row or column (i.e., changing all 0's to 1's, and all 1's to 0's).

Every row of the matrix is interpreted as a binary number, and the score of the matrix is the sum of these numbers.

Return the highest possible score after making any number of moves (including zero moves).

Link:
https://leetcode.com/problems/score-after-flipping-matrix/description/?envType=daily-question&envId=2024-05-13
*/
func matrixScore(grid [][]int) int {
	row_ints := make([]int, len(grid))
	cols := len(grid[0])
	total := 0
	for row := 0; row < len(row_ints); row++ {
		v := 0
		for col := 0; col < cols; col++ {
			v += grid[row][col] << (cols - 1 - col)
		}
		row_ints[row] = v
		total += v
	}

	// Now we are allowed to toggle any row or any column, any number of times
	// Note that since we can reflect each row, we KNOW that we can make the left column all 1's, which we obviously want to do
	// The question is - do we flip the left-most column, and then flip rows until they are all 1's? Or do we just flip rows?

	// Try flipping the left-most column, and then row-by-row get all 1's in the left-most column
	first_record := total
	for i := 0; i < len(row_ints); i++ {
		first_record -= row_ints[i]
		row_ints[i] = row_ints[i] ^ (1 << (cols - 1))
		first_record += row_ints[i]
	}
	// Now flip all the rows until you have all 1's in the left-most column
	for i := 0; i < len(row_ints); i++ {
		if row_ints[i] < (1 << (cols - 1)) {
			// Then flip the row
			first_record -= row_ints[i]
			row_ints[i] = row_ints[i] ^ ((1 << cols) - 1)
			first_record += row_ints[i]
		}
	}
	// Now look through each column, and IFF there are more 0's than 1's, flip the column
	for col := 1; col < cols; col++ {
		one_count := 0
		for r := 0; r < len(row_ints); r++ {
			if row_ints[r]&(1<<(cols-1-col)) == (1 << (cols - 1 - col)) {
				one_count++
			}
		}
		if one_count <= (len(row_ints) / 2) {
			// Then flip this column
			for j := 0; j < len(row_ints); j++ {
				first_record -= row_ints[j]
				row_ints[j] = row_ints[j] ^ (1 << (cols - 1 - col))
				first_record += row_ints[j]
			}
		}
	}

	// Now try to achieve the left most-column being all 1's ONLY by
	// Refresh our row integers
	for row := 0; row < len(row_ints); row++ {
		v := 0
		for col := 0; col < cols; col++ {
			v += grid[row][col] << (cols - 1 - col)
		}
		row_ints[row] = v
	}
	second_record := total
	// Flip all the rows until you have all 1's in the left-most column
	for i := 0; i < len(row_ints); i++ {
		if row_ints[i] < (1 << (cols - 1)) {
			// Then flip the row
			second_record -= row_ints[i]
			row_ints[i] = row_ints[i] ^ ((1 << cols) - 1)
			second_record += row_ints[i]
		}
	}
	// Now look through each column, and IFF there are more 0's than 1's, flip the column
	for col := 1; col < cols; col++ {
		one_count := 0
		for r := 0; r < len(row_ints); r++ {
			if row_ints[r]&(1<<(cols-1-col)) == (1 << (cols - 1 - col)) {
				one_count++
			}
		}
		if one_count <= (len(row_ints) / 2) {
			// Then flip this column
			for j := 0; j < len(row_ints); j++ {
				second_record -= row_ints[j]
				row_ints[j] = row_ints[j] ^ (1 << (cols - 1 - col))
				second_record += row_ints[j]
			}
		}
	}

	return max(first_record, second_record)
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
In a gold mine grid of size m x n, each cell in this mine has an integer representing the amount of gold in that cell, 0 if it is empty.

Return the maximum amount of gold you can collect under the conditions:
- Every time you are located in a cell you will collect all the gold in that cell.
- From your position, you can walk one step to the left, right, up, or down.
- You can't visit the same cell more than once.
- Never visit a cell with 0 gold.
- You can start and stop collecting gold from any position in the grid that has some gold.

Link:
https://leetcode.com/problems/path-with-maximum-gold/description/?envType=daily-question&envId=2024-05-14
*/
func getMaximumGold(grid [][]int) int {
	record := 0
	for row := 0; row < len(grid); row++ {
		for col := 0; col < len(grid[row]); col++ {
			record = max(record, exploreFromLocation(row, col, grid))
		}
	}
	return record
}

func exploreFromLocation(row int, col int, grid [][]int) int {
	if row < 0 || row >= len(grid) || col < 0 || col >= len(grid[0]) || grid[row][col] == 0 {
		return 0
	} else {
		total := grid[row][col]
		new_grid := make([][]int, len(grid))
		for r := 0; r < len(grid); r++ {
			new_grid[r] = make([]int, len(grid[r]))
			copy(new_grid[r], grid[r])
		}
		new_grid[row][col] = 0
		best_from_here := 0
		// Try up
		best_from_here = max(best_from_here, exploreFromLocation(row-1, col, new_grid))
		// Try down
		best_from_here = max(best_from_here, exploreFromLocation(row+1, col, new_grid))
		// Try left
		best_from_here = max(best_from_here, exploreFromLocation(row, col-1, new_grid))
		// Try right
		best_from_here = max(best_from_here, exploreFromLocation(row, col+1, new_grid))
		return total + best_from_here
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given a 0-indexed 2D matrix grid of size n x n, where (r, c) represents:

A cell containing a thief if grid[r][c] = 1
An empty cell if grid[r][c] = 0
You are initially positioned at cell (0, 0). In one move, you can move to any adjacent cell in the grid, including cells containing thieves.

The safeness factor of a path on the grid is defined as the minimum manhattan distance from any cell in the path to any thief in the grid.

Return the maximum safeness factor of all paths leading to cell (n - 1, n - 1).

An adjacent cell of cell (r, c), is one of the cells (r, c + 1), (r, c - 1), (r + 1, c) and (r - 1, c) if it exists.

The Manhattan distance between two cells (a, b) and (x, y) is equal to |a - x| + |b - y|, where |val| denotes the absolute value of val.

Link:
https://leetcode.com/problems/find-the-safest-path-in-a-grid/description/?envType=daily-question&envId=2024-05-15

NOTE:
I could not have figured this out without the editorial - and the binary search option was too slow - had to use Djikstra
*/
func maximumSafenessFactor(grid [][]int) int {
	if grid[0][0] == 1 {
		return 0
	}

	n := len(grid)
	thieves := [][]int{}
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			if grid[r][c] == 1 {
				thieves = append(thieves, []int{r, c})
			}
		}
	}

	visited := make([][]bool, n)
	distances := make([][]int, n)
	for i := 0; i < n; i++ {
		visited[i] = make([]bool, n)
		distances[i] = make([]int, n)
	}

	bfs := linked_list.NewQueue[[]int]()
	for _, thief := range thieves {
		visited[thief[0]][thief[1]] = true
		bfs.Enqueue(thief)
	}
	dist := 0
	for !bfs.Empty() {
		num := bfs.Length()
		for i := 0; i < num; i++ {
			posn := bfs.Dequeue()
			distances[posn[0]][posn[1]] = dist
			// Now enqueue the neighbors
			if posn[0] > 0 && !visited[posn[0]-1][posn[1]] {
				// Look up
				visited[posn[0]-1][posn[1]] = true
				bfs.Enqueue([]int{posn[0] - 1, posn[1]})
			}
			if posn[0] < n-1 && !visited[posn[0]+1][posn[1]] {
				// Look down
				visited[posn[0]+1][posn[1]] = true
				bfs.Enqueue([]int{posn[0] + 1, posn[1]})
			}
			if posn[1] > 0 && !visited[posn[0]][posn[1]-1] {
				// Look left
				visited[posn[0]][posn[1]-1] = true
				bfs.Enqueue([]int{posn[0], posn[1] - 1})
			}
			if posn[1] < n-1 && !visited[posn[0]][posn[1]+1] {
				// Look right
				visited[posn[0]][posn[1]+1] = true
				bfs.Enqueue([]int{posn[0], posn[1] + 1})
			}
		}
		dist++
	}

	// Use Djikstra's Algorithm to find the best path from the top left to bottom right
	for i := 0; i < n; i++ {
		// Refresh visited matrix
		visited[i] = make([]bool, n)
	}
	cell_heap := heap.NewCustomMaxHeap[[]int](func(first, second []int) bool {
		return distances[first[0]][first[1]] > distances[second[0]][second[1]]
	})
	visited[0][0] = true
	cell_heap.Insert([]int{0, 0})
	min_safety := math.MaxInt
	for !cell_heap.Empty() {
		next_cell := cell_heap.Extract()
		r := next_cell[0]
		c := next_cell[1]
		min_safety = min(min_safety, distances[r][c])
		if r == n-1 && c == n-1 {
			return min_safety
		}
		// Now throw the neighbors onto the heap
		if r > 0 && !visited[r-1][c] {
			// Look up
			visited[r-1][c] = true
			cell_heap.Insert([]int{r - 1, c})
		}
		if r < n-1 && !visited[r+1][c] {
			// Look down
			visited[r+1][c] = true
			cell_heap.Insert([]int{r + 1, c})
		}
		if c > 0 && !visited[r][c-1] {
			// Look left
			visited[r][c-1] = true
			cell_heap.Insert([]int{r, c - 1})
		}
		if c < n-1 && !visited[r][c+1] {
			// Look right
			visited[r][c+1] = true
			cell_heap.Insert([]int{r, c + 1})
		}
	}

	return 0
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a binary tree root and an integer target, delete all the leaf nodes with value target.

Note that once you delete a leaf node with value target, if its parent node becomes a leaf node and has the value target, it should also be deleted (you need to continue doing that until you cannot).

Link:
https://leetcode.com/problems/delete-leaves-with-a-given-value/description/?envType=daily-question&envId=2024-05-17
*/
func removeLeafNodes(root *binary_tree.TreeNode, target int) *binary_tree.TreeNode {
	parents := make(map[*binary_tree.TreeNode]*binary_tree.TreeNode)
	leaves := &[]*binary_tree.TreeNode{}
	traverseTree(root, parents, leaves)

	for _, leaf := range *leaves {
		deleteLeaf(leaf, target, parents)
	}

	if root.Left == nil && root.Right == nil && root.Val == target {
		return nil
	} else {
		return root
	}
}

/*
Helper function to create the map of nodes to their parents and make a list of root nodes
*/
func traverseTree(root *binary_tree.TreeNode, parents map[*binary_tree.TreeNode]*binary_tree.TreeNode, leaves *[]*binary_tree.TreeNode) {
	if root.Left == nil && root.Right == nil {
		*leaves = append(*leaves, root)
	} else {
		if root.Left != nil {
			parents[root.Left] = root
			traverseTree(root.Left, parents, leaves)
		}
		if root.Right != nil {
			parents[root.Right] = root
			traverseTree(root.Right, parents, leaves)
		}
	}
}

/*
Helper function to delete the leaf of a binary tree if it has the given value
*/
func deleteLeaf(leaf *binary_tree.TreeNode, target int, parents map[*binary_tree.TreeNode]*binary_tree.TreeNode) {
	if leaf.Val == target {
		parent, ok := parents[leaf]
		if ok {
			if parent.Left == leaf {
				parent.Left = nil
			} else {
				parent.Right = nil
			}
			if parent.Left == nil && parent.Right == nil {
				deleteLeaf(parent, target, parents)
			}
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given the root of a binary tree. We install cameras on the tree nodes where each camera at a node can monitor its parent, itself, and its immediate children.

Return the minimum number of cameras needed to monitor all nodes of the tree.

Link:
https://leetcode.com/problems/binary-tree-cameras/description/
*/
func minCameraCover(root *binary_tree.TreeNode) int {
	put_camera := make(map[*binary_tree.TreeNode]int)
	no_camera := make(map[*binary_tree.TreeNode]int)
	if isLeaf(root) { // We plain have to have a camera
		return 1
	} else { // Try putting a camera on the root, and try not putting a camera on the root
		return min(topDownPutCamera(root, put_camera, no_camera), topDownNoCamera(root, put_camera, no_camera))
	}
}

func topDownPutCamera(root *binary_tree.TreeNode, put_camera map[*binary_tree.TreeNode]int, no_camera map[*binary_tree.TreeNode]int) int {
	_, ok := put_camera[root]
	if !ok {
		// Need to solve this problem
		cameras := 1
		if root.Right != nil && !isLeaf(root.Right) {
			// Think about the right side
			// We can try giving the right a camera
			record := topDownPutCamera(root.Right, put_camera, no_camera)
			// We can try covering the right WITHOUT giving it a camera
			record = min(record, topDownNoCamera(root.Right, put_camera, no_camera))
			// We can try not covering the right since the root's camera covers it, and making sure the right's children are covered
			other_record := 0
			if root.Right.Left != nil {
				if isLeaf(root.Right.Left) {
					other_record++
				} else {
					other_record += min(topDownPutCamera(root.Right.Left, put_camera, no_camera), topDownNoCamera(root.Right.Left, put_camera, no_camera))
				}
			}
			if root.Right.Right != nil {
				if isLeaf(root.Right.Right) {
					other_record++
				} else {
					other_record += min(topDownPutCamera(root.Right.Right, put_camera, no_camera), topDownNoCamera(root.Right.Right, put_camera, no_camera))
				}
			}
			record = min(record, other_record)
			cameras += record
		}
		if root.Left != nil && !isLeaf(root.Left) {
			// Think about the left side
			record := topDownPutCamera(root.Left, put_camera, no_camera)
			record = min(record, topDownNoCamera(root.Left, put_camera, no_camera))
			other_record := 0
			if root.Left.Left != nil {
				if isLeaf(root.Left.Left) {
					other_record++
				} else {
					other_record += min(topDownPutCamera(root.Left.Left, put_camera, no_camera), topDownNoCamera(root.Left.Left, put_camera, no_camera))
				}
			}
			if root.Left.Right != nil {
				if isLeaf(root.Left.Right) {
					other_record++
				} else {
					other_record += min(topDownPutCamera(root.Left.Right, put_camera, no_camera), topDownNoCamera(root.Left.Right, put_camera, no_camera))
				}
			}
			record = min(record, other_record)
			cameras += record
		}
		put_camera[root] = cameras
	}
	return put_camera[root]
}

func topDownNoCamera(root *binary_tree.TreeNode, put_camera map[*binary_tree.TreeNode]int, no_camera map[*binary_tree.TreeNode]int) int {
	_, ok := no_camera[root]
	if !ok {
		// Need to solve this problem
		cameras := 0 // For root nodes this would return 0, but we will NEVER call this function on a root node
		if root.Right == nil {
			// Left is not nil
			// Camera must go on the left
			cameras += topDownPutCamera(root.Left, put_camera, no_camera)
		} else if root.Left == nil {
			// Right it not nil
			// Camera must go on the right
			cameras += topDownPutCamera(root.Right, put_camera, no_camera)
		} else {
			// Both children are not nil
			// For each child, if they are a root, they HAVE to have a camera
			// Otherwise, they get to pick if they have a camera
			// AT LEAST one of the children must have a camera regardless
			if isLeaf(root.Left) && isLeaf(root.Right) {
				cameras += 2
			} else if isLeaf(root.Left) {
				// Right child has a choice on if it gets a camera
				cameras += 1 + min(topDownPutCamera(root.Right, put_camera, no_camera), topDownNoCamera(root.Right, put_camera, no_camera))
			} else if isLeaf(root.Right) {
				// Left child has a choice on if it gets a camera
				cameras += 1 + min(topDownPutCamera(root.Left, put_camera, no_camera), topDownNoCamera(root.Left, put_camera, no_camera))
			} else {
				// At least one child must have a camera, but as long as one of them has a camera, the other can choose whether or not it gets a camera
				// Try putting a camera on the left child and giving the right child freedom
				additional := topDownPutCamera(root.Left, put_camera, no_camera) + min(topDownPutCamera(root.Right, put_camera, no_camera), topDownNoCamera(root.Right, put_camera, no_camera))
				// Try putting a camera on the right child and giving the left child freedom
				additional = min(additional, topDownPutCamera(root.Right, put_camera, no_camera)+min(topDownPutCamera(root.Left, put_camera, no_camera), topDownNoCamera(root.Left, put_camera, no_camera)))
				cameras += additional
			}
		}
		no_camera[root] = cameras
	}
	return no_camera[root]
}

/*
Helper function to determine if a node is a leaf node
*/
func isLeaf(root *binary_tree.TreeNode) bool {
	return root.Left == nil && root.Right == nil
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
There exists an undirected tree with n nodes numbered 0 to n - 1. You are given a 0-indexed 2D integer array edges of length n - 1, where edges[i] = [ui, vi] indicates that there is an edge between nodes ui and vi in the tree. You are also given a positive integer k, and a 0-indexed array of non-negative integers nums of length n, where nums[i] represents the value of the node numbered i.

Alice wants the sum of values of tree nodes to be maximum, for which Alice can perform the following operation any number of times (including zero) on the tree:

Choose any edge [u, v] connecting the nodes u and v, and update their values as follows:
nums[u] = nums[u] XOR k
nums[v] = nums[v] XOR k
Return the maximum possible sum of the values Alice can achieve by performing the operation any number of times.

Link:
https://leetcode.com/problems/find-the-maximum-sum-of-node-values/description/?envType=daily-question&envId=2024-05-19
*/
func maximumValueSum(nums []int, k int, edges [][]int) int64 {
	n := len(nums)

	// From the hint:
	// Let dp[b][i] be the best subtree sum we can achieve starting at node i, given that we either HAVE XORed the edge to the parent (b=1) or have not (b=0)
	// This is a tree, so we can make any node the root node
	dp := make([][]int64, 2)
	dp[0] = make([]int64, n)
	dp[1] = make([]int64, n)
	for i := 0; i < n; i++ {
		dp[0][i] = int64(math.MinInt64)
		dp[1][i] = int64(math.MinInt64)
	}

	// Determine how much each of our nodes changes if we xor it
	xors := make([]int, n)
	for idx, val := range nums {
		xors[idx] = val ^ k
	}

	// Now keep track of all the connections of our nodes
	// Creating a graph based off the edges will yield exactly one node not having a parent - we'll make that node our starting point
	connections := make([][]int, n)
	for _, edge := range edges {
		connections[edge[0]] = append(connections[edge[0]], edge[1])
		connections[edge[1]] = append(connections[edge[1]], edge[0])
	}

	// Find some node with only one connection - they can be our parent
	root := 0
	for idx, children := range connections {
		if len(children) == 1 {
			root = idx
			break
		}
	}

	// Now we need to solve the problem - just make the root 0
	visited := make([]bool, n)
	maximizeSum(nums, xors, root, connections, visited, dp)
	return dp[0][root] // equal to dp[1][root] because root has no parent
}

/*
Top-down helper method to maximize the sum of a sub-tree
*/
func maximizeSum(nums []int, xors []int, root int, connections [][]int, visited []bool, dp [][]int64) {
	if dp[0][root] == int64(math.MinInt64) {
		// Need to solve this problem
		visited[root] = true
		children := []int{}
		for _, connection := range connections[root] {
			// That firstly entails solving it for all children
			if !visited[connection] {
				maximizeSum(nums, xors, connection, connections, visited, dp)
				children = append(children, connection)
			}
		}
		// Sort the children such that children at the beginning benefit more from XORing
		sort.SliceStable(children, func(i, j int) bool {
			xor_i := dp[1][children[i]]
			no_xor_i := dp[0][children[i]]
			diff_i := no_xor_i - xor_i

			xor_j := dp[1][children[j]]
			no_xor_j := dp[0][children[j]]
			diff_j := no_xor_j - xor_j

			return diff_i < diff_j
		})
		// How many children benefit from xoring?
		differences := make([]int64, len(children))
		for idx, child := range children {
			xor_i := dp[1][child]
			no_xor_i := dp[0][child]
			diff_i := no_xor_i - xor_i
			differences[idx] = diff_i
		}
		num_children_want_xor := algorithm.BinarySearchMeetOrLower(differences, 0) + 1
		// First, assume that everyone who wants to be xored - including the root - gets xored - and we'll subtract as needed
		dp[0][root] = int64(max(nums[root], xors[root]))
		dp[1][root] = int64(max(nums[root], xors[root]))
		for i := range differences {
			if differences[i] >= 0 {
				// Go with non-xoring
				dp[0][root] += dp[0][children[i]]
				dp[1][root] += dp[0][children[i]]
			} else {
				// Go with xoring
				dp[0][root] += dp[1][children[i]]
				dp[1][root] += dp[1][children[i]]
			}
		}
		if len(children) == 1 && !visited[children[0]] { // NO PARENT
			// Then dp[0][root] == dp[1][root]
			if len(children) == 0 { // No XORing possible...
				dp[0][root] = int64(nums[root])
				dp[1][root] = int64(nums[root])
			} else { // Only XORing by the children possible...
				// If we xor an even number of edges, root's value stays the same
				// Otherwise, root's value goes to its xor
				child_loss := int64(0)
				node_loss := int64(0)
				if (num_children_want_xor%2 == 0) && nums[root] < xors[root] {
					// We're going to need to decide - should we stop one of our children from getting xored when they want to, force an extra one to be xored, or force the root to NOT be xored?
					child_loss_prevent_xor := int64(math.MaxInt64)
					if num_children_want_xor > 0 {
						child_loss_prevent_xor = dp[1][children[num_children_want_xor-1]] - dp[0][children[num_children_want_xor-1]]
					}
					child_loss_force_extra_xor := int64(math.MaxInt64)
					if num_children_want_xor < len(children) {
						child_loss_force_extra_xor = dp[0][children[num_children_want_xor]] - dp[1][children[num_children_want_xor]]
					}
					child_loss = min(child_loss_prevent_xor, child_loss_force_extra_xor)
					node_loss = int64(xors[root]) - int64(nums[root])
				} else if (num_children_want_xor%2 == 1) && xors[root] < nums[root] {
					// Then decide - should we stop one of our children from getting xored, force an extra to be xored, or force our root to be xored?
					child_loss_prevent_xor := dp[1][children[num_children_want_xor-1]] - dp[0][children[num_children_want_xor-1]]
					child_loss_force_extra_xor := int64(math.MaxInt64)
					if num_children_want_xor < len(children) {
						child_loss_force_extra_xor = dp[0][children[num_children_want_xor]] - dp[1][children[num_children_want_xor]]
					}
					child_loss = min(child_loss_prevent_xor, child_loss_force_extra_xor)
					node_loss = int64(nums[root]) - int64(xors[root])
				}
				dp[0][root] -= min(child_loss, node_loss)
				dp[1][root] -= min(child_loss, node_loss)
			}
		} else { // The root does have a parent
			child_loss_prevent_xor := int64(math.MaxInt64)
			if num_children_want_xor > 0 {
				child_loss_prevent_xor = dp[1][children[num_children_want_xor-1]] - dp[0][children[num_children_want_xor-1]]
			}
			child_loss_force_extra_xor := int64(math.MaxInt64)
			if num_children_want_xor < len(children) {
				child_loss_force_extra_xor = dp[0][children[num_children_want_xor]] - dp[1][children[num_children_want_xor]]
			}
			child_loss := min(child_loss_prevent_xor, child_loss_force_extra_xor)
			node_loss := int64(max(nums[root], xors[root]) - min(nums[root], xors[root]))
			if len(children) > 0 { // XORing by the parent AND by children possible
				// Suppose we decide we WILL xor the root with its parent
				// Xoring an EVEN number of children will leave the root xored - we could add or take away a child xor to change that
				old_node_loss := node_loss
				old_child_loss := child_loss
				if (num_children_want_xor%2 == 0) && (xors[root] > nums[root]) {
					// Then no use exploring forcing the root or any children to be xored/not-xored against its wishes
					node_loss = 0
					child_loss = 0
				} else if (num_children_want_xor%2 == 1) && (nums[root] > xors[root]) {
					// Same
					node_loss = 0
					child_loss = 0
				}
				dp[1][root] -= int64(min(node_loss, child_loss))

				node_loss = old_node_loss
				child_loss = old_child_loss
				// Similarly, if we will NOT xor the root with its parent
				// Xoring an ODD number of children will leave the root xored - we could add or take away a child xor to change that
				if (num_children_want_xor%2 == 1) && (xors[root] > nums[root]) {
					// Then no use exploring forcing the root or any children to be xored/not-xored against its wishes
					node_loss = 0
					child_loss = 0
				} else if (num_children_want_xor%2 == 0) && (nums[root] > xors[root]) {
					// Same
					node_loss = 0
					child_loss = 0
				}
				dp[0][root] -= int64(min(node_loss, child_loss))
			} else { // No children
				dp[0][root] = int64(nums[root])
				dp[1][root] = int64(xors[root])
			}
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
The following data structure will be helpful in the preceding problem.
*/
type node struct {
	id              int
	cost            int
	visited         bool
	neighbors       []*node
	subtree_records []int
}

/*
You are given an undirected tree with n nodes labeled from 0 to n - 1, and rooted at node 0. You are given a 2D integer array edges of length n - 1, where edges[i] = [ai, bi] indicates that there is an edge between nodes ai and bi in the tree.

You are also given a 0-indexed integer array cost of length n, where cost[i] is the cost assigned to the ith node.

You need to place some coins on every node of the tree.
The number of coins to be placed at node i can be calculated as:

  - If size of the subtree of node i is less than 3, place 1 coin.
  - Otherwise, place an amount of coins equal to the maximum product of cost values assigned to 3 distinct nodes in the subtree of node i.
    If this product is negative, place 0 coins.
    Return an array coin of size n such that coin[i] is the number of coins placed at node i.

Link:
https://leetcode.com/problems/find-number-of-coins-to-place-in-tree-nodes/description/
*/
func placedCoins(edges [][]int, cost []int) []int64 {
	nodes := make([]*node, len(cost))
	for idx, c := range cost {
		nodes[idx] = &node{id: idx, cost: c}
	}

	for _, edge := range edges {
		first := edge[0]
		second := edge[1]
		nodes[first].neighbors = append(nodes[first].neighbors, nodes[second])
		nodes[second].neighbors = append(nodes[second].neighbors, nodes[first])
	}

	coin_assignments := &[]int64{}
	for i := 0; i < len(cost); i++ {
		(*coin_assignments) = append(*coin_assignments, -1)
	}
	assignCoins(nodes[0], coin_assignments)

	return *coin_assignments
}

/*
Helper function to solve the preceding problem recursively
*/
func assignCoins(root *node, coin_assignments *[]int64) {
	// Visit all 'children' of this node first
	root.visited = true
	lowest_values := heap.NewMinHeap[int]()
	highest_values := heap.NewMaxHeap[int]()
	lowest_values.Insert(root.cost)
	highest_values.Insert(root.cost)
	for _, n := range root.neighbors {
		if !n.visited {
			assignCoins(n, coin_assignments)
			for _, v := range n.subtree_records {
				lowest_values.Insert(v)
				highest_values.Insert(v)
			}
		}
	}

	// Obtain all the new extreme values - 3 lowest and 3 highest
	if lowest_values.Size() > 6 {
		// This will take some finagling...
		for i := 0; i < 3; i++ {
			root.subtree_records = append(root.subtree_records, lowest_values.Extract())
		}
		for i := 0; i < 3; i++ {
			root.subtree_records = append(root.subtree_records, highest_values.Extract())
		}
		// Switch the 4th and 6th elements to preserve orders
		root.subtree_records[3], root.subtree_records[5] = root.subtree_records[5], root.subtree_records[3]
	} else {
		// We just need to empty the increasing heap
		for !lowest_values.Empty() {
			root.subtree_records = append(root.subtree_records, lowest_values.Extract())
		}
	}

	// Now update the value assigned to this tree
	if len(root.subtree_records) >= 3 {
		// Several things to try
		// Top 3
		top_3 := int64(root.subtree_records[len(root.subtree_records)-1]) * int64(root.subtree_records[len(root.subtree_records)-2]) * int64(root.subtree_records[len(root.subtree_records)-3])
		// Bottom 3
		bottom_3 := int64(root.subtree_records[0]) * int64(root.subtree_records[1]) * int64(root.subtree_records[2])
		// Bottom 2 and top
		bottom_2_top := int64(root.subtree_records[0]) * int64(root.subtree_records[1]) * int64(root.subtree_records[len(root.subtree_records)-1])
		// Bottom and top 2
		bottom_top_2 := int64(root.subtree_records[0]) * int64(root.subtree_records[len(root.subtree_records)-2]) * int64(root.subtree_records[len(root.subtree_records)-1])

		(*coin_assignments)[root.id] = max(
			0,
			top_3,
			bottom_3,
			bottom_2_top,
			bottom_top_2,
		)
	} else {
		(*coin_assignments)[root.id] = 1
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
There exists an undirected and unrooted tree with n nodes indexed from 0 to n - 1.
You are given an integer n and a 2D integer array edges of length n - 1, where edges[i] = [a_i, b_i] indicates that there is an edge between nodes ai and bi in the tree.
You are also given an array coins of size n where coins[i] can be either 0 or 1, where 1 indicates the presence of a coin in the vertex i.

Initially, you choose to start at any vertex in the tree.
Then, you can perform the following operations any number of times:
- Collect all the coins that are at a distance of at most 2 from the current vertex, or
- Move to any adjacent vertex in the tree.

Find the minimum number of edges you need to go through to collect all the coins and go back to the initial vertex.

Note that if you pass an edge several times, you need to count it into the answer several times.

Link:
https://leetcode.com/problems/collect-coins-in-a-tree/description/
*/
func collectTheCoins(coins []int, edges [][]int) int {
	// The hints on LeetCode were very helpful...
	// Firstly, any leaf node (one connection) without a coin is redundant. Remove them.
	// Secondly, in the resulting graph we have our leaf nodes and MAYBE their parents (if and only if that parent is connected to <2 non-leaf nodes) to remove, because those nodes still do not need to be reached directly.
	// All remaining nodes MUST be reached, which means all such edges in that subtree (# remaining nodes - 1) must be traversed twice.
	// So once you obtain the resulting graph, RETURN max(0, num_nodes-1)*2

	graph := make(map[int]map[int]bool) // An adjacency list of children
	// Keep track of the nodes you will need to delete from adjacency lists from which nodes
	to_delete := make(map[int]map[int]bool)
	for i := 0; i <= len(edges); i++ {
		graph[i] = make(map[int]bool)
		to_delete[i] = make(map[int]bool)
	}
	for _, edge := range edges {
		graph[edge[0]][edge[1]] = true
		graph[edge[1]][edge[0]] = true
	}

	// First remove all redundant nodes - which needs to be done iteratively
	// I timed out without using this queue idea - which I got from the following source:
	// https://leetcode.com/problems/collect-coins-in-a-tree/solutions/3343497/easy-bfs-intuition-explained-o-n-tc-and-sc-trim-c-java/
	leaf_queue := linked_list.NewQueue[int]()
	for node, neighbors := range graph {
		if len(neighbors) == 1 && coins[node] == 0 {
			leaf_queue.Enqueue(node)
		}
	}
	for !leaf_queue.Empty() {
		leaf := leaf_queue.Dequeue()
		for parent := range graph[leaf] {
			// There will only be one parent
			delete(graph[parent], leaf)
			if len(graph[parent]) == 1 && coins[parent] == 0 {
				leaf_queue.Enqueue(parent)
			}
		}
		delete(graph, leaf)
	}

	// Now delete all leaf nodes and their respective parent IF AND ONLY IF that parent is connected to <2 NON-LEAF nodes
	parent_for_removal := make(map[int]bool)
	for node, neighbors := range graph {
		if len(neighbors) == 1 {
			for parent := range neighbors {
				// There will only be one parent
				// The parent MAY ALSO need to be deleted
				_, ok := parent_for_removal[parent]
				if !ok {
					// We need to see if this node should be deleted - does it have less than 2 non-node neighbors?
					non_leaves := 0
					for other_neighbor := range graph[parent] {
						if len(graph[other_neighbor]) > 1 {
							non_leaves++
							if non_leaves > 1 {
								break
							}
						}
					}
					parent_for_removal[parent] = non_leaves < 2
					// We're NOT actually going to have to remove parent from ITS remaining neighbors' lists.
					// That's because this is the last phase of removal we will do and afterwards will only care about the number of nodes.
				}
			}
			delete(graph, node)
		}
	}

	// delete all parents marked for removal
	for node, delete_it := range parent_for_removal {
		if delete_it {
			delete(graph, node)
		}
	}

	// Now return the number of remaining edges times 2
	return max(0, len(graph)-1) * 2
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given an array nums of positive integers and a positive integer k.

A subset of nums is beautiful if it does not contain two integers with an absolute difference equal to k.

Return the number of non-empty beautiful subsets of the array nums.

A subset of nums is an array that can be obtained by deleting some (possibly none) elements from nums.
Two subsets are different if and only if the chosen indices to delete are different.

Link:
https://leetcode.com/problems/the-number-of-beautiful-subsets/description/?envType=daily-question&envId=2024-05-23
*/
func beautifulSubsets(nums []int, k int) int {
	// You COULD generate all possible subsets - the input size is small enough to do this and LeetCode accepts it
	n := len(nums)
	set_counter := 0
	beautiful_count := 0
	for set_counter < ((1 << n) - 1) { // 2^n
		set_counter++
		set := make(map[int]bool)
		for i := 0; i < n; i++ {
			if ((1 << (i)) & (set_counter)) == (1 << (i)) {
				// element i IS in our current subset
				_, ok := set[nums[i]]
				if !ok {
					set[nums[i]] = true
				}
			}
		}
		// Now that we have out subset, see if it is beautiful
		beautiful := true
		for v := range set {
			lower := v - k
			upper := v + k
			_, ok := set[lower]
			if !ok {
				_, ok := set[upper]
				if ok {
					beautiful = false
					break
				}
			} else {
				beautiful = false
				break
			}
		}
		if beautiful {
			beautiful_count++
		}
	}
	return beautiful_count
}

/*
BUT the editorial on LeetCode gave a much better and faster solution technique that I'm going to implement here...
*/
func countBeautifulSubsets(nums []int, k int) int {
	// Look at each element in nums, and group them all by remainder when dividing by k.
	// Further, within each remainder group, keep a count of how many of each element are present
	nums_by_count_group_mod_k := [][]int{}
	group_by_mod := make(map[int]map[int]int)
	for _, num := range nums {
		mod := num % k
		_, ok := group_by_mod[mod]
		if !ok {
			group_by_mod[mod] = make(map[int]int)
		}
		_, ok = group_by_mod[mod][num]
		if !ok {
			group_by_mod[mod][num] = 1
		} else {
			group_by_mod[mod][num]++
		}
	}
	for _, num_map := range group_by_mod {
		this_mod := [][]int{}
		for num, count := range num_map {
			this_mod = append(this_mod, []int{num, count})
		}
		sort.SliceStable(this_mod, func(i, j int) bool {
			return this_mod[i][0] < this_mod[j][0]
		})
		nums_by_count_group_mod_k = append(nums_by_count_group_mod_k, this_mod...)
	}
	// Now we have elements and their counts stored in an array, with elements of the same modulus k group next to each other
	n := len(nums_by_count_group_mod_k)
	counts := make([]int, n)
	counts[n-1] = (1 << nums_by_count_group_mod_k[n-1][1]) - 1 // However many of that last element there are, we can pick any NON-EMPTY subset
	if n > 1 {
		counts[n-2] = (1 << nums_by_count_group_mod_k[n-2][1]) - 1 + counts[n-1] // ONLY include this element, or completely exclude it.
		if max(nums_by_count_group_mod_k[n-1][0], nums_by_count_group_mod_k[n-2][0])-min(nums_by_count_group_mod_k[n-1][0], nums_by_count_group_mod_k[n-2][0]) != k {
			// Include ANY non-empty number of both this element and non-empty number of the one before it.
			counts[n-2] += ((1 << nums_by_count_group_mod_k[n-2][1]) - 1) * counts[n-1]
		}
		for i := n - 3; i >= 0; i-- {
			counts[i] = (1 << nums_by_count_group_mod_k[i][1]) - 1 + counts[i+1] // Either ONLY include this element, or completely exclude it.
			if max(nums_by_count_group_mod_k[i][0], nums_by_count_group_mod_k[i+1][0])-min(nums_by_count_group_mod_k[i][0], nums_by_count_group_mod_k[i+1][0]) != k {
				// Include ANY non-empty number of both this element and non-empty number of the one before it.
				counts[i] += ((1 << nums_by_count_group_mod_k[i][1]) - 1) * counts[i+1]
			} else {
				// Then this number is NOT compatible with the preceding number, so since the numbers grouped by mod k are SORTED per mod, it IS compatible with the NEXT preceding number
				counts[i] += ((1 << nums_by_count_group_mod_k[i][1]) - 1) * counts[i+2]
			}
		}
	}
	return counts[0]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a list of words, list of  single letters (might be repeating) and score of every character.

Return the maximum score of any valid set of words formed by using the given letters (words[i] cannot be used two or more times).

It is not necessary to use all characters in letters and each letter can only be used once.
Score of letters 'a', 'b', 'c', ... ,'z' is given by score[0], score[1], ... , score[25] respectively.

Link:
https://leetcode.com/problems/maximum-score-words-formed-by-letters/description/?envType=daily-question&envId=2024-05-24
*/
func maxScoreWords(words []string, letters []byte, score []int) int {
	// Find all of words that we can match
	char_counts := make(map[byte]int)
	for _, b := range letters {
		_, ok := char_counts[b]
		if !ok {
			char_counts[b] = 1
		} else {
			char_counts[b]++
		}
	}
	matchable_words := []string{}
	for _, str := range words {
		this_word := make(map[byte]int)
		for i := 0; i < len(str); i++ {
			b := str[i]
			_, ok := this_word[b]
			if !ok {
				this_word[b] = 1
			} else {
				this_word[b]++
			}
		}
		can_match := true
		for ch, count := range this_word {
			available, ok := char_counts[ch]
			if !ok || available < count {
				can_match = false
				break
			}
		}
		if can_match {
			matchable_words = append(matchable_words, str)
		}
	}

	n := len(matchable_words)
	word_values := make([]int, n)
	for idx, word := range matchable_words {
		value := 0
		for i := 0; i < len(word); i++ {
			ascii := word[i]
			value += score[ascii-97] // ASCII conversion
		}
		word_values[idx] = value
	}

	return recMaxScoreWords((1<<n)-1, word_values, matchable_words, char_counts)
}

/*
Brute force
*/
func recMaxScoreWords(available int, word_values []int, words []string, char_counts map[byte]int) int {
	// Every '1' bit in available corresponds to a word we can select, try selecting that word, and removing all now non-available words
	record := 0
	for i := 0; i < len(word_values); i++ {
		if (1<<i)&available == (1 << i) {
			// That word is available to pick - try picking it
			new_char_counts := make(map[byte]int)
			for key, value := range char_counts {
				new_char_counts[key] = value
			}
			pick := word_values[i]
			for j := 0; j < len(words[i]); j++ {
				new_char_counts[words[i][j]]--
				if new_char_counts[words[i][j]] == 0 {
					delete(new_char_counts, words[i][j])
				}
			}
			new_available := available ^ (1 << i)
			// Now that we have updated our available characters, cut out all other strings we can no longer match
			for k := 0; k < len(words); k++ {
				if k == i || ((1<<k)&available == 0) {
					continue
				}
				this_word := make(map[byte]int)
				str := words[k]
				for i := 0; i < len(str); i++ {
					b := str[i]
					_, ok := this_word[b]
					if !ok {
						this_word[b] = 1
					} else {
						this_word[b]++
					}
				}
				// Cut out this string if we can no longer match it
				for ch, count := range this_word {
					available, ok := new_char_counts[ch]
					if !ok || available < count {
						new_available = new_available ^ (1 << k)
						break
					}
				}
			}
			pick += recMaxScoreWords(new_available, word_values, words, new_char_counts)
			record = max(record, pick)
		}
	}

	return record
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Suppose you have n integers labeled 1 through n. A permutation of those n integers perm (1-indexed) is considered a beautiful arrangement if for every i (1 <= i <= n), either of the following is true:

perm[i] is divisible by i.
i is divisible by perm[i].
Given an integer n, return the number of the beautiful arrangements that you can construct.

Source for Inspiration:
https://leetcode.com/problems/beautiful-arrangement/solutions/1000132/python-dp-bitmasks-explained
*/
func countArrangement(n int) int {
	// Bitmask for the bits that need placement
	need_placement := (1 << n) - 1
	sols := make(map[int]map[int]int)
	for i := 1; i < n; i++ {
		sols[i] = make(map[int]int)
	}
	return topDownCountArrangements(n, need_placement, n-1, sols)
}

/*
Top-down recursive helper method
*/
func topDownCountArrangements(n int, need_placement int, posn int, sols map[int]map[int]int) int {
	if posn == 0 {
		return 1
	} else {
		_, ok := sols[posn][need_placement]
		if !ok {
			// We need to solve this problem
			arrangements_found := 0
			for value := 1; value <= n; value++ {
				if (need_placement & (1 << (value - 1))) == (1 << (value - 1)) {
					// We need to place value - try placing it
					if (value%(posn+1) == 0) || ((posn+1)%value == 0) {
						// We can place this value here
						new_need_placement := need_placement ^ (1 << (value - 1))
						arrangements_found += topDownCountArrangements(n, new_need_placement, posn-1, sols)
					}
				}
			}
			sols[posn][need_placement] = arrangements_found
		}
		return sols[posn][need_placement]
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
An attendance record for a student can be represented as a string where each character signifies whether the student was absent, late, or present on that day. The record only contains the following three characters:

'A': Absent.
'L': Late.
'P': Present.
Any student is eligible for an attendance award if they meet both of the following criteria:

The student was absent ('A') for strictly fewer than 2 days total.
The student was never late ('L') for 3 or more consecutive days.
Given an integer n, return the number of possible attendance records of length n that make a student eligible for an attendance award. The answer may be very large, so return it modulo 109 + 7.

Link:
https://leetcode.com/problems/student-attendance-record-ii/description/?envType=daily-question&envId=2024-05-26
*/
func checkRecord(n int) int {
	num_ways := 0
	// Either you have an absent, or you do not

	// Suppose you do have an absent
	// It could be in any place from posn 1 to posn n
	noThreeInRowSols := make(map[int]int)
	for i := 1; i <= n/2+(n%2); i++ {
		placement := 1
		num_left := i - 1
		num_right := n - i
		// To the left and the right, we just have to have no 3 'L's in a row
		multiplier := 1
		if i <= n/2 {
			multiplier++
		}
		num_ways = modulo.ModularAdd(num_ways,
			modulo.ModularMultiply(multiplier,
				modulo.ModularMultiply(
					modulo.ModularMultiply(placement,
						numNoThreeLateInRow(num_left, noThreeInRowSols),
					),
					numNoThreeLateInRow(num_right, noThreeInRowSols),
				),
			),
		)
	}

	// Suppose you have no absent
	// Then you just need to not have 3 lates in a row
	num_ways = modulo.ModularAdd(num_ways, numNoThreeLateInRow(n, noThreeInRowSols))

	return num_ways
}

/*
Helper function to count the number of records that do not have 3 'L's in a row when using only 'L' and 'P'
*/
func numNoThreeLateInRow(n int, noThreeInRowSols map[int]int) int {
	if n < 3 {
		// You have total freedom placing the 'L's and 'P's because you don't have enough spots to HAVE 3 'L's in a row
		return modulo.ModularPow(2, n)
	} else if n < 4 {
		return 7
	} else if n < 5 {
		return 13
	} else {
		_, ok := noThreeInRowSols[n]
		if !ok {
			// Need to solve this problem
			// For our starting sequence in the attendance record, we could have:
			// LLP -> n-3
			// LPP -> n-3
			// PPP -> n-3
			// PLP -> n-3
			// LPLLP -> n-5
			// LPLP -> n-4
			// PLLP -> n-4
			// PPLLP -> n-5
			// PPLP -> n-4
			num_ways := modulo.ModularAdd(modulo.ModularMultiply(4,
				numNoThreeLateInRow(n-3, noThreeInRowSols),
			),
				modulo.ModularAdd(modulo.ModularMultiply(3,
					numNoThreeLateInRow(n-4, noThreeInRowSols),
				),
					modulo.ModularMultiply(2,
						numNoThreeLateInRow(n-5, noThreeInRowSols),
					),
				),
			)
			noThreeInRowSols[n] = num_ways
		}
		return noThreeInRowSols[n]
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given several boxes with different colors represented by different positive numbers.

You may experience several rounds to remove boxes until there is no box left. Each time you can choose some continuous boxes with the same color (i.e., composed of k boxes, k >= 1), remove them and get k * k points.

Return the maximum points you can get.

Link:
https://leetcode.com/problems/remove-boxes/description/

Inspiration:
https://leetcode.com/problems/remove-boxes/solutions/101310/java-top-down-and-bottom-up-dp-solutions
*/
func removeBoxes(boxes []int) int {
	n := len(boxes)
	dp := make([][][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([][]int, n)
		for j := 0; j < n; j++ {
			dp[i][j] = make([]int, n)
		}
	}

	return topDownRemoveBoxes(0, n-1, 0, boxes, dp)
}

/*
Top-down recursive helper method to solve the removing boxes problem
*/
func topDownRemoveBoxes(start, end, num_left int, boxes []int, dp [][][]int) int {
	if end < start {
		return 0
	} else {
		if dp[start][end][num_left] == 0 {
			// Need to solve this problem
			if start == end {
				// Then we will be removing (num_left + 1) boxes
				dp[start][end][num_left] = (num_left + 1) * (num_left + 1)
			} else {
				// We CAN remove the concatenated left-most (num_left + 1) boxes - but that may not be our only option
				record := (num_left+1)*(num_left+1) + topDownRemoveBoxes(start+1, end, 0, boxes, dp)
				// OR we can try finding the next box of the same color at index m, picking up all boxes from start+1 to m-1, and then we have one additional box of the start color to the left, with a new starting position
				for m := start + 1; m <= end; m++ {
					// We need to try this for ALL occurrences of the starting color in our range
					if boxes[m] == boxes[start] {
						record = max(record, topDownRemoveBoxes(start+1, m-1, 0, boxes, dp)+topDownRemoveBoxes(m, end, num_left+1, boxes, dp))
					}
				}
				dp[start][end][num_left] = record
			}
		}
		return dp[start][end][num_left]
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given the root of a binary tree, calculate the vertical order traversal of the binary tree.

For each node at position (row, col), its left and right children will be at positions (row + 1, col - 1) and (row + 1, col + 1) respectively. The root of the tree is at (0, 0).

The vertical order traversal of a binary tree is a list of top-to-bottom orderings for each column index starting from the leftmost column and ending on the rightmost column. There may be multiple nodes in the same row and same column. In such a case, sort these nodes by their values.

Return the vertical order traversal of the binary tree.

Link:
https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/description/
*/
func verticalTraversal(root *binary_tree.TreeNode) [][]int {
	column_row_values := make(map[int]map[int][]int)
	column_row_pairs := &[][]int{}
	recordTree(0, 0, root, column_row_pairs, column_row_values)
	columns_and_rows := *column_row_pairs
	for _, row := range column_row_values {
		for _, node_values := range row {
			sort.SliceStable(node_values, func(i, j int) bool {
				return node_values[i] < node_values[j]
			})
		}
	}

	sort.SliceStable(columns_and_rows, func(i, j int) bool {
		if columns_and_rows[i][0] != columns_and_rows[j][0] {
			return columns_and_rows[i][0] < columns_and_rows[j][0]
		} else {
			return columns_and_rows[i][1] < columns_and_rows[j][1]
		}
	})
	unique_columns_and_rows := [][]int{columns_and_rows[0]}
	for idx := 1; idx < len(columns_and_rows); idx++ {
		if !(columns_and_rows[idx][0] == columns_and_rows[idx-1][0] && columns_and_rows[idx][1] == columns_and_rows[idx-1][1]) {
			unique_columns_and_rows = append(unique_columns_and_rows, columns_and_rows[idx])
		}
	}

	node_values := [][]int{}
	last_col := math.MinInt
	idx := -1
	for _, col_row := range unique_columns_and_rows {
		if col_row[0] != last_col {
			idx++
			last_col = col_row[0]
			node_values = append(node_values, column_row_values[col_row[0]][col_row[1]])
		} else {
			node_values[idx] = append(node_values[idx], column_row_values[col_row[0]][col_row[1]]...)
		}
	}

	return node_values
}

/*
Helper function to record a tree in a map
*/
func recordTree(col, row int, root *binary_tree.TreeNode, column_row_pairs *[][]int, column_row_values map[int]map[int][]int) {
	if root == nil {
		return
	}

	_, ok := column_row_values[col]
	if !ok {
		column_row_values[col] = make(map[int][]int)
	}
	_, ok = column_row_values[col][row]
	if !ok {
		column_row_values[col][row] = []int{}
	}

	*column_row_pairs = append(*column_row_pairs, []int{col, row})

	column_row_values[col][row] = append(column_row_values[col][row], root.Val)
	recordTree(col-1, row+1, root.Left, column_row_pairs, column_row_values)
	recordTree(col+1, row+1, root.Right, column_row_pairs, column_row_values)
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given the binary representation of an integer as a string s, return the number of steps to reduce it to 1 under the following rules:

If the current number is even, you have to divide it by 2.

If the current number is odd, you have to add 1 to it.

It is guaranteed that you can always reach one for all test cases.

Link:
https://leetcode.com/problems/number-of-steps-to-reduce-a-number-in-binary-representation-to-one/description/?envType=daily-question&envId=2024-05-29
*/
func numSteps(s string) int {
	binary_rep := make([]bool, len(s)+1)
	for i := 0; i < len(s); i++ {
		if s[i] == '1' {
			binary_rep[i+1] = true
		}
	}

	return countNumSteps(binary_rep)
}

/*
Helper method to convert the binary number into 1
*/
func countNumSteps(binary_rep []bool) int {
	n := len(binary_rep)
	lowest_1 := findLowest1(binary_rep)
	if lowest_1 == n {
		// 0 - just add 1
		return 1
	} else {
		if lowest_1 == n-1 {
			// Already 1
			return 0
		} else {
			steps := 0
			for lowest_1 < n-1 {
				// If the number is even, divide by 2, which just means shifting every 1 you see one place right
				if !binary_rep[n-1] {
					for i := n - 2; i >= lowest_1; i-- {
						if binary_rep[i] {
							binary_rep[i+1] = true
							binary_rep[i] = false
						}
					}
					lowest_1++
				} else { // The number is odd - so add 1
					carry := true
					binary_rep[n-1] = false
					posn := n - 2
					for carry {
						if !binary_rep[posn] {
							carry = false
						}
						binary_rep[posn] = !binary_rep[posn]
						posn--
						if posn < lowest_1 && carry {
							lowest_1--
							binary_rep[lowest_1] = true
							break
						}
					}
				}
				steps++
			}
			return steps
		}
	}
}

/*
Helper method to verify the highest present in a binary representation
*/
func findLowest1(binary_rep []bool) int {
	highest_1 := len(binary_rep)
	for i := 0; i < len(binary_rep); i++ {
		if binary_rep[i] {
			highest_1 = i
			break
		}
	}
	return highest_1
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given an integer array nums of length n.

Assume arrk to be an array obtained by rotating nums by k positions clock-wise.
We define the rotation function F on nums as follow:

F(k) = 0 * arrk[0] + 1 * arrk[1] + ... + (n - 1) * arrk[n - 1].
Return the maximum value of F(0), F(1), ..., F(n-1).

The test cases are generated so that the answer fits in a 32-bit integer.

Link:
https://leetcode.com/problems/rotate-function/description/
*/
func maxRotateFunction(nums []int) int {
	weighted_sum := 0
	total := 0
	for idx, v := range nums {
		weighted_sum += idx * v
		total += v
	}
	record := weighted_sum
	n := len(nums)
	for front := n - 1; front > 0; front-- {
		weighted_sum += total
		weighted_sum -= n * nums[front]
		record = max(record, weighted_sum)
	}

	return record
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an array of intervals intervals where intervals[i] = [starti, endi], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

Link:
https://leetcode.com/problems/non-overlapping-intervals/
*/
func eraseOverlapIntervals(intervals [][]int) int {
	n := len(intervals)

	sort.SliceStable(intervals, func(i, j int) bool {
		return intervals[i][1] < intervals[j][1]
	})

	last_kept := 0
	removed := 0
	for idx := 1; idx < n; idx++ {
		if intervals[idx][0] < intervals[last_kept][1] {
			removed++
		} else {
			last_kept = idx
		}
	}

	return removed
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are playing a variation of the game Zuma.

In this variation of Zuma, there is a single row of colored balls on a board, where each ball can be colored red 'R', yellow 'Y', blue 'B', green 'G', or white 'W'. You also have several colored balls in your hand.

Your goal is to clear all of the balls from the board. On each turn:

Pick any ball from your hand and insert it in between two balls in the row or on either end of the row.
If there is a group of three or more consecutive balls of the same color, remove the group of balls from the board.
If this removal causes more groups of three or more of the same color to form, then continue removing each group until there are none left.
If there are no more balls on the board, then you win the game.
Repeat this process until you either win or do not have any more balls in your hand.
Given a string board, representing the row of balls on the board, and a string hand, representing the balls in your hand, return the minimum number of balls you have to insert to clear all the balls from the board.
If you cannot clear all the balls from the board using the balls in your hand, return -1.

Link:
https://leetcode.com/problems/zuma-game/description/

Inspiration for Optimization:
https://leetcode.com/problems/zuma-game/solutions/1568450/python-easy-bfs-solution-with-explain/
*/
func findMinStep(board string, hand string) int {
	visited := make(map[string]map[string]bool)
	reductions := make(map[string]string)

	board = reduce(board, reductions)
	type subproblem struct {
		board string
		hand  string
	}
	subproblem_queue := linked_list.NewQueue[subproblem]()
	subproblem_queue.Enqueue(subproblem{board: board, hand: hand})
	visited[board] = make(map[string]bool)
	visited[board][hand] = true
	moves := -1
	for !subproblem_queue.Empty() {
		n := subproblem_queue.Length()
		moves++
		for count := 0; count < n; count++ {
			board_hand := subproblem_queue.Dequeue()
			board := board_hand.board
			hand := board_hand.hand
			_, ok := visited[board]
			if !ok {
				visited[board] = make(map[string]bool)
			}
			visited[board][hand] = true
			if len(board) == 0 {
				return moves
			}
			for i := 0; i < len(hand); i++ { // Look at each character in hand
				// Skip contiguous characters in the hand
				if i > 0 && hand[i] == hand[i-1] {
					continue
				}
				for j := 0; j < len(board); j++ {
					// Position to place the character in the new board_buffer
					// Only insert a character from the end at the beginning of a contiguous set of the same characters in the board
					if j > 0 && board[j-1] == hand[i] {
						continue
					}
					// Further, only insert a character under the conditions:
					// The character to insert matches the character at position j in the board
					// OR
					// The place before insertion and the place after insertion in the board match, and the character you want to insert DOES NOT match
					// X -> XX
					// XX -> XYX
					// NOTE - XZ -> XYZ is NEVER going to be helpful - there is no reason to do that
					if !((board[j] == hand[i]) || (j > 0 && board[j] == board[j-1] && board[j] != hand[i])) {
						continue
					}
					var new_board_buffer bytes.Buffer
					var new_hand_buffer bytes.Buffer
					for k := 0; k < len(hand); k++ {
						if k != i {
							new_hand_buffer.WriteByte(hand[k])
						}
					}
					for l := 0; l < len(board); l++ {
						if l == j {
							new_board_buffer.WriteByte(hand[i])
						}
						new_board_buffer.WriteByte(board[l])
					}
					if j == len(board) {
						new_board_buffer.WriteByte(hand[i])
					}
					new_board := reduce(new_board_buffer.String(), reductions)
					new_hand := new_hand_buffer.String()
					if len(new_board) > 0 && len(new_hand) == 0 {
						continue
					}
					_, ok := visited[new_board]
					if ok {
						_, ok = visited[new_board][new_hand]
						if !ok {
							subproblem_queue.Enqueue(subproblem{board: new_board, hand: new_hand})
						}
					} else {
						subproblem_queue.Enqueue(subproblem{board: new_board, hand: new_hand})
					}
				}
			}
		}
	}

	return -1
}

/*
Helper function to collapse a board string - any 3 or more in a row disappear
*/
func reduce(board string, reductions map[string]string) string {
	_, ok := reductions[board]
	if !ok {
		// Need to solve this problem
		var board_buffer bytes.Buffer
		for i := 0; i < len(board); i++ {
			if i < len(board)-2 && (board[i] != board[i+1] || board[i] != board[i+2]) {
				board_buffer.WriteByte(board[i])
			} else if i < len(board)-2 {
				// Skip ALL of this character
				j := i
				for board[j] == board[i] {
					j++
					if j == len(board) {
						break
					}
				}
				i = j - 1
			} else {
				board_buffer.WriteByte(board[i])
			}
		}
		new_board := board_buffer.String()
		if len(new_board) == len(board) {
			reductions[board] = new_board
		} else { // Reduction could happen again now that some cancellation of characters has occurred
			reductions[board] = reduce(board_buffer.String(), reductions)
		}
	}
	return reductions[board]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// We will need the following data structure for the preceding program
type Environment struct {
	definitions map[string]int
	parent      *Environment
}

func (e *Environment) get(key string) int {
	v, ok := e.definitions[key]
	if ok {
		return v
	}
	return e.parent.get(key)
}

/*
You are given a string expression representing a Lisp-like expression to return the integer value of.

The syntax for these expressions is given as follows.
- An expression is either an integer, let expression, add expression, mult expression, or an assigned variable.
- Expressions always evaluate to a single integer. (An integer could be positive or negative.)
- A let expression takes the form "(let v1 e1 v2 e2 ... vn en expr)", where let is always the string "let"
  - Then there are one or more pairs of alternating variables and expressions, meaning that the first variable v1 is assigned the value of the expression e1
  - The second variable v2 is assigned the value of the expression e2, and so on sequentially
  - The value of this let expression is the value of the expression expr.

- An add expression takes the form "(add e1 e2)" where add is always the string "add", there are always two expressions e1, e2 and the result is the addition of the evaluation of e1 and the evaluation of e2.
- A mult expression takes the form "(mult e1 e2)" where mult is always the string "mult", there are always two expressions e1, e2 and the result is the multiplication of the evaluation of e1 and the evaluation of e2.
- For this question, we will use a smaller subset of variable names.
  - A variable starts with a lowercase letter, then zero or more lowercase letters or digits.

- Additionally, for your convenience, the names "add", "let", and "mult" are protected and will never be used as variable names.
- Finally, there is the concept of scope.
  - When an expression of a variable name is evaluated, within the context of that evaluation, the innermost scope (in terms of parentheses) is checked first for the value of that variable, and then outer scopes are checked sequentially.
  - It is guaranteed that every expression is legal.
  - Please see the examples for more details on the scope.

Link:
https://leetcode.com/problems/parse-lisp-expression/description/
*/
func evaluate(expression string) int {
	values := &Environment{definitions: make(map[string]int), parent: nil}
	return parseExpression(expression, values)
}

/*
Recursive helper function to parse the expression
*/
func parseExpression(expression string, values *Environment) int {
	// Knock off the outer parentheses if we have not already
	if expression[0] == '(' {
		return parseExpression(expression[1:len(expression)-1], values)
	}
	// The parentheses have been knocked off.
	// Now we need to see if we are doing a let, add, or mult
	first_space := 0
	for expression[first_space] != ' ' {
		first_space++
		if first_space >= len(expression) {
			return parseLiteral(expression, values)
		}
	}
	operation := expression[:first_space]
	if operation == "let" {
		return parseLetExpression(expression[4:], values)
	} else if operation == "add" {
		return parseAddExpression(expression[4:], values)
	} else {
		return parseMultExpression(expression[5:], values)
	}
}

/*
Helper function to parse a let expression - this will call 'parseExpression' if necessary.
*/
func parseLetExpression(expression string, values *Environment) int {
	// name1 exp1 name2 exp2 name3 exp3 exp
	// First, let's get our hands on the last expression 'exp'
	new_values := &Environment{definitions: make(map[string]int), parent: values}
	end_scanning_idx := -1
	if expression[len(expression)-1] == ')' {
		// Then we need to find the CORRESPONDING closing parentheses
		st := linked_list.NewStack[int]()
		st.Push(len(expression) - 1)
		idx := len(expression) - 2
		for !st.Empty() {
			if expression[idx] == '(' {
				st.Pop()
			} else if expression[idx] == ')' {
				st.Push(idx)
			}
			idx--
		}
		end_scanning_idx = (idx + 1) - 2
	} else {
		// Just find the last space
		last_space := len(expression) - 1
		for expression[last_space] != ' ' {
			last_space--
		}
		end_scanning_idx = last_space - 1
	}

	// Now we are ready to scan for 'name1 exp1 name2 exp2 name3 exp3'
	index := 0
	for index < end_scanning_idx {
		// Parse a name, and parse an expression
		next_space := index
		for expression[next_space] != ' ' {
			next_space++
		}
		end_exp := -1
		expr_value := -1
		if expression[next_space+1] == '(' {
			st := linked_list.NewStack[int]()
			st.Push(next_space + 1)
			idx := next_space + 2
			for !st.Empty() {
				if expression[idx] == ')' {
					st.Pop()
				} else if expression[idx] == '(' {
					st.Push(idx)
				}
				idx++
			} // WHERE the final closing parentheses was
			// Remember to knock of the parentheses
			end_exp = idx
			expr_value = parseExpression(expression[next_space+1:end_exp], new_values)
		} else {
			// Just find the following space
			following_space := next_space + 1
			for expression[following_space] != ' ' {
				following_space++
			}
			end_exp = following_space
			expr_value = parseExpression(expression[next_space+1:end_exp], new_values)
		}
		new_values.definitions[expression[index:next_space]] = expr_value
		index = end_exp + 1
	}

	return parseExpression(expression[end_scanning_idx+2:], new_values)
}

/*
Helper function to parse an add expression - this will call 'parseExpression' if necessary.
*/
func parseAddExpression(expression string, values *Environment) int {
	first_value, second_value := parseTwoValues(expression, values)
	return first_value + second_value
}

/*
Helper function to parse a mult expression - this will call 'parseExpression' if necessary.
*/
func parseMultExpression(expression string, values *Environment) int {
	first_value, second_value := parseTwoValues(expression, values)
	return first_value * second_value
}

/*
Helper function to parse the two values that come in the form 'exp1 exp2'
*/
func parseTwoValues(expression string, values *Environment) (int, int) {
	first_value := -1
	start_of_second_exp := -1
	if expression[0] == '(' {
		// Then we need to find the CORRESPONDING closing parentheses
		st := linked_list.NewStack[int]()
		st.Push(0)
		idx := 1
		for !st.Empty() {
			if expression[idx] == ')' {
				st.Pop()
			} else if expression[idx] == '(' {
				st.Push(idx)
			}
			idx++
		}
		closing_paren := idx - 1
		first_value = parseExpression(expression[1:closing_paren], values)
		start_of_second_exp = closing_paren + 2
	} else {
		// Then there is no closing parentheses to find - just find the next space
		first_space := 0
		for expression[first_space] != ' ' {
			first_space++
		}
		first_value = parseExpression(expression[:first_space], values)
		start_of_second_exp = first_space + 1
	}

	second_value := parseExpression(expression[start_of_second_exp:], values)

	return first_value, second_value
}

/*
Helper function to parse a literal expression - either a named variable or just a plain number
*/
func parseLiteral(expression string, values *Environment) int {
	if !regexp.MustCompile(`^-?\d+`).MatchString(expression) {
		// Not a number - our map must have the values
		return values.get(expression)
	} else {
		// Is a number
		value, _ := strconv.Atoi(expression)
		return value
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a string formula representing a chemical formula, return the count of each atom.

The atomic element always starts with an uppercase character, then zero or more lowercase letters, representing the name.

One or more digits representing that element's count may follow if the count is greater than 1. If the count is 1, no digits will follow.

For example, "H2O" and "H2O2" are possible, but "H1O2" is impossible.
Two formulas are concatenated together to produce another formula.

For example, "H2O2He3Mg4" is also a formula.
A formula placed in parentheses, and a count (optionally added) is also a formula.

For example, "(H2O2)" and "(H2O2)3" are formulas.
Return the count of all elements as a string in the following form: the first name (in sorted order), followed by its count (if that count is more than 1), followed by the second name (in sorted order), followed by its count (if that count is more than 1), and so on.

The test cases are generated so that all the values in the output fit in a 32-bit integer.

Link:
https://leetcode.com/problems/number-of-atoms/description/
*/
func countOfAtoms(formula string) string {
	atomCounts := atomsByCount(formula)
	atoms := []string{}
	for atom := range atomCounts {
		atoms = append(atoms, atom)
	}
	sort.SliceStable(atoms, func(i, j int) bool {
		return atoms[i] < atoms[j]
	})
	var new_formula bytes.Buffer
	for _, atom := range atoms {
		new_formula.WriteString(atom)
		if atomCounts[atom] > 1 {
			new_formula.WriteString(strconv.Itoa(atomCounts[atom]))
		}
	}

	return new_formula.String()
}

/*
Helper method to return the map of a set of basic molecules by count for a given string
*/
func atomsByCount(formula string) map[string]int {
	if formula[0] == '(' {
		st := linked_list.NewStack[int]()
		st.Push(0)
		idx := 1
		for !st.Empty() {
			if formula[idx] == ')' {
				st.Pop()
			} else if formula[idx] == '(' {
				st.Push(idx)
			}
			idx++
		}
		if idx == len(formula) {
			return atomsByCount(formula[1 : len(formula)-1])
		} else {
			inner_counts := atomsByCount(formula[1 : idx-1])
			coeff_start := idx
			for idx < len(formula) && regexp.MustCompile(`\d`).MatchString(formula[idx:idx+1]) {
				idx++
			}
			coeff, _ := strconv.Atoi(formula[coeff_start:idx])
			for atom := range inner_counts {
				inner_counts[atom] *= max(coeff, 1)
			}
			if idx < len(formula) {
				following_counts := atomsByCount(formula[idx:])
				for atom, count := range following_counts {
					_, ok := inner_counts[atom]
					if ok {
						inner_counts[atom] += count
					} else {
						inner_counts[atom] = count
					}
				}
			}
			return inner_counts
		}
	}
	idx := 0
	for idx < len(formula) && !regexp.MustCompile(`\d`).MatchString(formula[idx:idx+1]) && formula[idx] != '(' {
		idx++
	}
	for idx < len(formula) && regexp.MustCompile(`\d`).MatchString(formula[idx:idx+1]) {
		idx++
	}
	// Just find the first atom and its count, and the recurse for the rest
	i := 1
	var atom_name bytes.Buffer
	atom_name.WriteByte(formula[0])
	for i < len(formula) && regexp.MustCompile("[a-z]").MatchString(formula[i:i+1]) {
		atom_name.WriteByte(formula[i])
		i++
	}
	digit_present := false
	var count bytes.Buffer
	for i < len(formula) && regexp.MustCompile(`\d`).MatchString(formula[i:i+1]) {
		digit_present = true
		count.WriteByte(formula[i])
		i++
	}
	atom := atom_name.String()
	next_counts := make(map[string]int)
	if i < len(formula) {
		next_counts = atomsByCount(formula[i:])
	}
	if !digit_present {
		_, ok := next_counts[atom]
		if ok {
			next_counts[atom]++
		} else {
			next_counts[atom] = 1
		}
	} else {
		num, _ := strconv.Atoi(count.String())
		_, ok := next_counts[atom]
		if ok {
			next_counts[atom] += num
		} else {
			next_counts[atom] = num
		}
	}

	return next_counts
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Alice has some number of cards and she wants to rearrange the cards into groups so that each group is of size groupSize, and consists of groupSize consecutive cards.

Given an integer array hand where hand[i] is the value written on the ith card and an integer groupSize, return true if she can rearrange the cards, or false otherwise.

Link:
https://leetcode.com/problems/hand-of-straights/description/?envType=daily-question&envId=2024-06-06
*/
func isNStraightHand(hand []int, groupSize int) bool {
	if len(hand)%groupSize != 0 {
		return false
	}

	sort.SliceStable(hand, func(i, j int) bool {
		return hand[i] < hand[j]
	})
	unique_values := []int{hand[0]}
	counts := make(map[int]int)
	counts[hand[0]] = 1
	for i := 1; i < len(hand); i++ {
		_, ok := counts[hand[i]]
		if !ok {
			counts[hand[i]] = 1
			unique_values = append(unique_values, hand[i])
		} else {
			counts[hand[i]]++
		}
	}

	idx := 0
	accounted := 0
	for idx <= len(unique_values)-groupSize {
		counts[unique_values[idx]]--
		accounted++
		for i := 1; i < groupSize; i++ {
			if unique_values[idx+i]-unique_values[idx+i-1] != 1 {
				return false
			} else if counts[unique_values[idx+i]] == 0 {
				return false
			} else {
				counts[unique_values[idx+i]]--
				accounted++
			}
		}
		for idx <= len(unique_values)-groupSize && counts[unique_values[idx]] == 0 {
			idx++
		}
		if accounted == len(hand) {
			break
		}
	}
	return accounted == len(hand)
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
An integer array is called arithmetic if it consists of at least three elements and if the difference between any two consecutive elements is the same.

For example, [1,3,5,7,9], [7,7,7,7], and [3,-1,-5,-9] are arithmetic sequences.
Given an integer array nums, return the number of arithmetic subarrays of nums.

A subarray is a contiguous subsequence of the array.

Link:
https://leetcode.com/problems/arithmetic-slices/description/
*/
func numberOfArithmeticSlices(nums []int) int {
	count_ending := make([]int, len(nums))
	count_ending[0] = 1
	total := 1
	if len(nums) > 1 {
		count_ending[1] = 2
		total += 2
	}
	for i := 2; i < len(count_ending); i++ {
		// How many arithmetic subarrays end at this index?
		if nums[i]-nums[i-1] == nums[i-1]-nums[i-2] {
			count_ending[i] = 1 + count_ending[i-1]
		} else {
			count_ending[i] = 2
		}
		total += count_ending[i]
	}

	// Return the total, minus every length 1 subsequence, minus every length 2 subsequence
	return max(0, total-len(nums)-len(nums)+1)
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an integer array nums, return the number of all the arithmetic subsequences of nums.

A sequence of numbers is called arithmetic if it consists of at least three elements and if the difference between any two consecutive elements is the same.

For example, [1, 3, 5, 7, 9], [7, 7, 7, 7], and [3, -1, -5, -9] are arithmetic sequences.
For example, [1, 1, 2, 5, 7] is not an arithmetic sequence.
A subsequence of an array is a sequence that can be formed by removing some elements (possibly none) of the array.

For example, [2,5,10] is a subsequence of [1,2,1,2,4,1,5,10].
The test cases are generated so that the answer fits in 32-bit integer.

Link:
https://leetcode.com/problems/arithmetic-slices-ii-subsequence/description/
*/
func numberOfArithmeticSubsequences(nums []int) int {
	// Ask the question - how many subsequences with a step of x end at index i?
	n := len(nums)
	num_end := make(map[int]map[int]int)
	num_end[0] = make(map[int]int)
	total := 0
	for i := 1; i < n; i++ {
		for j := 0; j < i; j++ {
			diff := nums[i] - nums[j]
			_, ok := num_end[i]
			if !ok {
				num_end[i] = make(map[int]int)
			}
			_, ok = num_end[i][diff]
			if !ok {
				// We have now found one subsequence of size 2 that ends at this index with the given difference
				num_end[i][diff] = 1
			} else {
				num_end[i][diff]++
			}
			count, ok := num_end[j][diff]
			if ok {
				num_end[i][diff] += count
				total += count // Include these newly found arithmetic subsequences that end in nums[i] with diff of length >= 3
			}
			total += 1 // We know we have at least a pair with nums[i] and nums[j], so count that as it has not yet been counted
		}
	}

	// Return the total minus all pairs - because the pairs were counted but they don't count
	return total - n*(n-1)/2
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an integer array nums and an integer k, return the number of non-empty subarrays that have a sum divisible by k.

A subarray is a contiguous part of an array.

Link:
https://leetcode.com/problems/subarray-sums-divisible-by-k/description/?envType=daily-question&envId=2024-06-09
*/
func subarraysDivByK(nums []int, k int) int {
	// FROM THE EDITORIAL:
	// sum(i,j) = 0 (mod k) IFF sum(0,i)=sum(0,j)(mod k)
	n := len(nums)
	mods := make([]int, n)
	for idx, v := range nums {
		mods[idx] = v % k
		if mods[idx] < 0 {
			mods[idx] += k
		}
	}
	// For sum(0,i), we care about the sum modulus k - keep track of the count for how many
	mod_counts := make(map[int]int)
	total := 0
	current_mod := mods[0]
	mod_counts[current_mod] = 1
	if current_mod == 0 {
		total++
	}
	for i := 1; i < len(nums); i++ {
		current_mod = (current_mod + mods[i]) % k
		// This whole contiguous array counts
		if current_mod == 0 {
			total++
		}
		count, ok := mod_counts[current_mod]
		if ok {
			// Then we have that many previous subarrays with this same modulus
			// Each of those subarrays' ending point forms a starting point for a new array up to index i which achieves a sum of 0 mod k
			total += count
			mod_counts[current_mod]++
		} else {
			mod_counts[current_mod] = 1
		}
	}

	return total
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an integer array nums, return true if you can partition the array into two subsets such that the sum of the elements in both subsets is equal or false otherwise.

Link:
https://leetcode.com/problems/partition-equal-subset-sum/description/

Inspiration:
https://leetcode.com/problems/partition-equal-subset-sum/solutions/90592/0-1-knapsack-detailed-explanation/
*/
func canPartition(nums []int) bool {
	// Sort the list and have two outer pointers to gouge out a middle
	n := len(nums)
	sum := 0
	for _, v := range nums {
		sum += v
	}
	if sum%2 == 1 {
		return false
	} else {
		dp := make([]map[int]bool, n)
		for end := 0; end < n; end++ {
			dp[end] = make(map[int]bool)
		}
		return canAchieveSubsetSum(n-1, sum/2, nums, dp)
	}
}

/*
Top down helper method to try to see if it is possible to pick a subset from nums[0:end+1] with a sum equal to the input sum
*/
func canAchieveSubsetSum(end int, sum int, nums []int, dp []map[int]bool) bool {
	if sum == 0 {
		return true
	} else if sum < 0 {
		return false
	} else if end == 0 {
		return false
	} else {
		_, ok := dp[end][sum]
		if !ok {
			// Need to solve this problem - try including the number at nums[end], or NOT including it
			dp[end][sum] = canAchieveSubsetSum(end-1, sum, nums, dp) || canAchieveSubsetSum(end-1, sum-nums[end], nums, dp)
		}
		return dp[end][sum]
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
We define str = [s, n] as the string str which consists of the string s concatenated n times.

For example, str == ["abc", 3] =="abcabcabc".
We define that string s1 can be obtained from string s2 if we can remove some characters from s2 such that it becomes s1.
** I.E. -> s1 is a SUBSEQUENCE of s2 - so longest common subsequence between the two IS s1... **

For example, s1 = "abc" can be obtained from s2 = "abdbec" based on our definition by removing 'd', 'b', and 'e'.
You are given two strings s1 and s2 and two integers n1 and n2.
You have the two strings str1 = [s1, n1] and str2 = [s2, n2].

Return the maximum integer m such that str = [str2, m] can be obtained from str1.

Link:
https://leetcode.com/problems/count-the-repetitions/description/

Inspiration:
https://leetcode.com/problems/count-the-repetitions/solutions/4553332/maximising-repetitions-through-iterative-subsequence-matching/
*/
func getMaxRepetitions(s1 string, n1 int, s2 string, n2 int) int {
	// First count how many times n1 repetitions of s1 can include all characters of s2
	count := 0
	s2_posn := 0
	for i := 0; i < n1; i++ {
		for j := 0; j < len(s1); j++ {
			if s2[s2_posn] == s1[j] {
				s2_posn++
			}
			if s2_posn == len(s2) {
				count++
				s2_posn = 0
			}
		}
	}
	return count / n2
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
We define the string base to be the infinite wraparound string of "abcdefghijklmnopqrstuvwxyz", so base will look like this:
"...zabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcd..."

Given a string s, return the number of unique non-empty substrings of s are present in base.

Link:
https://leetcode.com/problems/unique-substrings-in-wraparound-string/description/
*/
func findSubstringInWraproundString(s string) int {
	// This question is essentially asking how many unique substrings of s are there?
	streaks := make([]int, len(s))
	streaks[0] = 1
	for i := 1; i < len(s); i++ {
		if (s[i-1] == s[i]-1) || (s[i-1] == 'z' && s[i] == 'a') {
			streaks[i] = streaks[i-1] + 1
		} else {
			streaks[i] = 1
		}
	}

	// For every character, we need the longest streak of consecutive characters that start at said character
	streaks_by_char := make(map[byte]int)
	for i := 0; i < len(s); i++ {
		starting_char := findStartingChar(s[i], streaks[i])
		for jump := 0; jump < min(26, streaks[i]); jump++ {
			length := streaks[i] - jump
			char := byte('a')
			if starting_char+byte(jump) > 'z' {
				char = 'a' + (starting_char + byte(jump) - 'z') - 1
			} else {
				char = starting_char + byte(jump)
			}
			_, ok := streaks_by_char[char]
			if ok {
				streaks_by_char[char] = max(streaks_by_char[char], length)
			} else {
				streaks_by_char[char] = length
			}
		}
	}

	// Add up all the lengths, because each map entry corresponds to a different starting character.
	// Therefore, is character ALPHA has length L attached to it, there are L unique consecutive substrings in 's' that START with L
	total := 0
	for _, length := range streaks_by_char {
		total += length
	}

	return total
}

/*
Helper function to find the starting character given the length of a streak of consecutive characters plus the ending character
*/
func findStartingChar(ch byte, streak_length int) byte {
	subtract_length := (streak_length - 1) % 26
	if ch-byte(subtract_length) < 'a' {
		return 'z' - (byte(subtract_length) - (ch - 97)) + 1
	} else {
		return ch - byte(subtract_length)
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Suppose LeetCode will start its IPO soon. In order to sell a good price of its shares to Venture Capital, LeetCode would like to work on some projects to increase its capital before the IPO. Since it has limited resources, it can only finish at most k distinct projects before the IPO. Help LeetCode design the best way to maximize its total capital after finishing at most k distinct projects.

You are given n projects where the ith project has a pure profit profits[i] and a minimum capital of capital[i] is needed to start it.

Initially, you have w capital. When you finish a project, you will obtain its pure profit and the profit will be added to your total capital.

Pick a list of at most k distinct projects from given projects to maximize your final capital, and return the final maximized capital.

The answer is guaranteed to fit in a 32-bit signed integer.

Link:
https://leetcode.com/problems/ipo/description/?envType=daily-question&envId=2024-06-15

Inspiration:
Discussion posts for the question...
*/
func findMaximizedCapital(k int, w int, profits []int, capital []int) int {
	// Sort the capital values and profit values based on increasing capital value
	tasks := make([][]int, len(profits))
	for i := 0; i < len(tasks); i++ {
		tasks[i] = []int{profits[i], capital[i]}
	}
	sort.SliceStable(tasks, func(i, j int) bool {
		return tasks[i][1] < tasks[j][1]
	})
	// We need to pick k tasks - note that we need an efficient way to remember the tasks we cannot yet afford but may be able to later
	total := w
	picked := 0
	// Note that we DO NOT lose capital when we invest in a project
	// Add projects starting with the cheapest ones to do, and put them in a priority queue based on profit
	h := heap.NewMaxHeap[int]()
	i := 0
	for i < len(tasks) && tasks[i][1] <= total {
		h.Insert(tasks[i][0])
		i++
	}
	for !h.Empty() && picked < k {
		picked++
		total += h.Extract()
		for i < len(tasks) && tasks[i][1] <= total {
			h.Insert(tasks[i][0])
			i++
		}
	}

	return total
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a sorted integer array nums and an integer n, add/patch elements to the array such that any number in the range [1, n] inclusive can be formed by the sum of some elements in the array.

Return the minimum number of patches required.

Link:
https://leetcode.com/problems/patching-array/description/?envType=daily-question&envId=2024-06-16

Inspiration:
Discussion Thread
*/
func minPatches(nums []int, n int) int {
	sort.SliceStable(nums, func(i, j int) bool {
		return nums[i] < nums[j]
	})
	patches := 0
	reached := 0
	idx := 0
	for idx < len(nums) && reached < n {
		if nums[idx]-reached >= 2 {
			// We are missing the numbers between (reached + 1) and (nums[idx] - 1), INCLUSIVE
			// Greedily, patch in the number (reached + 1), which now lets us cover up through (2 * reached + 1)
			// Now reached becomes (2 * reached + 1)
			// Keep doing this until reached is greater than or equal to nums[idx]
			for reached < nums[idx]-1 && reached < n {
				// The number we are about to patch in is reached + 1 - make sure we don't have that number in our array already
				patches++
				reached = 2*reached + 1
			}
		}
		// Now we make the following change because if we can reach all {1,2,...reached}, then we can add nums[idx] to any of those achievable sums
		reached += nums[idx]

		idx++
	}
	// Finally, we may have run out of elements to to explore in nums, but we still have more numbers we need to reach
	for reached < n {
		patches++
		reached = 2*reached + 1
	}

	return patches
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane, return the maximum number of points that lie on the same straight line.

Link:
https://leetcode.com/problems/max-points-on-a-line/description/
*/
func maxPoints(points [][]int) int {
	float_points := [][]float64{}
	for _, point := range points {
		float_points = append(float_points, []float64{float64(point[0]), float64(point[1])})
	}
	record_hits := 1 // If there's only one point, then it will be hit
	// Find every possible slope between two lines, and find how many points each of those lines hits
	for i := 0; i < len(float_points)-1; i++ {
		p1 := float_points[i]
		for j := i + 1; j < len(float_points); j++ {
			p2 := float_points[j]
			hits := 0
			if p2[0] == p1[0] {
				// Vertical line
				for _, point := range float_points {
					if p1[0] == point[0] {
						hits++
					}
				}
			} else {
				// Non-vertical line
				m := (p2[1] - p1[1]) / (p2[0] - p1[0])
				for _, point := range float_points {
					if float_rounding.RoundFloat(point[1]-p1[1], 5) == float_rounding.RoundFloat(m*(point[0]-p1[0]), 5) {
						hits++
					}
				}
			}
			record_hits = max(record_hits, hits)
		}
	}
	return record_hits
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a matrix and a target, return the number of non-empty submatrices that sum to target.

A submatrix x1, y1, x2, y2 is the set of all cells matrix[x][y] with x1 <= x <= x2 and y1 <= y <= y2.

Two submatrices (x1, y1, x2, y2) and (x1', y1', x2', y2') are different if they have some coordinate that is different: for example, if x1 != x1'.

Link:
https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/description/?envType=daily-question&envId=2024-06-19

Inspiration:
https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/submissions/1294070215/?envType=daily-question&envId=2024-06-19
*/
func numSubmatrixSumTarget(matrix [][]int, target int) int {
	row_sums := make([][]int, len(matrix)) // Row number, start column, end column
	for r := 0; r < len(matrix); r++ {
		row_sums[r] = make([]int, len(matrix[r]))
		row_sums[r][0] = matrix[r][0]
	}
	for r := 0; r < len(row_sums); r++ {
		for c := 1; c < len(row_sums[r]); c++ {
			row_sums[r][c] = row_sums[r][c-1] + matrix[r][c]
		}
	}

	// Now pick every possible pair of columns
	// For each such pair, start from the top and add up the row sums
	// Each time you add the sum of a row (between those two columns) to your sum, check if you've run into the DIFFERENCE between target and your current sum already (and how many times you have)
	// Each time you do, that's another submatrix which sums to target
	col_sums_seen := make([][]map[int]int, len(matrix[0]))
	for c := 0; c < len(col_sums_seen); c++ {
		col_sums_seen[c] = make([]map[int]int, len(matrix[0]))
		for i := 0; i < len(col_sums_seen); i++ {
			col_sums_seen[c][i] = make(map[int]int)
			col_sums_seen[c][i][0] = 1
		}
	}
	count := 0
	for c_left := 0; c_left < len(col_sums_seen); c_left++ {
		for c_right := c_left; c_right < len(col_sums_seen); c_right++ {
			sum := 0
			for r := 0; r < len(matrix); r++ {
				// Add this row (in between c_left and c_right) to our sum
				row_sum := row_sums[r][c_right]
				if c_left > 0 {
					row_sum -= row_sums[r][c_left-1]
				}
				sum += row_sum
				diff := sum - target
				_, ok := col_sums_seen[c_left][c_right][diff]
				if ok {
					// Then going from whatever row(s) achieved that difference AS ITS SUM to THIS current row - between these two columns - achieves the target sum
					// However many times said rows occurred
					count += col_sums_seen[c_left][c_right][diff]
				}
				// Record the difference we have just seen
				_, ok = col_sums_seen[c_left][c_right][sum]
				if ok {
					col_sums_seen[c_left][c_right][sum]++
				} else {
					col_sums_seen[c_left][c_right][sum] = 1
				}
			}
		}
	}

	return count
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You start with an initial power of power, an initial score of 0, and a bag of tokens given as an integer array tokens, where each tokens[i] denotes the value of tokeni.

Your goal is to maximize the total score by strategically playing these tokens. In one move, you can play an unplayed token in one of the two ways (but not both for the same token):

Face-up: If your current power is at least tokens[i], you may play token_i, losing tokens[i] power and gaining 1 score.
Face-down: If your current score is at least 1, you may play tokeni, gaining tokens[i] power and losing 1 score.
Return the maximum possible score you can achieve after playing any number of tokens.

Link:
https://leetcode.com/problems/bag-of-tokens/description/
*/
func bagOfTokensScore(tokens []int, power int) int {
	sort.SliceStable(tokens, func(i, j int) bool {
		return tokens[i] < tokens[j]
	})

	score := 0
	left := 0
	right := len(tokens) - 1
	for left <= right {
		if power >= tokens[left] {
			power -= tokens[left]
			score++
			left++
		} else if score > 0 && right > left {
			power += tokens[right]
			score--
			right--
		} else {
			break
		}
	}

	return score
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a string s, return the longest
palindromic substring in s.

Link:
https://leetcode.com/problems/longest-palindromic-substring/description/
*/
func longestPalindrome(s string) string {
	record := s[0:1]
	isPalindrome := make([][]bool, len(s))
	for i := 0; i < len(isPalindrome); i++ {
		isPalindrome[i] = make([]bool, len(s))
		isPalindrome[i][i] = true
	}

	// Now solve the problem using bottom-up dynamic programming
	for length := 2; length <= len(s); length++ {
		for start := 0; start <= len(s)-length; start++ {
			end := start + length - 1
			if s[start] == s[end] {
				isPalindrome[start][end] = (start == end-1) || isPalindrome[start+1][end-1]
				if isPalindrome[start][end] && (end-start+1 > len(record)) {
					record = s[start : end+1]
				}
			}
		}
	}

	return record
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:

'.' Matches any single character.​​​​
'*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).

Link:
https://leetcode.com/problems/regular-expression-matching/description/
*/
func isMatch(s string, p string) bool {
	// first key is s-index, second key is p-index
	isMatch := make(map[int]map[int]bool)
	for i := -1; i < len(s); i++ {
		isMatch[i] = make(map[int]bool)
	}
	return topDownIsMatch(s, p, len(s)-1, len(p)-1, isMatch)
}

/*
Top-down recursive helper method to solve the above problem
*/
func topDownIsMatch(s string, p string, s_idx int, p_idx int, isMatch map[int]map[int]bool) bool {
	_, ok := isMatch[s_idx][p_idx]
	if !ok {
		// We need to solve this problem

		// Base cases
		if (s_idx < 0) && (p_idx < 0) {
			// Both ran out
			isMatch[s_idx][p_idx] = true
		} else if p_idx < 0 {
			// Only the pattern ran out
			isMatch[s_idx][p_idx] = false
		} else if s_idx < 0 {
			// Only the string ran out - whatever's left in the pattern had better be optional
			isMatch[s_idx][p_idx] = p_idx >= 1 && p[p_idx] == '*' && topDownIsMatch(s, p, s_idx, p_idx-2, isMatch)
		} else {
			// Non-base cases
			if p_idx == 0 {
				// We are on the last index in the pattern to match
				if s_idx == 0 {
					isMatch[s_idx][p_idx] = s[s_idx] == p[p_idx] || p[p_idx] == '.'
				} else {
					isMatch[s_idx][p_idx] = false
				}
			} else {
				if p[p_idx] == '.' || s[s_idx] == p[p_idx] {
					isMatch[s_idx][p_idx] = topDownIsMatch(s, p, s_idx-1, p_idx-1, isMatch)
				} else if p[p_idx] == '*' && (p[p_idx-1] == s[s_idx] || p[p_idx-1] == '.') {
					// Try matching and and making it the last match
					isMatch[s_idx][p_idx] = topDownIsMatch(s, p, s_idx-1, p_idx-2, isMatch)
					// Try matching and NOT making it the last match
					isMatch[s_idx][p_idx] = isMatch[s_idx][p_idx] || topDownIsMatch(s, p, s_idx-1, p_idx, isMatch)
					// Try not matching
					isMatch[s_idx][p_idx] = isMatch[s_idx][p_idx] || topDownIsMatch(s, p, s_idx, p_idx-2, isMatch)
				} else if p[p_idx] == '*' {
					// Forced to skip the preceding character in the pattern (not matching)
					isMatch[s_idx][p_idx] = topDownIsMatch(s, p, s_idx, p_idx-2, isMatch)
				} else {
					isMatch[s_idx][p_idx] = false
				}
			}
		}
	}
	return isMatch[s_idx][p_idx]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

Link:
https://leetcode.com/problems/generate-parentheses/description/
*/
func generateParenthesis(n int) []string {
	output := []string{"()"}
	for i := 2; i <= n; i++ {
		new_parens := []string{}
		// Where are all the places we could put the new two parentheses?
		for _, s := range output {
			var buffer bytes.Buffer
			// These two new parentheses could go around any (*) valid set of parentheses (because remember 's' is valid)
			st := linked_list.NewStack[int]()
			for idx := 0; idx <= len(s); idx++ {
				if st.Empty() {
					// Wrap whatever is left with a '(*)'
					if idx > 0 {
						buffer.WriteString(s[:idx])
					}
					buffer.WriteString("(")
					if idx < len(s) {
						buffer.WriteString(s[idx:])
					}
					buffer.WriteString(")")
					new_parens = append(new_parens, buffer.String())
					buffer.Reset()
				}
				if idx < len(s) && s[idx] == ')' {
					st.Pop()
				} else {
					st.Push(idx)
				}
			}
		}
		output = new_parens
	}

	return output
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a string containing just the characters '(' and ')', return the length of the longest valid (well-formed) parentheses substring.

Link:
https://leetcode.com/problems/longest-valid-parentheses/description/
*/
func longestValidParentheses(s string) int {
	longestStartingHere := make(map[int]int)
	record := 0
	for i := 0; i < len(s); i++ {
		record = max(record, findLongestStartingHere(i, s, longestStartingHere))
	}
	return record
}

/*
Helper function (top-down) to find the longest valid set of parentheses starting at the given index
*/
func findLongestStartingHere(i int, s string, longestStartingHere map[int]int) int {
	if i >= len(s)-1 {
		return 0
	} else if s[i] == ')' {
		return 0
	} else {
		_, ok := longestStartingHere[i]
		if !ok {
			// Need to solve this problem
			if s[i+1] == ')' {
				// ()*
				longestStartingHere[i] = 2 + findLongestStartingHere(i+2, s, longestStartingHere)
			} else {
				// (*)* MAYBE - if we can find that later parentheses
				nest := findLongestStartingHere(i+1, s, longestStartingHere)
				if nest > 0 && i+nest+1 < len(s) && s[i+nest+1] == ')' {
					longestStartingHere[i] = nest + 2 + findLongestStartingHere(i+nest+2, s, longestStartingHere)
				} else {
					longestStartingHere[i] = 0
				}
			}
		}
		return longestStartingHere[i]
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

Link:
https://leetcode.com/problems/trapping-rain-water/description/
*/
func trap(height []int) int {
	i := 0
	for i < len(height) && height[i] == 0 {
		i++
	}

	total := 0
	st := linked_list.NewStack[int]()
	st.Push(i)
	for i < len(height)-1 {
		i++
		h := height[i]
		if h == 0 {
			continue
		}
		next_highest := 0
		for !st.Empty() {
			relevant_height := min(height[st.Peek()], h)
			total += (i - st.Peek() - 1) * (relevant_height - next_highest)
			if height[st.Peek()] < h {
				next_highest = height[st.Pop()]
			} else {
				if height[st.Peek()] == h {
					st.Pop()
				}
				break
			}
		}
		st.Push(i)
	}

	return total
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given an integer array nums.
You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.

Return true if you can reach the last index, or false otherwise.

Link:
https://leetcode.com/problems/jump-game/description/
*/
func canJump(nums []int) bool {
	canReach := make([]bool, len(nums))
	// For a given index, can we reach the end from that index?
	canReach[len(canReach)-1] = true
	for i := len(canReach) - 2; i >= 0; i-- {
		max_jump_dist := nums[i]
		for jump := 1; jump <= min(max_jump_dist, len(nums)-i-1); jump++ {
			canReach[i] = canReach[i+jump]
			if canReach[i] {
				break
			}
		}
	}
	return canReach[0]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given a 0-indexed array of integers nums of length n.
You are initially positioned at nums[0].

Each element nums[i] represents the maximum length of a forward jump from index i.
In other words, if you are at nums[i], you can jump to any nums[i + j] where:

0 <= j <= nums[i] and
i + j < n

Return the minimum number of jumps to reach nums[n - 1].
The test cases are generated such that you can reach nums[n - 1].

Link:
https://leetcode.com/problems/jump-game-ii/description/
*/
func jump(nums []int) int {
	min_jumps := make([]int, len(nums))
	for i := 0; i < len(min_jumps)-1; i++ {
		min_jumps[i] = math.MaxInt
	}

	for i := len(min_jumps) - 2; i >= 0; i-- {
		for jump := 1; jump <= min(nums[i], len(nums)-i-1); jump++ {
			if min_jumps[i+jump] != math.MaxInt {
				min_jumps[i] = min(min_jumps[i], 1+min_jumps[i+jump])
			}
		}
	}

	return min_jumps[0]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an integer array nums, find the subarray with the largest sum, and return its sum.

Link:
https://leetcode.com/problems/maximum-subarray/
*/
func maxSubArray(nums []int) int {
	records := make([]int, len(nums))
	records[0] = nums[0]
	best := records[0]
	// Consider, if you MUST include nums[i], and have nums[0:i+1] to work with, what's the best sum you could achieve?
	for i := 1; i < len(records); i++ {
		// DON'T include the immediate left-most best sum achievable, or DO include it - whichever achieves a greater sum.
		records[i] = max(nums[i], nums[i]+records[i-1])
		best = max(best, records[i])
	}

	return best
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

Link:
https://leetcode.com/problems/two-sum/description/
*/
func twoSum(nums []int, target int) []int {
	val_to_indices := make(map[int][]int)
	for idx, v := range nums {
		_, ok := val_to_indices[v]
		if !ok {
			val_to_indices[v] = []int{idx}
		} else {
			val_to_indices[v] = append(val_to_indices[v], idx)
		}
	}

	sort.SliceStable(nums, func(i, j int) bool {
		return nums[i] < nums[j]
	})

	left := 0
	right := len(nums) - 1
	for nums[left]+nums[right] != target {
		if nums[left]+nums[right] < target {
			left++
		} else {
			right--
		}
	}
	if nums[left] == nums[right] {
		return []int{val_to_indices[nums[left]][0], val_to_indices[nums[left]][1]}
	} else {
		return []int{val_to_indices[nums[left]][0], val_to_indices[nums[right]][0]}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given the root of a binary search tree, return a balanced binary search tree with the same node values.
If there is more than one answer, return any of them.

A binary search tree is balanced if the depth of the two subtrees of every node never differs by more than 1.

Link:
https://leetcode.com/problems/balance-a-binary-search-tree/description/?envType=daily-question&envId=2024-06-26
*/
func balanceBST(root *binary_tree.TreeNode) *binary_tree.TreeNode {
	st := linked_list.NewStack[*binary_tree.TreeNode]()
	explored := make(map[*binary_tree.TreeNode]bool)
	st.Push(root)
	values := []int{}
	for !st.Empty() {
		if st.Peek().Left != nil {
			_, ok := explored[st.Peek().Left]
			if !ok {
				explored[st.Peek().Left] = true
				st.Push(st.Peek().Left)
			} else {
				curr := st.Pop()
				values = append(values, curr.Val)
				if curr.Right != nil {
					st.Push(curr.Right)
				}
			}
		} else {
			curr := st.Pop()
			values = append(values, curr.Val)
			if curr.Right != nil {
				st.Push(curr.Right)
			}
		}
	}

	nodes := make([]*binary_tree.TreeNode, len(values))
	idx_stack := linked_list.NewStack[[]int]()
	idx_stack.Push([]int{0, len(values)})
	for !idx_stack.Empty() {
		left := idx_stack.Peek()[0]
		right := idx_stack.Peek()[1]
		mid := (left + right) / 2
		if mid == left { // Base case - size 1
			idx_stack.Pop()
			nodes[mid] = &binary_tree.TreeNode{Val: values[mid], Left: nil, Right: nil}
		} else if mid == left-1 { // Base case - size 2
			idx_stack.Pop()
			nodes[left] = &binary_tree.TreeNode{Val: values[left], Left: nil, Right: nil}
			nodes[mid] = &binary_tree.TreeNode{Val: values[mid], Left: nodes[left], Right: nil}
		} else {
			if nodes[mid] == nil {
				// Do left
				nodes[mid] = &binary_tree.TreeNode{Val: values[mid], Left: nil, Right: nil}
				idx_stack.Push([]int{left, mid})
			} else if (mid+1+right)/2 < right && nodes[(mid+1+right)/2] == nil {
				// Do right
				idx_stack.Push([]int{mid + 1, right})
			} else {
				// Both left and right are done
				nodes[mid].Left = nodes[(left+mid)/2]
				if (mid+1+right)/2 < right {
					nodes[mid].Right = nodes[(mid+1+right)/2]
				}
				idx_stack.Pop()
			}
		}
	}

	return nodes[len(nodes)/2]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
There is a robot on an m x n grid.
The robot is initially located at the top-left corner (i.e., grid[0][0]).
The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]).
The robot can only move either down or right at any point in time.

Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The test cases are generated so that the answer will be less than or equal to 2 * 10^9.

Link:
https://leetcode.com/problems/unique-paths/description/
*/
func uniquePaths(m int, n int) int {
	// We will have to move (m-1 + n-1) times
	// ANY m-1 of those moves will be horizontal
	// You also hence immediately choose the reamining n-1 of those moves to be vertical
	return combinations.Choose(m-1+n-1, m-1)
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given an m x n integer array grid.
There is a robot initially located at the top-left corner (i.e., grid[0][0]).
The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]).
The robot can only move either down or right at any point in time.

An obstacle and space are marked as 1 or 0 respectively in grid.
A path that the robot takes cannot include any square that is an obstacle.

Return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The testcases are generated so that the answer will be less than or equal to 2 * 10^9.

Link:
https://leetcode.com/problems/unique-paths-ii/description/
*/
func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	r := len(obstacleGrid)
	c := len(obstacleGrid[0])
	sols := make([][]int, r)
	for i := 0; i < r; i++ {
		sols[i] = make([]int, c)
	}
	if obstacleGrid[r-1][c-1] == 0 {
		sols[r-1][c-1] = 1
		for row := r - 1; row >= 0; row-- {
			for col := c - 1; col >= 0; col-- {
				if row == r-1 && col == c-1 {
					continue
				} else {
					if obstacleGrid[row][col] == 1 {
						sols[row][col] = 0
					} else { // Go down or go right
						if row < r-1 {
							// Count ways if we go down
							sols[row][col] += sols[row+1][col]
						}
						if col < c-1 {
							// Count ways if we go right
							sols[row][col] += sols[row][col+1]
						}
					}
				}
			}
		}
		return sols[0][0]
	} else {
		return 0
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

Link:
https://leetcode.com/problems/minimum-path-sum/description/
*/
func minPathSum(grid [][]int) int {
	r := len(grid)
	c := len(grid[0])
	min_sums := make([][]int, r)
	for i := 0; i < r; i++ {
		min_sums[i] = make([]int, c)
	}
	min_sums[r-1][c-1] = grid[r-1][c-1]
	for row := r - 1; row >= 0; row-- {
		for col := c - 1; col >= 0; col-- {
			if row == r-1 && col == c-1 {
				continue
			} else {
				min_sums[row][col] = grid[row][col]
				addition := math.MaxInt
				if col < c-1 {
					addition = min(addition, min_sums[row][col+1])
				}
				if row < r-1 {
					addition = min(addition, min_sums[row+1][col])
				}
				min_sums[row][col] += addition
			}
		}
	}
	return min_sums[0][0]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are climbing a staircase.
It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Link:
https://leetcode.com/problems/climbing-stairs/description/
*/
func climbStairs(n int) int {
	prev_prev := 1
	prev := 1
	curr := 1
	for i := 1; i < n; i++ {
		curr = prev_prev + prev
		prev_prev = prev
		prev = curr
	}
	return curr
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.

You have the following three operations permitted on a word:

- Insert a character
- Delete a character
- Replace a character

Link:
https://leetcode.com/problems/edit-distance/description/
*/
func minDistance(word1 string, word2 string) int {
	i, j := 0, 0
	sols := make(map[int]map[int]int)
	return topDownMinDistance(i, word1, j, word2, sols)
}

/*
Top down helper method to find the minimum edit distance based on the two strings
*/
func topDownMinDistance(i int, word1 string, j int, word2 string, sols map[int]map[int]int) int {
	_, ok := sols[i]
	if !ok {
		sols[i] = make(map[int]int)
	}
	_, ok = sols[i][j]
	if !ok {
		// Need to solve this problem
		if i == len(word1) && j == len(word2) {
			// Words are completed
			sols[i][j] = 0
		} else if i == len(word1) {
			// Add the rest of the characters in word2 to what you have in word1
			sols[i][j] = len(word2) - j
		} else if j == len(word2) {
			// Delete the characters in word1 that you don't have in word2
			sols[i][j] = len(word1) - i
		} else {
			if word1[i] == word2[j] {
				// If the characters match, just move on
				sols[i][j] = topDownMinDistance(i+1, word1, j+1, word2, sols)
			} else {
				// Try inserting the character from word2 into word1
				sols[i][j] = 1 + topDownMinDistance(i, word1, j+1, word2, sols)
				// Try deleting the character from word1 that doesn't match in word2
				sols[i][j] = min(sols[i][j], 1+topDownMinDistance(i+1, word1, j, word2, sols))
				// Try replacing the character in word1 that doesn't match with the character in word2
				sols[i][j] = min(sols[i][j], 1+topDownMinDistance(i+1, word1, j+1, word2, sols))
			}
		}
	}
	return sols[i][j]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
The next greater element of some element x in an array is the first greater element that is to the right of x in the same array.

You are given two distinct 0-indexed integer arrays nums1 and nums2, where nums1 is a subset of nums2.

For each 0 <= i < nums1.length, find the index j such that nums1[i] == nums2[j] and determine the next greater element of nums2[j] in nums2.
If there is no next greater element, then the answer for this query is -1.

Return an array ans of length nums1.length such that ans[i] is the next greater element as described above.

Link:
https://leetcode.com/problems/next-greater-element-i/description/

Inspiration:
https://leetcode.com/problems/next-greater-element-i/solutions/97595/java-10-lines-linear-time-complexity-o-n-with-explanation/
*/
func nextGreaterElement(nums1 []int, nums2 []int) []int {
	next_greater := make(map[int]int)
	// We need to maintain a stack of decreasing values
	decreasing_stack := linked_list.NewStack[int]()
	for _, v := range nums2 {
		if decreasing_stack.Empty() || decreasing_stack.Peek() > v {
			decreasing_stack.Push(v)
		} else {
			for !decreasing_stack.Empty() && decreasing_stack.Peek() < v {
				next_greater[decreasing_stack.Pop()] = v
			}
			decreasing_stack.Push(v)
		}
	}
	for !decreasing_stack.Empty() {
		next_greater[decreasing_stack.Pop()] = -1
	}

	results := []int{}
	for _, v := range nums1 {
		next_greater_value, ok := next_greater[v]
		if ok {
			results = append(results, next_greater_value)
		} else {
			results = append(results, -1)
		}
	}

	return results
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a circular integer array nums (i.e., the next element of nums[nums.length - 1] is nums[0]), return the next greater number for every element in nums.

The next greater number of a number x is the first greater number to its traversing-order next in the array, which means you could search circularly to find its next greater number.
If it doesn't exist, return -1 for this number.

Link:
https://leetcode.com/problems/next-greater-element-ii/description/

Inspiration:
The Editorial
*/
func nextGreaterElements(nums []int) []int {
	decreasing_stack := linked_list.NewStack[int]()
	// For a given index, record the next greater element
	next_greater := make(map[int]int)
	for repeat := 0; repeat < 2; repeat++ {
		for i := len(nums) - 1; i >= 0; i-- {
			for !decreasing_stack.Empty() && nums[decreasing_stack.Peek()] <= nums[i] {
				decreasing_stack.Pop()
			}
			if decreasing_stack.Empty() {
				next_greater[i] = -1
			} else {
				next_greater[i] = nums[decreasing_stack.Peek()]
			}
			decreasing_stack.Push(i)
		}
	}
	results := make([]int, len(nums))
	for i := 0; i < len(results); i++ {
		results[i] = next_greater[i]
	}
	return results
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a positive integer n, find the smallest integer which has exactly the same digits existing in the integer n and is greater in value than n.
If no such positive integer exists, return -1.

Note that the returned integer should fit in 32-bit integer, if there is a valid answer but it does not fit in 32-bit integer, return -1.

Link:
https://leetcode.com/problems/next-greater-element-iii/description/

Inspiration:
The discussion forums...
*/
func nextGreaterElementIII(n int) int {
	// 10 digits for max 32-bit integer
	// NOTE - 10! is under 4 million - brute force will work
	digits := make([]int, int(math.Log10(float64(n)))+1)
	for i := 0; i < len(digits); i++ {
		// rounded_num will have i+1 digits - we want the right-most one
		rounded_num := n / (int(math.Pow(float64(10), float64(len(digits)-1-i))))
		digits[i] = rounded_num % 10
	}
	// Going from right, find the first digit smaller than previous digit.
	// Find the first digit bigger than said digit. Swap those two values.
	// Given your new array of digits, reverse the order of all digits preceding the number that swapped places with that aforementioned digit
	for i := len(digits) - 2; i >= 0; i-- {
		if digits[i] < digits[i+1] {
			for k := len(digits) - 1; k >= i+1; k-- {
				if digits[i] < digits[k] {
					digits[i], digits[k] = digits[k], digits[i]
					break
				}
			}
			// Now we need to flip all of the digits that precede the digit position we just swapped - because they were in decreasing order - making them increasing will minimize this next larger number
			num_digits_to_flip := len(digits) - i - 1
			for j := i + 1; j < i+1+num_digits_to_flip/2; j++ {
				digits[j], digits[len(digits)-(j-i)] = digits[len(digits)-(j-i)], digits[j]
			}
			res := 0
			for k := 0; k < len(digits); k++ {
				res += digits[k] * int(math.Pow(float64(10), float64(len(digits)-1-k)))
				if res < 0 || res > 2147483647 {
					return -1
				}
			}
			return res
		}
	}

	return -1
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

Link:
https://leetcode.com/problems/largest-rectangle-in-histogram/description/

Inspiration:
https://medium.com/algorithms-digest/largest-rectangle-in-histogram-234004ecd15a#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjJhZjkwZTg3YmUxNDBjMjAwMzg4OThhNmVmYTExMjgzZGFiNjAzMWQiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJhdWQiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJzdWIiOiIxMDY2NTg0ODA4ODE4Njk5ODMxMTMiLCJlbWFpbCI6Im1pa2V5dGFsbHlmZXJndXNvbkBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwibmJmIjoxNzE5NjA5OTc1LCJuYW1lIjoiTWlrZXkgRmVyZ3Vzb24iLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUNnOG9jS3NGNk9hLWtGNlk5QnREWGJQbjNiMTZqZmE0Zkl1cHJIWjJBaEJWWW9nX2ZfTE5RMD1zOTYtYyIsImdpdmVuX25hbWUiOiJNaWtleSIsImZhbWlseV9uYW1lIjoiRmVyZ3Vzb24iLCJpYXQiOjE3MTk2MTAyNzUsImV4cCI6MTcxOTYxMzg3NSwianRpIjoiN2NmNGNjNDRhNmNmZjQyN2QxODAzNmE3YjY2YzA2MzlkM2Y2ZWRkNSJ9.lNkIfnS1XT3LzW-OCPU_VBaluRCBvzIJBPucyywxH6rMlrN8Jx48g72_EOOj4Jf_ViyWStPI1uDDCFfp4wVMltMMlwoxHz1P4s3yYqGozjT2izF8ZqSK8wWwUNBw4LN0T9W9UDabv-q_EKM5UKS-sMLgWFfWcNQAcrrAo0xk-uNfNeD1DGQcP3pbMDlayzldXxi9K-Yl0cMqP-RbSwvc75pcUPb1CPzZsXqJnZQZT0KHSdeyEbUbJ8mBCPocGu_9V-EyDbHIrWDh6qMuSFV7DQUWxz5KQPfOLxHVy6AwteZyO8jTkMlc6x9bcP2LPgaNdbSU_YXs96K-Mv12cyf-Tw
*/
func largestRectangleArea(heights []int) int {
	// The first thing we need to do is, for each height in heights, find the first LEFT BOUND height smaller than this height and record its position
	// Do the same for the first RIGHT BOUND height lower than this height
	// L is left posn index, R is right posn index, width is L-R-1, and height is the height of our current rectangle
	// See if that area breaks the record

	// SO FIRST, find those left bound lesser height positions for all elements in heights
	first_left_less := make([]int, len(heights))
	non_decreasing_stack := linked_list.NewStack[int]()
	for i := len(heights) - 1; i >= 0; i-- {
		for !non_decreasing_stack.Empty() && heights[non_decreasing_stack.Peek()] > heights[i] {
			first_left_less[non_decreasing_stack.Pop()] = i
		}
		non_decreasing_stack.Push(i)
	}
	for !non_decreasing_stack.Empty() {
		first_left_less[non_decreasing_stack.Pop()] = -1
	}

	// Now do the same for the right bound lesser height positions
	first_right_less := make([]int, len(heights))
	for i := 0; i < len(heights); i++ {
		for !non_decreasing_stack.Empty() && heights[non_decreasing_stack.Peek()] > heights[i] {
			first_right_less[non_decreasing_stack.Pop()] = i
		}
		non_decreasing_stack.Push(i)
	}
	for !non_decreasing_stack.Empty() {
		first_right_less[non_decreasing_stack.Pop()] = len(heights)
	}

	// Now try making every possible height in heights THE height to be our largest possible rectangle
	record := 0
	for i := 0; i < len(heights); i++ {
		width := first_right_less[i] - first_left_less[i] - 1
		record = max(record, width*heights[i])
	}

	return record
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area

Link:
https://leetcode.com/problems/maximal-rectangle/

Inspiration:
The discussion forums were helpful!
*/
func maximalRectangle(matrix [][]byte) int {
	// We will repeat the problem of maximal rectangle area as we progress down the rows
	rectangles := make([]int, len(matrix[0]))
	record := 0
	for row := 0; row < len(matrix); row++ {
		for i := 0; i < len(matrix[row]); i++ {
			if matrix[row][i] == '1' {
				rectangles[i]++
			} else {
				rectangles[i] = 0
			}
		}
		record = max(record, largestRectangleArea(rectangles))
	}

	return record
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Alice and Bob have an undirected graph of n nodes and three types of edges:

Type 1: Can be traversed by Alice only.
Type 2: Can be traversed by Bob only.
Type 3: Can be traversed by both Alice and Bob.
Given an array edges where edges[i] = [type_i, u_i, v_i] represents a bidirectional edge of type typei between nodes u_i and v_i, find the maximum number of edges you can remove so that after removing the edges, the graph can still be fully traversed by both Alice and Bob.
The graph is fully traversed by Alice and Bob if starting from any node, they can reach all other nodes.

Return the maximum number of edges you can remove, or return -1 if Alice and Bob cannot fully traverse the graph.

Link:
https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/description/?envType=daily-question&envId=2024-06-30

Inspiration:
The LeetCode hints...
*/
func maxNumEdgesToRemove(n int, edges [][]int) int {
	// Sort the edges so that we analyze the SHARED edges first
	sort.SliceStable(edges, func(i, j int) bool {
		return edges[i][0] > edges[j][0]
	})

	alice_node_factory := disjointset.NewSetOfSets[int]()
	bob_node_factory := disjointset.NewSetOfSets[int]()
	for i:=1; i<=n; i++ {
		alice_node_factory.MakeNode(i)
		bob_node_factory.MakeNode(i)
	}

	// Now join our nodes based on the edges and remove as we go
	removed := 0
	for _, edge := range edges {
		n1 := edge[1]
		n2 := edge[2]
		if edge[0] == 3 {
			// Shared edge
			node1_alice := alice_node_factory.GetNode(n1)
			node1_bob := bob_node_factory.GetNode(n1)
			node2_alice := alice_node_factory.GetNode(n2)
			node2_bob := bob_node_factory.GetNode(n2)
			// See if both Alice and Bob already have these two nodes connected
			bob_needs := node1_bob.RootValue() != node2_bob.RootValue()
			alice_needs := node1_alice.RootValue() != node2_alice.RootValue()
			if bob_needs || alice_needs {
				node1_bob.Join(node2_bob)
				node1_alice.Join(node2_alice)
			} else {
				removed++
			}
		} else if edge[0] == 2 {
			// Bob edge
			node_1 := bob_node_factory.GetNode(n1)
			node_2 := bob_node_factory.GetNode(n2)

			// Does Bob need this edge?
			if node_1.RootValue() == node_2.RootValue() {
				// Nope
				removed++
			} else {
				node_1.Join(node_2)
			}
		} else {
			// Alice edge
			node_1 := alice_node_factory.GetNode(n1)
			node_2 := alice_node_factory.GetNode(n2)

			// Does Alice need this edge?
			if node_1.RootValue() == node_2.RootValue() {
				// Nope
				removed++
			} else {
				node_1.Join(node_2)
			}
		}
	}

	// Check and see if all nodes are connected.
	// If not then neither Alice nor Bob could fully traverse in the first place.
	bob_root := bob_node_factory.GetNode(1).RootValue()
	alice_root := alice_node_factory.GetNode(1).RootValue()
	for i:=2; i<=n; i++ {
		if bob_node_factory.GetNode(i).RootValue() != bob_root {
			return -1
		} else if alice_node_factory.GetNode(i).RootValue() != alice_root {
			return -1
		}
	}

	return removed
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
We can scramble a string s to get a string t using the following algorithm:

If the length of the string is 1, stop.
If the length of the string is > 1, do the following:
- Split the string into two non-empty substrings at a random index, i.e., if the string is s, divide it to x and y where s = x + y.
- Randomly decide to swap the two substrings or to keep them in the same order. i.e., after this step, s may become s = x + y or s = y + x.
- Apply step 1 recursively on each of the two substrings x and y.

Given two strings s1 and s2 of the same length, return true if s2 is a scrambled string of s1, otherwise, return false.

Link:
https://leetcode.com/problems/scramble-string/description/
*/
func isScramble(s1 string, s2 string) bool {
	is_scramble := make(map[string]bool)
	return topDownIsScramble(s1, s2, is_scramble)
}

/*
Top-down recursive helper method to solve the isScramble problem
*/
func topDownIsScramble(s1 string, s2 string, is_scramble map[string]bool) bool {
	var buffer bytes.Buffer
	buffer.WriteString(s1)
	buffer.WriteString(s2)
	combined := buffer.String()
	_, ok := is_scramble[combined]
	if !ok {
		if s1 == s2 {
			is_scramble[combined] = true
		} else if len(s1) == 1 {
			is_scramble[combined] = false
		} else {
			for split := 1; split < len(s1); split++ {
				left_s1 := s1[:split]
				right_s1 := s1[split:]
				// Try swapping the two strings in s1
				left_s2_swap := s2[len(s2)-split:]
				right_s2_swap := s2[:len(s2)-split]
				if topDownIsScramble(left_s1, left_s2_swap, is_scramble) && topDownIsScramble(right_s1, right_s2_swap, is_scramble) {
					is_scramble[combined] = true
					break
				} else {
					// Not swapping the two strings in s1 is our only hope
					left_s2_no_swap := s2[:split]
					right_s2_no_swap := s2[split:]
					is_scramble[combined] = topDownIsScramble(left_s1, left_s2_no_swap, is_scramble) && topDownIsScramble(right_s1, right_s2_no_swap, is_scramble)
					if is_scramble[combined] {
						break
					}
				}
			}
		}
	}
	return is_scramble[combined]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
A message containing letters from A-Z can be encoded into numbers using the following mapping:
- 'A' -> "1"
- 'B' -> "2"
- ...
- 'Z' -> "26"

To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). 
For example, "11106" can be mapped into:
- "AAJF" with the grouping (1 1 10 6)
- "KJF" with the grouping (11 10 6)
- Note that the grouping (1 11 06) is invalid because "06" cannot be mapped into 'F' since "6" is different from "06".

Given a string s containing only digits, return the number of ways to decode it.

The test cases are generated so that the answer fits in a 32-bit integer.

Link:
https://leetcode.com/problems/decode-ways/description/
*/
func numDecodings(s string) int {
	num_to_char := make(map[string]rune)
	for i:=1; i<=26; i++ {
		num_to_char[strconv.Itoa(i)] = rune('A' + i - 1)
	}
	num_decodings := make(map[string]int)
	return topDownNumDecodings(s, num_decodings, num_to_char)
}

/*
Top down helper method to solve the numDecodings problem
*/
func topDownNumDecodings(s string, num_decodings map[string]int, num_to_char map[string]rune) int {
	_, ok := num_decodings[s]
	if !ok {
		// Need to solve this problem
		if len(s) == 1 {
			_, ok := num_to_char[s]
			if !ok {
				num_decodings[s] = 0
			} else {
				num_decodings[s] = 1
			}
		} else if len(s) == 2 {
			_, ok := num_to_char[s]
			if !ok {
				left_count := topDownNumDecodings(s[:1], num_decodings, num_to_char)
				right_count := topDownNumDecodings(s[1:], num_decodings, num_to_char)
				if left_count == 1 && right_count == 1 {
					num_decodings[s] = 1
				}
			} else {
				num_decodings[s] = 1
				left_count := topDownNumDecodings(s[:1], num_decodings, num_to_char)
				right_count := topDownNumDecodings(s[1:], num_decodings, num_to_char)
				if left_count == 1 && right_count == 1 {
					num_decodings[s] += 1
				}
			}
		} else {
			num_decodings[s] = 0
			first := s[:1]
			_, ok := num_to_char[first]
			if ok {
				num_decodings[s] += topDownNumDecodings(s[1:], num_decodings, num_to_char)
			}
			first_two := s[:2]
			_, ok = num_to_char[first_two]
			if ok {
				num_decodings[s] += topDownNumDecodings(s[2:], num_decodings, num_to_char)
			}
		}
	}
	return num_decodings[s]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given an integer array nums.

In one move, you can choose one element of nums and change it to any value.

Return the minimum difference between the largest and smallest value of nums after performing at most three moves.

Link:
https://leetcode.com/problems/minimum-difference-between-largest-and-smallest-value-in-three-moves/description/?envType=daily-question&envId=2024-07-03

Inspiration:
The Editorial
*/
func minDifference(nums []int) int {
	if len(nums) <= 4 {
		return 0
	}
	sort.SliceStable(nums, func(i, j int) bool {
		return nums[i] < nums[j]
	})
	n := len(nums)
	// Try deleting lowest 3
	// Try deleting lowest 2 and highest
	// Try deleting lowest and highest 2
	// Try deleting highest 3
	return min(nums[n-4] - nums[0],
		min(nums[n-3] - nums[1],
			min(
				nums[n-2] - nums[2],
				nums[n-1] - nums[3],
			),
		),
	)
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given the head of a linked list, which contains a series of integers separated by 0's. The beginning and end of the linked list will have Node.val == 0.

For every two consecutive 0's, merge all the nodes lying in between them into a single node whose value is the sum of all the merged nodes. The modified list should not contain any 0's.

Return the head of the modified linked list.

Link:
https://leetcode.com/problems/merge-nodes-in-between-zeros/description/?envType=daily-question&envId=2024-07-04
*/
func mergeNodes(head *list_node.ListNode) *list_node.ListNode {
    new_head := &list_node.ListNode{Val: 0, Next: nil}
	current_addition := new_head
	running_sum := 0
	current := head.Next
	for current != nil {
		if current.Val == 0 {
			current_addition.Val = running_sum
			if current.Next != nil {
				new_addition := &list_node.ListNode{Val: 0, Next: nil}
				current_addition.Next = new_addition
				current_addition = new_addition
			}
			running_sum = 0
		} else {
			running_sum += current.Val
		}
		current = current.Next
	}

	return new_head
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an integer n, return all the structurally unique BST's (binary search trees), which has exactly n nodes of unique values from 1 to n. 
Return the answer in any order.

Link:
https://leetcode.com/problems/unique-binary-search-trees-ii/description/
*/
func generateTrees(n int) []*binary_tree.TreeNode {
	sols := make([][][]*binary_tree.TreeNode, n)
	for i:=0; i<n; i++ {
		sols[i] = make([][]*binary_tree.TreeNode, n)
		sols[i][i] = []*binary_tree.TreeNode{{Val: i+1, Left: nil, Right: nil}}
	}

	for num_values := 2; num_values <= n; num_values++ {
		for start := 0; start <= n - num_values; start++ {
			end := start + num_values - 1
			trees := []*binary_tree.TreeNode{}
			for root := start; root <= end; root++ {
				if root == start {
					for _, tree := range sols[root+1][end] {
						trees = append(trees, &binary_tree.TreeNode{Val: root+1, Left: nil, Right: tree})
					}
				} else if root == end {
					for _, tree := range sols[start][root-1] {
						trees = append(trees, &binary_tree.TreeNode{Val: root+1, Left: tree, Right: nil})
					}
				} else {
					for _, left_tree := range sols[start][root-1] {
						for _, right_tree := range sols[root+1][end] {
							trees = append(trees, &binary_tree.TreeNode{Val: root+1, Left: left_tree, Right: right_tree})
						}
					}
				}
			}
			sols[start][end] = make([]*binary_tree.TreeNode, len(trees))
			copy(sols[start][end], trees)
		}
	}

	return sols[0][n-1]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an integer n, return the number of structurally unique BST's (binary search trees) which has exactly n nodes of unique values from 1 to n.

Link:
https://leetcode.com/problems/unique-binary-search-trees/description/
*/
func numTrees(n int) int {
	sols := make([]int, n+1)
	sols[0] = 1 // One way to have no nodes in a tree
    sols[1] = 1 // One way to have one node in a tree
	for i:=2; i<=n; i++ {
		for num_in_left := 0; num_in_left < i; num_in_left++ {
			num_in_right := i - 1 - num_in_left
			sols[i] += sols[num_in_left] * sols[num_in_right]
		}
	}

    return sols[n]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given strings s1, s2, and s3, find whether s3 is formed by an interleaving of s1 and s2.

An interleaving of two strings s and t is a configuration where s and t are divided into n and m substrings respectively, such that:

s = s1 + s2 + ... + sn
t = t1 + t2 + ... + tm
|n - m| <= 1
The interleaving is s1 + t1 + s2 + t2 + s3 + t3 + ... or t1 + s1 + t2 + s2 + t3 + s3 + ...
Note: a + b is the concatenation of strings a and b.

Link:
https://leetcode.com/problems/interleaving-string/description/
*/
func isInterleave(s1 string, s2 string, s3 string) bool {
	if len(s3) < len(s1) + len(s2) {
		return false
	}
    sols := make(map[int]map[int]bool)
	for i:=0; i<=len(s1); i++ {
		sols[i] = make(map[int]bool)
	}
	// If we manage to use all the characters, then we made it
	return topDownIsInterleave(s1, s2, s3, 0, 0, sols)
}

/*
Top-down helper method for the above problem
*/
func topDownIsInterleave(s1 string, s2 string, s3 string, s1_used int, s2_used int, sols map[int]map[int]bool) bool {
	_, ok := sols[s1_used][s2_used]
	if !ok {
		// Need to solve this problem
		if s1_used == len(s1) && s2_used == len(s2) { 
			// We managed to use up all the characters in both s1 and s2 matching along s3
			sols[s1_used][s2_used] = len(s1) + len(s2) == len(s3) // Make sure that matches ALL of s3
		} else if s1_used == len(s1) {
			// We can only use characters from s2 now
			sols[s1_used][s2_used] = s2[s2_used:] == s3[s1_used + s2_used:]
		} else if s2_used == len(s2) {
			// We can only use characters from s1 now
			sols[s1_used][s2_used] = s1[s1_used:] == s3[s1_used + s2_used:]
		} else {
			if (s1[s1_used] != s3[s1_used + s2_used]) && (s2[s2_used] != s3[s1_used + s2_used]) {
				// No first characters match - we're at a dead end
				sols[s1_used][s2_used] = false
			} else {
				sols[s1_used][s2_used] = false
				if s1[s1_used] == s3[s1_used + s2_used] {
					// Try matching the first character in s1 since we can
					sols[s1_used][s2_used] = sols[s1_used][s2_used] || topDownIsInterleave(s1, s2, s3, s1_used+1, s2_used, sols)
				}
				if s2[s2_used] == s3[s1_used + s2_used] {
					// Try matching the first character in s2 since we can
					sols[s1_used][s2_used] = sols[s1_used][s2_used] || topDownIsInterleave(s1, s2, s3, s1_used, s2_used+1, sols)
				} 
			}
		}
	}
	return sols[s1_used][s2_used]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given two strings s and t, return the number of distinct subsequences of s which equals t.

The test cases are generated so that the answer fits on a 32-bit signed integer.

Link:
https://leetcode.com/problems/distinct-subsequences/description/

Inspiration:
https://leetcode.com/problems/distinct-subsequences/solutions/5402100/dp-tabulation-memoization-c-must-watch/
*/
func numDistinct(s string, t string) int {
	// For a given index in s and index in t, how many subsequences in s that that use up to the first s_count characters in s equal t up to its first t_count characters?
	counts := make(map[int]map[int]int)
	for i:=0; i<=len(s); i++ {
		counts[i] = make(map[int]int)
	}
	return topDownNumDistinct(s, len(s), t, len(t), counts)
}

/*
Top down helper method to find how many subsequences in s that END at the s-index equal t up to t-index
*/
func topDownNumDistinct(s string, s_count int, t string, t_count int, counts map[int]map[int]int) int {
	_, ok := counts[s_count][t_count]
	if !ok {
		// Need to solve this problem
		if t_count == 0 {
			counts[s_count][t_count] = 1 // There is one way to match the empty sequence of characters - with the empty sequence
		} else if s_count == 0 {
			counts[s_count][t_count] = 0 // Ran out of characters in s to match the required characters in t
		} else {
			counts[s_count][t_count] = 0
			if s[s_count-1] == t[t_count-1] {
				// Try matching these two characters
				counts[s_count][t_count] += topDownNumDistinct(s, s_count-1, t, t_count-1, counts)
			}
			// Try not matching these two characters
			counts[s_count][t_count] += topDownNumDistinct(s, s_count-1, t, t_count, counts)
		}
	}
	return counts[s_count][t_count]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an integer numRows, return the first numRows of Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:

Link:
https://leetcode.com/problems/pascals-triangle/description/
*/
func generate(numRows int) [][]int {
	rows := [][]int{{1}}
	if numRows >= 2 {
		rows = append(rows, []int{1,1})
		for i:=3; i<=numRows; i++ {
			prev_row := rows[len(rows)-1]
			next_row := []int{1}
			for j:=0; j<len(prev_row)-1; j++ {
				next_row = append(next_row, prev_row[j] + prev_row[j+1])
			}
			next_row = append(next_row, 1)
			rows = append(rows, next_row)
		}
	}
    return rows
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an integer rowIndex, return the rowIndexth (0-indexed) row of the Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:

Link:
https://leetcode.com/problems/pascals-triangle-ii/
*/
func getRow(rowIndex int) []int {
	current_row := []int{1}
	for i:=1; i<=rowIndex; i++ {
		next_row := []int{1}
		for j:=0; j<len(current_row)-1; j++ {
			next_row = append(next_row, current_row[j] + current_row[j+1])
		}
		next_row = append(next_row, 1)
		current_row = next_row
	}
	return current_row
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a triangle array, return the minimum path sum from top to bottom.

For each step, you may move to an adjacent number of the row below. More formally, if you are on index i on the current row, you may move to either index i or index i + 1 on the next row.

Link:
https://leetcode.com/problems/triangle/description/
*/
func minimumTotal(triangle [][]int) int {
	min_sums := []int{triangle[0][0]}

	for i:=1; i<len(triangle); i++ {
		new_min_sums := make([]int, len(triangle[i]))
		new_min_sums[0] = min_sums[0] + triangle[i][0]
		for j:=1; j<len(min_sums); j++ {
			new_min_sums[j] = min(min_sums[j-1], min_sums[j]) + triangle[i][j]
		}
		new_min_sums[len(new_min_sums)-1] = min_sums[len(min_sums)-1] + triangle[i][len(new_min_sums)-1]
		min_sums = new_min_sums
	}

	min_sum := math.MaxInt
	for _, v := range min_sums {
		min_sum = min(min_sum, v)
	}
    return min_sum
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given two string arrays positive_feedback and negative_feedback, containing the words denoting positive and negative feedback, respectively. 
Note that no word is both positive and negative.

Initially every student has 0 points. 
Each positive word in a feedback report increases the points of a student by 3, whereas each negative word decreases the points by 1.

You are given n feedback reports, represented by a 0-indexed string array report and a 0-indexed integer array student_id, where student_id[i] represents the ID of the student who has received the feedback report report[i]. 
The ID of each student is unique.

Given an integer k, return the top k students after ranking them in non-increasing order by their points. 
In case more than one student has the same points, the one with the lower ID ranks higher.

Link:
https://leetcode.com/problems/reward-top-k-students/description/
*/
func topStudents(positive_feedback []string, negative_feedback []string, report []string, student_id []int, k int) []int {
	positives := make(map[string]bool)
	negatives := make(map[string]bool)
	for _, v := range positive_feedback {
		positives[v] = true
	}
	for _, v := range negative_feedback {
		negatives[v] = true
	}

	scores := make(map[int]int)
	for idx, id := range student_id {
		student_report := report[idx]
		scores[id] = 0
		words := strings.Split(student_report, " ")
		for _, w := range words {
			_, ok := positives[w]
			if ok {
				scores[id] += 3
			} else {
				_, ok = negatives[w]
				if ok {
					scores[id]--
				}
			}
		}
	}

	sort.SliceStable(student_id, func(i, j int) bool {
		id_i := student_id[i]
		id_j := student_id[j]
		if scores[id_i] != scores[id_j] {
			return scores[id_i] > scores[id_j]
		} else {
			return id_i < id_j
		}
	})
    return student_id[:k]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. 
If you cannot achieve any profit, return 0.

Link:
https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/
*/
func maxProfit(prices []int) int {
	record := 0
	// The price of whatever we decide to buy
	buy_price := prices[0]
	for idx:=1; idx<len(prices); idx++ {
		price := prices[idx]
		if price > buy_price {
			record = max(record, price - buy_price)
		} else {
			buy_price = price
		}
	}
    return record
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given an integer array prices where prices[i] is the price of a given stock on the ith day.

On each day, you may decide to buy and/or sell the stock. 
You can only hold at most one share of the stock at any time. 
However, you can buy it then immediately sell it on the same day.

Find and return the maximum profit you can achieve.

Link:
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/
*/
func maxProfit2(prices []int) int {
	// First we need to find this out for all elements
	first_greater_idx := make([]int, len(prices))
	non_increasing_stack := linked_list.NewStack[int]()
	non_increasing_stack.Push(0)
	for i:=1; i<len(prices); i++ {
		for !non_increasing_stack.Empty() && prices[non_increasing_stack.Peek()] < prices[i] {
			first_greater_idx[non_increasing_stack.Pop()] = i
		}
		non_increasing_stack.Push(i)
	}
	for !non_increasing_stack.Empty() {
		first_greater_idx[non_increasing_stack.Pop()] = -1
	}

	// Now we are ready to solve this problem bottom-up style
	sols := make([]int, len(prices))
	for i:=len(prices)-2; i>=0; i-- {
		if first_greater_idx[i] == -1 {
			// Just DO NOT buy stock on this day
			sols[i] = sols[i+1]
		} else {
			// There is a way to profit if we buy today - try buying and try not buying
			sols[i] = max(sols[i+1], prices[first_greater_idx[i]] - prices[i] + sols[first_greater_idx[i]])
		}
	}
    return sols[0]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given an array prices where prices[i] is the price of a given stock on the ith day.

Find the maximum profit you can achieve. 
You may complete at most two transactions.

Note: You may not engage in multiple transactions simultaneously 
(i.e., you must sell the stock before you buy again).

Link:
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/description/

Inspiration:
https://cpexplanations.blogspot.com/2021/04/123-best-time-to-buy-and-sell-stock-iii.html
*/
func maxProfit3(prices []int) int {
	sols := make([]map[int]int, 4)
	sols[0] = make(map[int]int) // BSBS
	sols[0][len(prices)-1] = 0 // On the last element - don't buy
	sols[1] = make(map[int]int) // SBS
	sols[1][len(prices)-1] = prices[len(prices)-1] // On the last element - sell
	sols[2] = make(map[int]int) // BS
	sols[2][len(prices)-1] = 0 // On the last element - don't buy
	sols[3] = make(map[int]int) // S
	sols[3][len(prices)-1] = prices[len(prices)-1] // On the last element - sell
    return topDownMaxProfit3(0, 0, prices, sols)
}

/*
Top-down helper method to solve the above problem
*/
func topDownMaxProfit3(idx int, operations int, prices []int, sols []map[int]int) int {
	_, ok := sols[operations][idx]
	if !ok {
		// Need to solve this problem
		if operations == 3 {
			// S - try selling and try not selling
			sols[operations][idx] = max(topDownMaxProfit3(idx+1, operations, prices, sols), prices[idx])
		} else if operations == 1 {
			// SBS - again try sell or not selling
			sols[operations][idx] = max(topDownMaxProfit3(idx+1, operations, prices, sols), prices[idx] + topDownMaxProfit3(idx+1, operations+1, prices, sols))
		} else {
			// B* - try buying and try not buying
			sols[operations][idx] = max(topDownMaxProfit3(idx+1, operations, prices, sols), topDownMaxProfit3(idx+1, operations + 1, prices, sols) - prices[idx])
		}
	}
	return sols[operations][idx]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given an integer array prices where prices[i] is the price of a given stock on the ith day, and an integer k.

Find the maximum profit you can achieve. 
You may complete at most k transactions: i.e. you may buy at most k times and sell at most k times.

Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

Link:
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/description/
*/
func maxProfit4(k int, prices []int) int {
    sols := make([]map[int]int, 2*k)
	for i:=0; i<2*k; i++ {
		sols[i] = make(map[int]int)
		if i % 2 == 1 { // Start with a sell
			sols[i][len(prices)-1] = prices[len(prices)-1]
		} else { // Start with a buy
			sols[i][len(prices)-1] = 0
		}
	}
    return topDownMaxProfit4(0, 0, prices, sols)
}

/*
Top-down helper method to solve the above problem
*/
func topDownMaxProfit4(idx int, operations int, prices []int, sols []map[int]int) int {
	_, ok := sols[operations][idx]
	if !ok {
		// Need to solve this problem
		if operations == len(sols)-1 {
			// Only one option left - to sell - try selling and try not selling
			sols[operations][idx] = max(topDownMaxProfit4(idx+1, operations, prices, sols), prices[idx])
		} else if operations % 2 == 1 {
			// S* - again try sell or not selling
			sols[operations][idx] = max(topDownMaxProfit4(idx+1, operations, prices, sols), prices[idx] + topDownMaxProfit4(idx+1, operations+1, prices, sols))
		} else {
			// B* - try buying and try not buying
			sols[operations][idx] = max(topDownMaxProfit4(idx+1, operations, prices, sols), topDownMaxProfit4(idx+1, operations + 1, prices, sols) - prices[idx])
		}
	}
	return sols[operations][idx]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given an array prices where prices[i] is the price of a given stock on the ith day.

Find the maximum profit you can achieve. 
You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times) with the following restrictions:
- After you sell your stock, you cannot buy stock on the next day (i.e., cooldown one day).
- Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

Link:
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/description/
*/
func maxProfit5(prices []int) int {
	// Ready to buy
	best_given_can_buy := make([]int, len(prices))
	best_given_can_buy[len(prices)-1] = 0

	// Ready to sell
	best_given_can_sell := make([]int, len(prices))
	best_given_can_sell[len(prices)-1] = prices[len(prices)-1]
	if len(prices) >= 2 { // Also a base case, since if you sell on the second to last day, you're still done
		best_given_can_sell[len(prices)-2] = max(prices[len(prices)-2], prices[len(prices)-1])
	}
 
	for i:=len(prices)-2; i>=0; i-- {
		// Suppose we're ready to buy
		// Try buying
		best_given_can_buy[i] = best_given_can_sell[i+1] - prices[i]
		// Try not buying
		best_given_can_buy[i] = max(best_given_can_buy[i], best_given_can_buy[i+1])

		if i < len(prices)-2 {
			// Suppose we're ready to sell
			// Try selling - remember the cooldown
			best_given_can_sell[i] = prices[i] + best_given_can_buy[i+2]
			// Try not selling
			best_given_can_sell[i] = max(best_given_can_sell[i], best_given_can_sell[i+1])
		}
	}

    return best_given_can_buy[0]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
There is a restaurant with a single chef. 
You are given an array customers, where customers[i] = [arrival_i, time_i]:

arrival_i is the arrival time of the ith customer. 
The arrival times are sorted in non-decreasing order.
time_i is the time needed to prepare the order of the ith customer.
When a customer arrives, he gives the chef his order, and the chef starts preparing it once he is idle. 
The customer waits till the chef finishes preparing his order. 
The chef does not prepare food for more than one customer at a time. 
The chef prepares food for customers in the order they were given in the input.

Return the average waiting time of all customers. 
Solutions within 10^-5 from the actual answer are considered accepted.
*/
func averageWaitingTime(customers [][]int) float64 {
    n := float64(len(customers))
	total_wait_time := float64(0)
	float_customers := make([][]float64, len(customers))
	for idx, v := range customers {
		float_customers[idx] = make([]float64, 2)
		float_customers[idx][0] = float64(v[0])
		float_customers[idx][1] = float64(v[1])
	}

	for idx:=0; idx<len(float_customers)-1; idx++ {
		first_arrival := float_customers[idx][0]
		meal_time := float_customers[idx][1]
		total_wait_time += meal_time
		second_arrival := float_customers[idx+1][0]
		wait_time := meal_time - (second_arrival - first_arrival)
		if wait_time < 0 {
			wait_time = float64(0)
		}
		// Treat the next person as if they arrived at a later time so that the FOLLOWING customer's wait time will be accurate
		float_customers[idx+1][0] += wait_time
		total_wait_time += wait_time
	}
	total_wait_time += float_customers[len(float_customers)-1][1]

	return total_wait_time / n
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. 
A node can only appear in the sequence at most once. 
Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the root of a binary tree, return the maximum path sum of any non-empty path.

Link:
https://leetcode.com/problems/binary-tree-maximum-path-sum/description/
*/
func maxPathSum(root *binary_tree.TreeNode) int {
	max_path_left_only := make(map[*binary_tree.TreeNode]int)
	max_path_right_only := make(map[*binary_tree.TreeNode]int)
	record := []int{root.Val}
	topDownFindRecordPath(root, max_path_left_only, max_path_right_only, record)
    return record[0]
}

func topDownFindRecordPath(root *binary_tree.TreeNode, max_path_left_only map[*binary_tree.TreeNode]int, max_path_right_only map[*binary_tree.TreeNode]int, record []int) {
	if root.Left == nil {
		max_path_left_only[root] = root.Val
		record[0] = max(record[0], root.Val)
	}
	if root.Right == nil {
		max_path_right_only[root] = root.Val
		record[0] = max(record[0], root.Val)
	}
	if root.Left != nil {
		topDownFindRecordPath(root.Left, max_path_left_only, max_path_right_only, record)
		max_path_left_only[root] = max(root.Val, root.Val + max(max_path_left_only[root.Left], max_path_right_only[root.Left]))
		record[0] = max(record[0], max_path_left_only[root])
	}
	if root.Right != nil {
		topDownFindRecordPath(root.Right, max_path_left_only, max_path_right_only, record)
		max_path_right_only[root] = max(root.Val, root.Val + max(max_path_left_only[root.Right], max_path_right_only[root.Right]))
		record[0] = max(record[0], max_path_right_only[root])
	}
	if root.Left != nil && root.Right != nil {
		record[0] = max(record[0], max_path_left_only[root] + max_path_right_only[root] - root.Val)
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a string s, partition s such that every substring of the partition is a palindrome. 
Return all possible palindrome partitioning of s.

Link:
https://leetcode.com/problems/palindrome-partitioning/description/
*/
func partition(s string) [][]string {
	// sols[i][j] answers - "Is s[i:j+1] a palindrome?"
	sols := make(map[int]map[int]bool)
	for i:=0; i<len(s); i++ {
		sols[i] = make(map[int]bool)
	}

	// partitions[i] contains all of the palindrome partitionings of s[0:i+1]
	partitions := make([][][]string, len(s))
	partitions[0] = [][]string{{s[0:1]}}
	for i:=1; i<len(partitions); i++ {
		these_palindrome_partitions := [][]string{}
		for j:=i; j>=0; j-- {
			if isPalindrome(j, i, s, sols) {
				if j == 0 {
					these_palindrome_partitions = append(these_palindrome_partitions, []string{s[j:i+1]})
				} else {
					prev_partitions := partitions[j-1]
					for _, palindrome_list := range prev_partitions {
						new_palindrome_list := []string{}
						new_palindrome_list = append(new_palindrome_list, palindrome_list...)
						new_palindrome_list = append(new_palindrome_list, s[j:i+1])
						these_palindrome_partitions = append(these_palindrome_partitions, new_palindrome_list)
					}
				}
			}
		}
		partitions[i] = these_palindrome_partitions
	}

    return partitions[len(s)-1]
}

/*
Top down helper method to determine if s[start:end+1] is a palindrome
*/
func isPalindrome(start int, end int, s string, sols map[int]map[int]bool) bool {
	_, ok := sols[start][end]
	if !ok {
		// Have NOT solved the problem yet
		if start >= end {
			return true
		} else if s[start] != s[end] {
			return false
		} else {
			sols[start][end] = isPalindrome(start+1, end-1, s, sols)
		}
	}
	return sols[start][end]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a string s, partition s such that every substring of the partition is a palindrome.

Return the minimum cuts needed for a palindrome partitioning of s.

Link:
https://leetcode.com/problems/palindrome-partitioning-ii/description/

Inspiration:
https://leetcode.com/problems/palindrome-partitioning-ii/solutions/42213/easiest-java-dp-solution-97-36/
*/
func minCut(s string) int {
	// is_palindrome[i][j] answers - "Is s[i:j+1] a palindrome?"
	is_palindrome := make(map[int]map[int]bool)
	for i:=0; i<len(s); i++ {
		is_palindrome[i] = make(map[int]bool)
	}

	// sols[i] answers - "What is the minimum number of cuts to partition s[0:i+1] into palindromes?"
	sols := make([]int, len(s))
	for i:=1; i<len(sols); i++ {
		record := math.MaxInt
		for j:=0; j<=i; j++ {
			if isPalindrome(j, i, s, is_palindrome) {
				if j > 0 {
					record = min(record, 1 + sols[j-1])
				} else {
					record = 0
					break
				}
			}
		}
		sols[i] = record
	}

	return sols[len(sols)-1]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

Link:
https://leetcode.com/problems/word-break/description/
*/
func wordBreak(s string, wordDict []string) bool {
	wordMap := make(map[string]bool)
	for _, word := range wordDict {
		wordMap[word] = true
	}
	// From s[idx:], is the substring comprised of words from wordDict?
	comprised := make([]int, len(s))
	return topDownWordBreak(s, 0, comprised, wordMap)
}

/*
Top-down helper method for the function above
*/
func topDownWordBreak(s string, start int, comprised []int, wordMap map[string]bool) bool {
	if comprised[start] == 0 {
		// Need to solve this problem
		comprised[start] = 1
		for i:=start; i<len(s); i++ {
			_, ok := wordMap[s[start:i+1]]
			if ok && (i == len(s)-1 || topDownWordBreak(s, i+1, comprised, wordMap)) {
				comprised[start] = 2
				break
			}
		}
	}
	return comprised[start] == 2
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a string s and a dictionary of strings wordDict, add spaces in s to construct a sentence where each word is a valid dictionary word. 
Return all such possible sentences in any order.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

Link:
https://leetcode.com/problems/word-break-ii/description/
*/
func wordBreak2(s string, wordDict []string) []string {
	wordMap := make(map[string]bool)
	for _, word := range wordDict {
		wordMap[word] = true
	}
	// From s[idx:], is the substring comprised of words from wordDict?
	sentences := make(map[int][]string)
	return topDownWordBreak2(s, 0, sentences, wordMap)
}

/*
Top-down helper method for the function above
*/
func topDownWordBreak2(s string, start int, sentences map[int][]string, wordMap map[string]bool) []string {
	_, ok := sentences[start]
	if !ok {
		// Need to solve this problem
		new_sentences := []string{}
		for i:=start; i<len(s); i++ {
			_, ok := wordMap[s[start:i+1]]
			if ok {
				if i == len(s)-1 {
					new_sentences = append(new_sentences, s[start:i+1])
				} else if len(topDownWordBreak2(s, i+1, sentences, wordMap)) > 0 {
					for _, following_sentence := range topDownWordBreak2(s, i+1, sentences, wordMap) {
						var buffer bytes.Buffer
						buffer.WriteString(s[start:i+1])
						buffer.WriteString(" ")
						buffer.WriteString(following_sentence)
						new_sentences = append(new_sentences, buffer.String())
					}
				}
			}
		}
		sentences[start] = new_sentences
	}
	return sentences[start]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given a string s that consists of lower case English letters and brackets.

Reverse the strings in each pair of matching parentheses, starting from the innermost one.

Your result should not contain any brackets.

Link:
https://leetcode.com/problems/reverse-substrings-between-each-pair-of-parentheses/description/?envType=daily-question&envId=2024-07-11
*/
func reverseParentheses(s string) string {
    idx_stack := linked_list.NewStack[int]()
	chars := make([]byte, len(s))
	for i:=0; i<len(s); i++ {
		chars[i] = s[i]
	}
	idx := 0
	for idx < len(s) {
		for idx < len(s) && s[idx] != '(' {
			idx++
		}
		if idx < len(s) {
			idx_stack.Push(idx)
			idx++
		}
		for !idx_stack.Empty() {
			for s[idx] != '(' && s[idx] != ')' {
				idx++
			}
			if s[idx] == '(' {
				idx_stack.Push(idx)
			} else {
				left := idx_stack.Pop()+1
				right := idx-1
				for left < right {
					chars[left], chars[right] = chars[right], chars[left]
					left++
					right--
				}
			}
			idx++
		}
	}

	var buffer bytes.Buffer
	for _, v := range chars {
		if v != '(' && v != ')' {
			buffer.WriteByte(v)
		}
	}
	return buffer.String()
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given a string s and two integers x and y. 
You can perform two types of operations any number of times.

Remove substring "ab" and gain x points.
For example, when removing "ab" from "cabxbae" it becomes "cxbae".

Remove substring "ba" and gain y points.
For example, when removing "ba" from "cabxbae" it becomes "cabxe".

Return the maximum points you can gain after applying the above operations on s.

Link:
https://leetcode.com/problems/maximum-score-from-removing-substrings/description/?envType=daily-question&envId=2024-07-12

Inspiration:
LeetCode's hint...
*/
func maximumGain(s string, x int, y int) int {
	score := 0
	take_ab := x >= y
	char_stack := linked_list.NewStack[byte]()
	left_over_chars := []byte{}
	for i:=0; i<len(s); i++ {
		if s[i] == 'a'{
			if !char_stack.Empty() && char_stack.Peek() == 'b' && !take_ab {
				// take 'ba'
				char_stack.Pop()
				score += y
			} else {
				char_stack.Push(s[i])
			}
		} else if s[i] == 'b' {
			if !char_stack.Empty() && char_stack.Peek() == 'a' && take_ab {
				// take 'ab'
				char_stack.Pop()
				score += x
			} else {
				char_stack.Push(s[i])
			}
		} else {
			char_stack.Push(s[i])
		}
	}
	// Empty the stack one last time in case it's necessary
	for !char_stack.Empty() {
		left_over_chars = append(left_over_chars, char_stack.Pop())
	}
	for i:=0; i<len(left_over_chars)/2; i++ {
		left := i
		right := len(left_over_chars)-i-1
		left_over_chars[left], left_over_chars[right] = left_over_chars[right], left_over_chars[left]
	}

	// Now go through another pass for the opposite pair
	for i:=0; i<len(left_over_chars); i++ {
		if left_over_chars[i] == 'a'{
			if !char_stack.Empty() && char_stack.Peek() == 'b'{
				// take 'ba'
				char_stack.Pop()
				score += y
			} else {
				char_stack.Push(left_over_chars[i])
			}
		} else if left_over_chars[i] == 'b' {
			if !char_stack.Empty() && char_stack.Peek() == 'a' {
				// take 'ab'
				char_stack.Pop()
				score += x
			} else {
				char_stack.Push(left_over_chars[i])
			}
		} else {
			char_stack.Push(left_over_chars[i])
		}
	}

	return score
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
There are n 1-indexed robots, each having a position on a line, health, and movement direction.

You are given 0-indexed integer arrays positions, healths, and a string directions (directions[i] is either 'L' for left or 'R' for right). 
All integers in positions are unique.

All robots start moving on the line simultaneously at the same speed in their given directions. 
If two robots ever share the same position while moving, they will collide.

If two robots collide, the robot with lower health is removed from the line, and the health of the other robot decreases by one. 
The surviving robot continues in the same direction it was going. 
If both robots have the same health, they are both removed from the line.

Your task is to determine the health of the robots that survive the collisions, in the same order that the robots were given, i.e. final heath of robot 1 (if survived), final health of robot 2 (if survived), and so on. 
If there are no survivors, return an empty array.

Return an array containing the health of the remaining robots (in the order they were given in the input), after no further collisions can occur.

Note: The positions may be unsorted.

Link:
https://leetcode.com/problems/robot-collisions/description/?envType=daily-question&envId=2024-07-13

Inspiration:
The hint on LeetCode
*/
func survivedRobotsHealths(positions []int, healths []int, directions string) []int {
    robots := make([][]int, len(positions))
	for i:=0; i<len(positions); i++ {
		direction := 0
		if directions[i] == 'R' {
			direction++
		}
		// ID, positions, healths, direction
		robots[i] = []int{i+1, positions[i], healths[i], direction}
	}
	// Sort by position
	sort.SliceStable(robots, func(i, j int) bool {
		return robots[i][1] < robots[j][1]
	})

	// Use a stack to analyze the relevant collisions
	robot_stack := linked_list.NewStack[[]int]()
	for i:=0; i<len(robots); i++ {
		for !robot_stack.Empty() && robots[i][3] == 0 && robot_stack.Peek()[3] == 1 && robots[i][2] > 0 {
			if robots[i][2] > robot_stack.Peek()[2] {
				// Last robot destroyed
				robot_stack.Pop()
				robots[i][2]--
			} else if robots[i][2] < robot_stack.Peek()[2]{
				// This robot destroyed
				robots[i][2] = 0
				robot_stack.Peek()[2]--
				break
			} else {
				// Both robots destroyed
				robots[i][2] = 0
				robot_stack.Pop()
				break
			}
		}
		if robots[i][2] > 0 {
			robot_stack.Push(robots[i])
		}
	}

	robot_id_posn := [][]int{}
	for !robot_stack.Empty() {
		robot_id_posn = append(robot_id_posn, []int{robot_stack.Peek()[0], robot_stack.Pop()[2]})
	}
	sort.SliceStable(robot_id_posn, func(i, j int) bool {
		return robot_id_posn[i][0] < robot_id_posn[j][0]
	})

	remaining_healths := make([]int, len(robot_id_posn))
	for idx, robot := range robot_id_posn {
		remaining_healths[idx] = robot[1]
	}
	return remaining_healths
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an integer array nums, find a subarray that has the largest product, and return the product.

The test cases are generated so that the answer will fit in a 32-bit integer.

Link:
https://leetcode.com/problems/maximum-product-subarray/description/
*/
func maxProduct(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}

	zeros := []int{}
	for idx, v := range nums {
		if v == 0 {
			zeros = append(zeros, idx)
		}
	}
	record := 0
	if len(zeros) > 0 {
		if zeros[0] > 0 {
			record = max(record, findNonZeroMaxProduct(0, zeros[0]-1, nums))
		}
		if zeros[len(zeros)-1] < len(nums)-1 {
			record = max(record, findNonZeroMaxProduct(zeros[len(zeros)-1] + 1, len(nums)-1, nums))
		}
		for i:=0; i<len(zeros)-1; i++ {
			if zeros[i+1] - zeros[i] > 1 {
				record = max(record, findNonZeroMaxProduct(zeros[i]+1, zeros[i+1]-1, nums))
			}
		}
	} else {
		record = max(record, findNonZeroMaxProduct(0, len(nums)-1, nums))
	}

    return record
}

/*
Helper method to find the maximum product within a slice of an array that has NO zeros in it
*/
func findNonZeroMaxProduct(start int, end int, nums []int) int {
	if start == end {
		return nums[start]
	}

	// For left-most negative, have left product of it (if left-most negative is start, then that product is 1)
	left_most_product := int64(1) // what we divide by if we remove the left-most negative's left subarray
	true_left_most_product := int64(-1)
	left_most_negative := int64(1)
	negative_found := false
	for i:=start; i<=end; i++ {
		if nums[i] < 0 {
			negative_found = true
			left_most_negative *= int64(nums[i])
			break
		} else {
			true_left_most_product = int64(math.Abs(float64(true_left_most_product)))
			true_left_most_product *= int64(nums[i])
			left_most_product *= int64(nums[i])
		}
	}
	if !negative_found {
		return int(left_most_product)
	}

	// For right-most negative, have right product of it (if right-most negative is end, then that product is 1)
	right_most_product := int64(1) // what we divide by if we remove the right-most negative's subarray
	true_right_most_product := int64(-1) 
	right_most_negative := int64(1)
	for i:=end; i>=start; i-- {
		if nums[i] < 0 {
			right_most_negative *= int64(nums[i])
			break
		} else {
			true_right_most_product = int64(math.Abs(float64(true_right_most_product)))
			true_right_most_product *= int64(nums[i])
			right_most_product *= int64(nums[i])
		}
	}

	// Find total product and all negative values
	product := int64(1)
	for i:=start; i<=end; i++ {
		v := int64(nums[i])
		old_product := product
		product *= int64(v)
		if int64(math.Abs(float64(product) / float64(v))) != int64(math.Abs(float64(old_product)))  {
			// Blew the integer value - according to problem constraints the total product will not be the answer
			product = int64(-1)
			break
		}
	}
	
	// Return the max of the whole product, taking out the left most negative subarray, taking out the right most negative subarray, leaving ONLY the left-most negative subarray, and leaving ONLY the right-most negative subarray
	return int(max(
		product,
		max(
			true_left_most_product, 
			max(
				true_right_most_product,
				max(
					product / left_most_product / left_most_negative,
					product / right_most_product / right_most_negative,
				),
			),
		),
	))
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
The demons had captured the princess and imprisoned her in the bottom-right corner of a dungeon. 
The dungeon consists of m x n rooms laid out in a 2D grid. 
Our valiant knight was initially positioned in the top-left room and must fight his way through dungeon to rescue the princess.

The knight has an initial health point represented by a positive integer. 
If at any point his health point drops to 0 or below, he dies immediately.

Some of the rooms are guarded by demons (represented by negative integers), so the knight loses health upon entering these rooms; other rooms are either empty (represented as 0) or contain magic orbs that increase the knight's health (represented by positive integers).
To reach the princess as quickly as possible, the knight decides to move only rightward or downward in each step.
Return the knight's minimum initial health so that he can rescue the princess.

Note that any room can contain threats or power-ups, even the first room the knight enters and the bottom-right room where the princess is imprisoned.

Link:
https://leetcode.com/problems/dungeon-game/description/
*/
func calculateMinimumHP(dungeon [][]int) int {
	min_health := make([][]int, len(dungeon))
	for i:=0; i<len(dungeon); i++ {
		min_health[i] = make([]int, len(dungeon[i]))
	}
	for r:=len(dungeon)-1; r>=0; r-- {
		for c:=len(dungeon[r])-1; c>=0; c-- {
			min_health[r][c] = max(1, -dungeon[r][c] + 1)
			if r < len(dungeon) - 1 && c < len(dungeon[r])-1 {
				// Look both down and right
				min_health[r][c] = max(1,
									min(
										min_health[r+1][c]-dungeon[r][c],
										min_health[r][c+1]-dungeon[r][c],
									),
								)
			} else if r < len(dungeon) - 1 {
				// Look down
				min_health[r][c] = max(1, min_health[r+1][c]-dungeon[r][c])
			} else if c < len(dungeon[r]) - 1 {
				// Look right
				min_health[r][c] = max(1, min_health[r][c+1]-dungeon[r][c])
			}
		}
	}
    return min_health[0][0]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given the root of a binary tree with n nodes. 
Each node is uniquely assigned a value from 1 to n. 
You are also given an integer startValue representing the value of the start node s, and a different integer destValue representing the value of the destination node t.

Find the shortest path starting from node s and ending at node t. 
Generate step-by-step directions of such path as a string consisting of only the uppercase letters 'L', 'R', and 'U'. 

Each letter indicates a specific direction:
'L' means to go from a node to its left child node.
'R' means to go from a node to its right child node.
'U' means to go from a node to its parent node.

Return the step-by-step directions of the shortest path from node s to node t.

Link:
https://leetcode.com/problems/step-by-step-directions-from-a-binary-tree-node-to-another/description/?envType=daily-question&envId=2024-07-16
*/
func getDirections(root *binary_tree.TreeNode, startValue int, destValue int) string {
	parents := make(map[*binary_tree.TreeNode]*binary_tree.TreeNode)
	nodes := make(map[int]*binary_tree.TreeNode)
	recordParents(root, parents, nodes)
	correct_direction := make(map[*binary_tree.TreeNode]bool)
	findTarget(nil, startValue, destValue, nodes, parents, correct_direction)
	char_queue := linked_list.NewQueue[byte]()
	fillPath(nil, startValue, destValue, nodes, parents, correct_direction, char_queue)
    var buffer bytes.Buffer
	for !char_queue.Empty() {
		buffer.WriteByte(char_queue.Dequeue())
	}
	return buffer.String()
}

/*
Helper method to traverse the tree and record all the parents
*/
func recordParents(root *binary_tree.TreeNode, parents map[*binary_tree.TreeNode]*binary_tree.TreeNode, nodes map[int]*binary_tree.TreeNode) {
	nodes[root.Val] = root
	if root.Left != nil {
		parents[root.Left] = root
		recordParents(root.Left, parents, nodes)
	} 
	if root.Right != nil {
		parents[root.Right] = root
		recordParents(root.Right, parents, nodes)
	}
}

/*
Helper method to start at a source and mark all the nodes needed to reach the destination as true
*/
func findTarget(came_from *binary_tree.TreeNode, startValue int, destValue int, nodes map[int]*binary_tree.TreeNode, parents map[*binary_tree.TreeNode]*binary_tree.TreeNode, correct_direction map[*binary_tree.TreeNode]bool) {
	correct_direction[nodes[startValue]] = false // By default
	if startValue == destValue {
		correct_direction[nodes[startValue]] = true
	} else {
		// Check for parent
		parent, ok := parents[nodes[startValue]]
		if ok && parent != came_from {
			findTarget(nodes[startValue], parent.Val, destValue, nodes, parents, correct_direction)
			if correct_direction[parent] {
				correct_direction[nodes[startValue]] = true
			}
		} 
		// Check for left child
		if nodes[startValue].Left != nil{
			left := nodes[startValue].Left
			if left != came_from {
				findTarget(nodes[startValue], left.Val, destValue, nodes, parents, correct_direction)
				if correct_direction[left] {
					correct_direction[nodes[startValue]] = true
				}
			}
		} 
		// Check for right child
		if nodes[startValue].Right != nil {
			right := nodes[startValue].Right
			if right != came_from {
				findTarget(nodes[startValue], right.Val, destValue, nodes, parents, correct_direction)
				if correct_direction[right] { 
					correct_direction[nodes[startValue]] = true
				}
			}
		}
	}
}

/*
Once all nodes are marked as important or not, traverse from the startValue to the endValue recording each movement in the buffer
*/
func fillPath(came_from *binary_tree.TreeNode, startValue int, destValue int, nodes map[int]*binary_tree.TreeNode, parents map[*binary_tree.TreeNode]*binary_tree.TreeNode, correct_direction map[*binary_tree.TreeNode]bool, char_queue *linked_list.Queue[byte]) {
	if startValue != destValue {
		// Check parent
		parent, ok := parents[nodes[startValue]]
		if ok && correct_direction[parent] && parent != came_from {
			char_queue.Enqueue('U')
			fillPath(nodes[startValue], parent.Val, destValue, nodes, parents, correct_direction, char_queue)
		} else if nodes[startValue].Left != nil && correct_direction[nodes[startValue].Left] && nodes[startValue].Left != came_from {
			char_queue.Enqueue('L')
			fillPath(nodes[startValue], nodes[startValue].Left.Val, destValue, nodes, parents, correct_direction, char_queue)
		} else if nodes[startValue].Right != nil && correct_direction[nodes[startValue].Right] && nodes[startValue].Right != came_from {
			char_queue.Enqueue('R')
			fillPath(nodes[startValue], nodes[startValue].Right.Val, destValue, nodes, parents, correct_direction, char_queue)
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given the root of a binary tree, each node in the tree has a distinct value.

After deleting all nodes with a value in to_delete, we are left with a forest (a disjoint union of trees).

Return the roots of the trees in the remaining forest. You may return the result in any order.

Link:
https://leetcode.com/problems/delete-nodes-and-return-forest/description/?envType=daily-question&envId=2024-07-17
*/
func delNodes(root *binary_tree.TreeNode, to_delete []int) []*binary_tree.TreeNode {
	delete_set := make(map[int]bool)
	for _, v := range to_delete {
		delete_set[v] = true
	}

	// Check if the root needs to be deleted; if not then add it to our list of root nodes
	roots := &[]*binary_tree.TreeNode{}
	_, ok := delete_set[root.Val]
	if !ok {
		*roots = append(*roots, root)
	}

	// Now we are ready for the recursion
	recDelNodes(root, roots, delete_set)

    return *roots
}

/*
Recursive helper method to delete tree nodes
*/
func recDelNodes(root *binary_tree.TreeNode, roots *[]*binary_tree.TreeNode, delete_set map[int]bool) {
	_, ok := delete_set[root.Val]
	// We need to get rid of this node
	if ok {
		if root.Left != nil {
			_, ok = delete_set[root.Left.Val]
			if !ok {
				// This left child - since we are not going to delete it - is a new root
				*roots = append(*roots, root.Left)
			}
			recDelNodes(root.Left, roots, delete_set)
		}
		if root.Right != nil {
			_, ok = delete_set[root.Right.Val]
			if !ok {
				// This right child - since we are not going to delete it - is a new root
				*roots = append(*roots, root.Right)
			}
			recDelNodes(root.Right, roots, delete_set)
		}
	} else {
		// Do not get rid of this node
		if root.Left != nil {
			_, ok = delete_set[root.Left.Val]
			if ok {
				// We need to get rid of this left child
				recDelNodes(root.Left, roots, delete_set)
				root.Left = nil
			} else {
				recDelNodes(root.Left, roots, delete_set)
			}
		}
		if root.Right != nil {
			_, ok = delete_set[root.Right.Val]
			if ok {
				// We need to get rid of this left child
				recDelNodes(root.Right, roots, delete_set)
				root.Right = nil
			} else {
				recDelNodes(root.Right, roots, delete_set)
			}
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given the root of a binary tree and an integer distance. 
A pair of two different LEAF nodes of a binary tree is said to be good if the length of the shortest path between them is less than or equal to distance.

Return the number of good leaf node pairs in the tree.

Link:
https://leetcode.com/problems/number-of-good-leaf-nodes-pairs/description/?envType=daily-question&envId=2024-07-18
*/
func countPairs(root *binary_tree.TreeNode, distance int) int {
	parents := make(map[*binary_tree.TreeNode]*binary_tree.TreeNode)
	nodes := make(map[*binary_tree.TreeNode]bool)
	findParents(root, parents, nodes)

	pairs := 0
	for node := range nodes {
		if node.Left == nil && node.Right == nil {
			// This is a leaf node
			visited := make(map[*binary_tree.TreeNode]bool)
			node_queue := linked_list.NewQueue[*binary_tree.TreeNode]()
			node_queue.Enqueue(node)
			d := 0
			for d < distance && !node_queue.Empty() {
				d++;
				empty_this_many := node_queue.Length()
				for i:=0; i<empty_this_many; i++ {
					// Perform the next breadth expansion
					next_node := node_queue.Dequeue()
					visited[next_node] = true
					if next_node.Left != nil {
						// See if we have visited this left child
						_, ok := visited[next_node.Left]
						if !ok {
							if next_node.Left.Left == nil && next_node.Left.Right == nil {
								// We just found a leaf pair
								pairs++
							}
							// Either way, still contribute to the BFS
							node_queue.Enqueue(next_node.Left)
						}
					}
					if next_node.Right != nil {
						// Similarly, see if we have visited this right child
						_, ok := visited[next_node.Right]
						if !ok {
							if next_node.Right.Left == nil && next_node.Right.Right == nil {
								// Another leaf pair
								pairs++
							}
							// Contribute to BFS
							node_queue.Enqueue(next_node.Right)
						}
					}
					// See if this node has a parent that we can add to our BFS
					parent, ok := parents[next_node]
					if ok {
						_, ok = visited[parent]
						if !ok {
							node_queue.Enqueue(parent)
						}
					}
				}
			}
		}
	}

	// Be sure not to double count each pair
    return pairs / 2
}

/*
Helper method to traverse the tree and record all the parents
*/
func findParents(root *binary_tree.TreeNode, parents map[*binary_tree.TreeNode]*binary_tree.TreeNode, nodes map[*binary_tree.TreeNode]bool) {
	nodes[root] = true
	if root.Left != nil {
		parents[root.Left] = root
		findParents(root.Left, parents, nodes)
	} 
	if root.Right != nil {
		parents[root.Right] = root
		findParents(root.Right, parents, nodes)
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are a professional robber planning to rob houses along a street. 
Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

Link:
https://leetcode.com/problems/house-robber/description/
*/
func rob(nums []int) int {
    rob_prev_prev := nums[len(nums)-1]
	no_rob_prev_prev := 0
	if len(nums) == 1 {
		return rob_prev_prev
	}
	rob_prev := nums[len(nums)-2]
	no_rob_prev := nums[len(nums)-1]
	if len(nums) == 2 {
		return max(rob_prev, no_rob_prev)
	}
	rob_curr := 0
	no_rob_curr := 0
	for i:=len(nums)-3; i>=0; i-- {
		rob_curr = nums[i] + max(no_rob_prev_prev, rob_prev_prev)
		no_rob_curr = max(no_rob_prev, rob_prev)
		rob_prev_prev = rob_prev
		no_rob_prev_prev = no_rob_prev
		rob_prev = rob_curr
		no_rob_prev = no_rob_curr
	}
	return max(rob_curr, no_rob_curr)
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are a professional robber planning to rob houses along a street. 
Each house has a certain amount of money stashed. 
All houses at this place are arranged in a circle. 
That means the first house is the neighbor of the last one. 
Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

Link:
https://leetcode.com/problems/house-robber-ii/description/
*/
func rob2(nums []int) int {
	// Try allowing us to rob the first house
	rob_prev_prev := nums[0]
	no_rob_prev_prev := 0
	if len(nums) == 1 {
		return rob_prev_prev
	}
	rob_prev := nums[1]
	no_rob_prev := nums[0]
	if len(nums) == 2 {
		return max(rob_prev, no_rob_prev)
	}
	rob_curr := max(rob_prev_prev, rob_prev)
	no_rob_curr := max(rob_prev_prev, rob_prev)
	for i:=2; i<len(nums)-1; i++ {
		rob_curr = nums[i] + max(rob_prev_prev, no_rob_prev_prev)
		no_rob_curr = max(rob_prev, no_rob_prev)
		no_rob_prev_prev = no_rob_prev
		rob_prev_prev = rob_prev
		no_rob_prev = no_rob_curr
		rob_prev = rob_curr
	}
	allow_rob_first := max(rob_curr, no_rob_curr)

	// Try allow us to rob the last house
	rob_prev_prev = nums[len(nums)-1]
	no_rob_prev_prev = 0
	rob_prev = nums[len(nums)-2]
	no_rob_prev = nums[len(nums)-1]
	rob_curr = max(rob_prev_prev, rob_prev)
	no_rob_curr = max(rob_prev_prev, rob_prev)
	for i:=len(nums)-3; i>=1; i-- {
		rob_curr = nums[i] + max(rob_prev_prev, no_rob_prev_prev)
		no_rob_curr = max(rob_prev, no_rob_prev)
		no_rob_prev_prev = no_rob_prev
		rob_prev_prev = rob_prev
		no_rob_prev = no_rob_curr
		rob_prev = rob_curr
	}
	allow_rob_last := max(rob_curr, no_rob_curr)

	// Try robbing neither
	return max(allow_rob_first, allow_rob_last)
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
The thief has found himself a new place for his thievery again. 
There is only one entrance to this area, called root.

Besides the root, each house has one and only one parent house. 
After a tour, the smart thief realized that all houses in this place form a binary tree. 
It will automatically contact the police if two directly-linked houses were broken into on the same night.

Given the root of the binary tree, return the maximum amount of money the thief can rob without alerting the police.

Link:
https://leetcode.com/problems/house-robber-iii/description/
*/
func rob3(root *binary_tree.TreeNode) int {
	rob := make(map[*binary_tree.TreeNode]int)
	no_rob := make(map[*binary_tree.TreeNode]int)
	return topDownRob3(root, rob, no_rob)
}

/*
Top-down helper method for the above problem
*/
func topDownRob3(root *binary_tree.TreeNode, rob map[*binary_tree.TreeNode]int, no_rob map[*binary_tree.TreeNode]int) int {
	_, ok := rob[root]
	if !ok {
		// Need to solve this problem
		if root.Left == nil && root.Right == nil {
			rob[root] = root.Val
			no_rob[root] = 0
		} else if root.Left == nil {
			// Only worry about the right child
			no_rob[root] = topDownRob3(root.Right, rob, no_rob)
			rob[root] = root.Val + no_rob[root.Right]
		} else if root.Right == nil {
			// Only worry about the left child
			no_rob[root] = topDownRob3(root.Left, rob, no_rob)
			rob[root] = root.Val + no_rob[root.Left]
		} else {
			// Both children are present
			no_rob[root] = topDownRob3(root.Left, rob, no_rob) + topDownRob3(root.Right, rob, no_rob)
			rob[root] = root.Val + no_rob[root.Left] + no_rob[root.Right]
		}
	}
	return max(rob[root], no_rob[root])
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
There are several consecutive houses along a street, each of which has some money inside. 
There is also a robber, who wants to steal money from the homes, but he refuses to steal from adjacent homes.

The capability of the robber is the maximum amount of money he steals from one house of all the houses he robbed.

You are given an integer array nums representing how much money is stashed in each house. 
More formally, the ith house from the left has nums[i] dollars.

You are also given an integer k, representing the minimum number of houses the robber will steal from. 
It is always possible to steal at least k houses.

Return the minimum capability of the robber out of all the possible ways to steal at least k houses.

Link:
https://leetcode.com/problems/house-robber-iv/description/

Inspiration:
The LeetCode hints were helpful!
*/
func minCapability(nums []int, k int) int {
	min_value := math.MaxInt
	max_value := math.MinInt
	for _, v := range nums {
		min_value = min(min_value, v)
		max_value = max(max_value, v)
	}

	left := min_value
	right := max_value
	for left < right {
		mid := (left + right) / 2 
		if canAchieveCapability(mid, nums, k) {
			// Try decreasing the achievable capability
			right = mid
		} else {
			// One must increase the achievable capability
			left = mid+1
		}
	}

    return left
}

/*
Helper method to solve the above problem
*/
func canAchieveCapability(capability int, nums []int, length int) bool {
	indices := []int{}
	for i:=0; i<len(nums); i++ {
		if nums[i] <= capability {
			indices = append(indices, i)
		}
	}
	if len(indices) < length {
		return false
	} else {
		// We must remove consecutive indices
		i:=0
		removed := 0
		for i<len(indices)-1 {
			if indices[i] == indices[i+1] - 1 {
				// We are only answering the question if the given capability CAN be achieved, so remove the second value
				removed++
				i += 2
			} else {
				i++
			}
		}
		return len(indices) - removed >= length
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an m x n binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

Link:
https://leetcode.com/problems/maximal-square/description/
*/
func maximalSquare(matrix [][]byte) int {
	sols := make([][]int, len(matrix))
	one_streaks_along_row_end_here := make([][]int, len(matrix))
	one_streaks_along_col_end_here := make([][]int, len(matrix))
	for i:=0; i<len(sols); i++ {
		sols[i] = make([]int, len(matrix[0]))
		one_streaks_along_row_end_here[i] = make([]int, len(matrix[0]))
		one_streaks_along_col_end_here[i] = make([]int, len(matrix[0]))
	}
	record := 0
	for i:=0; i<len(matrix); i++ {
		for j:=0; j<len(matrix[i]); j++ {
			if matrix[i][j] == '1' {
				one_streaks_along_col_end_here[i][j]++
				one_streaks_along_row_end_here[i][j]++
				if i > 0 {
					one_streaks_along_col_end_here[i][j] += one_streaks_along_col_end_here[i-1][j]
				}
				if j > 0 {
					one_streaks_along_row_end_here[i][j] += one_streaks_along_row_end_here[i][j-1]
				}
				if i > 0 && j > 0 {
					top_left_square_length := int(math.Sqrt(float64(sols[i-1][j-1])))
					new_square_length := min(top_left_square_length, min(one_streaks_along_row_end_here[i][j-1], one_streaks_along_col_end_here[i-1][j])) + 1
					sols[i][j] = new_square_length * new_square_length
				} else {
					sols[i][j] = 1
				}
				record = max(record, sols[i][j])
			}
		}
	}
    return record
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a string expression of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators. 
You may return the answer in any order.

The test cases are generated such that the output values fit in a 32-bit integer and the number of different results does not exceed 10^4.

Link:
https://leetcode.com/problems/different-ways-to-add-parentheses/description/
*/
func diffWaysToCompute(expression string) []int {
	// Create a map of solutions, which stores all possible ways of computing expression[start][end]
	sols := make(map[int]map[int][]int)

	// Create a list of numbers and operations - note that the operation between nums[i] and nums[i+1] is operations[i]
	nums := []int{}
	operations := []byte{}
	i := 0
	for i < len(expression) {
		var number_buffer bytes.Buffer
		for i < len(expression) && expression[i] != '+' && expression[i] != '-' && expression[i] != '*' {
			number_buffer.WriteByte(expression[i])
			i++
		}
		num := number_buffer.String()
		if len(num) > 0 {
			sols[len(nums)] = make(map[int][]int)
			number, _ := strconv.Atoi(num)
			nums = append(nums, number)
		} else {
			operations = append(operations, expression[i])
			i++
		}
	}

	return topDownDiffWaysToCompute(0, len(nums)-1, nums, operations, sols)
}

/*
Top-down helper method to solve the above problem
*/
func topDownDiffWaysToCompute(startNumIdx int, endNumIdx int, nums []int, operations []byte, sols map[int]map[int][]int) []int {
	_, ok := sols[startNumIdx][endNumIdx]
	if !ok {
		// Need to solve this problem
		if startNumIdx == endNumIdx {
			sols[startNumIdx][endNumIdx] = []int{nums[startNumIdx]}
		} else if startNumIdx == endNumIdx - 1 {
			if operations[startNumIdx] == '-' {
				sols[startNumIdx][endNumIdx] = []int{nums[startNumIdx] - nums[endNumIdx]}
			} else if operations[startNumIdx] == '+' {
				sols[startNumIdx][endNumIdx] = []int{nums[startNumIdx] + nums[endNumIdx]}
			} else {
				sols[startNumIdx][endNumIdx] = []int{nums[startNumIdx] * nums[endNumIdx]}
			}
		} else {
			// Pick some operation to do last
			results := []int{}
			for operation_idx := startNumIdx; operation_idx < endNumIdx; operation_idx++ {
				left_results := topDownDiffWaysToCompute(startNumIdx, operation_idx, nums, operations, sols)
				right_results := topDownDiffWaysToCompute(operation_idx+1, endNumIdx, nums, operations, sols)
				for _, left_value := range left_results {
					for _, right_value := range right_results {
						if operations[operation_idx] == '-' {
							results = append(results, left_value - right_value)
						} else if operations[operation_idx] == '+' {
							results = append(results, left_value + right_value)
						} else {
							results = append(results, left_value * right_value)
						}
					}
				}
			}
			sols[startNumIdx][endNumIdx] = results
		}
	}
	return sols[startNumIdx][endNumIdx]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given two strings s and t, return true if t is an anagram of s, and false otherwise.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

Link:
https://leetcode.com/problems/valid-anagram/description/
*/
func isAnagram(s string, t string) bool {
	// Simply turn both strings into a hash set of characters
    if len(s) != len(t) {
		return false
	} else {
		s_map := make(map[byte]int)
		t_map := make(map[byte]int)
		for i:=0; i<len(s); i++ {
			_, ok := s_map[s[i]]
			if !ok {
				s_map[s[i]] = 1
			} else {
				s_map[s[i]]++
			}
			_, ok = t_map[t[i]]
			if !ok {
				t_map[t[i]] = 1
			} else {
				t_map[t[i]]++
			}
		}
		for c, s_count := range s_map {
			t_count, ok := t_map[c]
			if (!ok) || (t_count != s_count) {
				return false
			}
		}
		return true
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an integer array nums, return the length of the longest strictly increasing 
subsequence.

Link:
https://leetcode.com/problems/longest-increasing-subsequence/description/
(See the algorithm package for my source...)
*/
func lengthOfLIS(nums []int) int {
    return algorithm.LongestIncreasingSubsequence(nums)
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
An ugly number is a positive integer whose prime factors are limited to 2, 3, and 5.

Given an integer n, return the nth ugly number.

Link:
https://leetcode.com/problems/ugly-number-ii/description/
*/
func nthUglyNumber(n int) int {
	two_heap := heap.NewMinHeap[int]()
	two_heap.Insert(2)
	three_heap := heap.NewMinHeap[int]()
	three_heap.Insert(3)
	five_heap := heap.NewMinHeap[int]()
	five_heap.Insert(5)

	// Note that 1 is not divisible by any prime numbers which are not 2, 3, or 5.
	// So 1 is an ugly number!
	val := 1
	seen := make(map[int]bool)
	for i:=1; i<n; i++ {
		// Pick the lowest top value from each heap
		from_two := two_heap.Peek()
		from_three := three_heap.Peek()
		from_five := five_heap.Peek()
		if from_two < from_three && from_two < from_five {
			val = two_heap.Extract()
			_, ok := seen[val * 2]
			if !ok {
				seen[val * 2] = true
				two_heap.Insert(val * 2)
			}
			_, ok = seen[val * 3]
			if !ok {
				seen[val * 3] = true
				two_heap.Insert(val * 3)
			}
			_, ok = seen[val * 5]
			if !ok {
				seen[val * 5] = true
				two_heap.Insert(val * 5)
			}
		} else if from_three < from_two && from_three < from_five {
			val = three_heap.Extract()
			_, ok := seen[val * 3]
			if !ok {
				seen[val * 3] = true
				three_heap.Insert(val * 3)
			}
			_, ok = seen[val * 5]
			if !ok {
				seen[val * 5] = true
				three_heap.Insert(val * 5)
			}
		} else {
			val = five_heap.Extract()
			_, ok := seen[val * 5]
			if !ok {
				seen[val * 5] = true
				five_heap.Insert(val * 5)
			}
		}
	}
    return val
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an integer n, return the least number of perfect square numbers that sum to n.

A perfect square is an integer that is the square of an integer; in other words, it is the product of some integer with itself. 
For example, 1, 4, 9, and 16 are perfect squares while 3 and 11 are not.

Link:
https://leetcode.com/problems/perfect-squares/description/
*/
func numSquares(n int) int {
	sols := make([]int, n+1)
	for i:=0; i<len(sols); i++ {
		sols[i] = math.MaxInt
	}
	sols[0] = 0
	if n > 0 {
		sols[1] = 1
	}
	for i:=2; i<=n; i++ {
		num := 1
		for num * num <= i {
			sols[i] = min(sols[i], 1 + sols[i - num*num])
			num++
		}
	}

    return sols[n]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given a 0-indexed integer array mapping which represents the mapping rule of a shuffled decimal system. 
mapping[i] = j means digit i should be mapped to digit j in this system.

The mapped value of an integer is the new integer obtained by replacing each occurrence of digit i in the integer with mapping[i] for all 0 <= i <= 9.

You are also given another integer array nums. 
Return the array nums sorted in non-decreasing order based on the mapped values of its elements.

Notes:
- Elements with the same mapped values should appear in the same relative order as in the input.
- The elements of nums should only be sorted based on their mapped values and not be replaced by them.

Link:
https://leetcode.com/problems/sort-the-jumbled-numbers/description/?envType=daily-question&envId=2024-07-24
*/
func sortJumbled(mapping []int, nums []int) []int {
	nums_map := make(map[int]int)

	// Perform the conversion
	for _, v := range nums {
		str_v := strconv.Itoa(v)
		digits := make([]int, len(str_v))
		for i:=0; i<len(str_v); i++ {
			// ASCII offset fix
			digits[i] = int(str_v[i] - byte(48))
		}
		var buffer bytes.Buffer
		for _, v := range digits {
			buffer.WriteString(strconv.Itoa(mapping[v]))
		}
		value, _ := strconv.Atoi(buffer.String())
		nums_map[v] = value
	}

	sort.SliceStable(nums, func(i, j int) bool {
		if nums_map[nums[i]] != nums_map[nums[j]] {
			return nums_map[nums[i]] < nums_map[nums[j]]
		} else {
			return i < j
		}
	})
    return nums
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an array of integers nums, sort the array in ascending order and return it.

You must solve the problem without using any built-in functions in O(nlog(n)) time complexity and with the smallest space complexity possible.

Link:
https://leetcode.com/problems/sort-an-array/description/?envType=daily-question&envId=2024-07-25
*/
func sortArray(nums []int) []int {
    return nums
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given n balloons, indexed from 0 to n - 1. 
Each balloon is painted with a number on it represented by an array nums. 
You are asked to burst all the balloons.

If you burst the ith balloon, you will get nums[i - 1] * nums[i] * nums[i + 1] coins. 
If i - 1 or i + 1 goes out of bounds of the array, then treat it as if there is a balloon with a 1 painted on it.

Return the maximum coins you can collect by bursting the balloons wisely.

Link:
https://leetcode.com/problems/burst-balloons/description/
*/
func maxCoins(nums []int) int {
	sols := make([][]map[int]map[int]int, len(nums))
	for i:=0; i<len(nums); i++ {
		sols[i] = make([]map[int]map[int]int, len(nums))
		for j:=i; j<len(nums); j++ {
			sols[i][j] = make(map[int]map[int]int)
		}
	}
    return topDownMaxCoins(1, 0, len(nums)-1, 1, nums, sols)
}

/*
Top-down helper method to solve the above burst balloons problem
*/
func topDownMaxCoins(left int, left_idx int, right_idx int, right int, nums []int, sols [][]map[int]map[int]int) int {
	_, ok := sols[left_idx][right_idx][left]
	if ok {
		_, ok = sols[left_idx][right_idx][left][right]
		if ok {
			return sols[left_idx][right_idx][left][right]
		}
	} else {
		sols[left_idx][right_idx][left] = make(map[int]int)
	}
	// Now we need to solve the problem, because we have not already
	if left_idx == right_idx {
		sols[left_idx][right_idx][left][right] = left * nums[left_idx] * right
	} else {
		// Pick the last balloon to pop
		// Pop left_idx last
		record := left * nums[left_idx] * right + topDownMaxCoins(nums[left_idx], left_idx+1, right_idx, right, nums, sols)
		// Pop right_idx last
		record = max(record, left * nums[right_idx] * right + topDownMaxCoins(left, left_idx, right_idx-1, nums[right_idx], nums, sols))
		// Try popping every balloon in between last
		for last := left_idx+1; last < right_idx; last++ {
			record = max(record, left * nums[last] * right + topDownMaxCoins(left, left_idx, last-1, nums[last], nums, sols) + topDownMaxCoins(nums[last], last+1, right_idx, right, nums, sols))
		}
		sols[left_idx][right_idx][left][right] = record
	}

	return sols[left_idx][right_idx][left][right]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an m x n integers matrix, return the length of the longest increasing path in matrix.

From each cell, you can either move in four directions: left, right, up, or down. 
You may not move diagonally or move outside the boundary (i.e., wrap-around is not allowed).

Link:
https://leetcode.com/problems/longest-increasing-path-in-a-matrix/description/
*/
func longestIncreasingPath(matrix [][]int) int {
	cells := [][]int{}
	for r:=0; r<len(matrix); r++ {
		for c:=0; c<len(matrix[r]); c++ {
			cells = append(cells, []int{matrix[r][c], r, c})
		}
	}
	sort.SliceStable(cells, func(i, j int) bool {
		return cells[i][0] < cells[j][0]
	})

	path_lengths := make([][]int, len(matrix))
	for r:=0; r<len(matrix); r++ {
		path_lengths[r] = make([]int, len(matrix[r]))
		for c:=0; c<len(matrix[r]); c++ {
			path_lengths[r][c] = 1
		}
	}

	record := 1
	for _, cell := range cells {
		v := cell[0]
		r := cell[1]
		c := cell[2]
		if r > 0 {
			// Look up
			v_neighbor := matrix[r-1][c]
			if v_neighbor < v {
				path_lengths[r][c] = max(path_lengths[r][c], path_lengths[r-1][c] + 1)
			}
		}
		if c > 0 {
			// Look left
			v_neighbor := matrix[r][c-1]
			if v_neighbor < v {
				path_lengths[r][c] = max(path_lengths[r][c], path_lengths[r][c-1] + 1)
			}
		}
		if r < len(matrix)-1 {
			// Look down
			v_neighbor := matrix[r+1][c]
			if v_neighbor < v {
				path_lengths[r][c] = max(path_lengths[r][c], path_lengths[r+1][c] + 1)
			}
		}
		if c < len(matrix[r])-1 {
			// Look right
			v_neighbor := matrix[r][c+1]
			if v_neighbor < v {
				path_lengths[r][c] = max(path_lengths[r][c], path_lengths[r][c+1] + 1)
			}
		}
		record = max(record, path_lengths[r][c])
	}

    return record
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
There are n cities numbered from 0 to n-1. 
Given the array edges where edges[i] = [from_i, to_i, weight_i] represents a bidirectional and weighted edge between cities from_i and to_i, and given the integer distanceThreshold.

Return the city with the smallest number of cities that are reachable through some path and whose distance is at most distanceThreshold.
If there are multiple such cities, return the city with the greatest number.

Notice that the distance of a path connecting cities i and j is equal to the sum of the edges' weights along that path.

Link:
https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/description/?envType=daily-question&envId=2024-07-26
*/
func findTheCity(n int, edges [][]int, distanceThreshold int) int {
	counts := make([]int, n)
	record := math.MaxInt

	neighbors := make([][][]int, n)
	for i:=0; i<n; i++ {
		neighbors[i] = [][]int{}
	}
	for _, edge := range edges {
		from := edge[0]
		to := edge[1]
		weight := edge[2]
		neighbors[from] = append(neighbors[from], []int{to, weight})
		neighbors[to] = append(neighbors[to], []int{from, weight})
	}

	// Perform breadth-first search from all nodes
	for source:=0; source<n; source++ {
		determined := make([]bool, n)
		distances := make([]int, n)
		node_heap := heap.NewCustomMinHeap(func(first []int, second []int) bool {
			return first[1] < second[1] 
		})
		node_heap.Insert([]int{source, 0})
		for !node_heap.Empty() {
			next_edge := node_heap.Extract()
			to := next_edge[0]
			if determined[to] {
				continue
			}
			cost_to_reach := next_edge[1]
			determined[to] = true
			distances[to] = cost_to_reach
			for _, connection := range neighbors[to] {
				next := connection[0]
				weight := connection[1]
				node_heap.Insert([]int{next, distances[to] + weight})
			}
		}

		count_within_threshold := 0
		for _, distance := range distances {
			if distance <= distanceThreshold {
				count_within_threshold++
			}
		}
		record = min(record, count_within_threshold)
		counts[source] = count_within_threshold
	}

	for i:=n-1; i>=0; i-- {
		if counts[i] == record {
			return i
		}
	}
    return 0
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
You are given two 0-indexed strings source and target, both of length n and consisting of lowercase English letters. 
You are also given two 0-indexed character arrays original and changed, and an integer array cost, where cost[i] represents the cost of changing the character original[i] to the character changed[i].

You start with the string source. 
In one operation, you can pick a character x from the string and change it to the character y at a cost of z if there exists any index j such that cost[j] == z, original[j] == x, and changed[j] == y.

Return the minimum cost to convert the string source to the string target using any number of operations. 
If it is impossible to convert source to target, return -1.

Note that there may exist indices i, j such that original[j] == original[i] and changed[j] == changed[i].

Link:
https://leetcode.com/problems/minimum-cost-to-convert-string-i/description/?envType=daily-question&envId=2024-07-27

Inspiration:
The LeetCode editorial was helpful!

PLEA FOR HELP:
This solution does not work when you submit it on LeetCode, and quite frankly I do not know why.
I see no difference between this code and the code that is shown in the LeetCode editorial.
If anyone can find my error, I'd love it if you could let me know!
*/
func minimumCost(source string, target string, original []byte, changed []byte, cost []int) int64 {
	// We need to shortest path from each character in source to each character in target
	shortest_paths := make([][]int64, 26)
	for i:=0; i<26; i++ {
		shortest_paths[i] = make([]int64, 26)
        for j:=0; j<26; j++ {
            shortest_paths[i][j] = int64(math.MaxInt32)
        }
	}
	
	for idx, b1 := range original {
		c1 := b1 - 'a'
		c2 := changed[idx] - 'a'
		edge_weight := cost[idx]
		shortest_paths[c1][c2] = min(shortest_paths[c1][c2], int64(edge_weight))
	}

	// Now we build up dynamically to solve all shortest paths
	for j:=0; j<26; j++ {
		for i:=0; i<26; i++ {
			for k:=0; k<26; k++ {
				shortest_paths[i][k] = min(shortest_paths[i][k],
									shortest_paths[i][j] + shortest_paths[j][k])
			}
		}
	}
	
	total_cost := int64(0)
	for i:=0; i<len(source); i++ {
		if source[i] == target[i] {
			continue
		}
		idx1 := source[i] - 'a'
		idx2 := target[i] - 'a'
		path_length := shortest_paths[idx1][idx2]
		if path_length >= int64(math.MaxInt32/2) { // Not reachable
			return -1
		} else {
			total_cost += path_length
		}
	}

	return total_cost
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
A city is represented as a bi-directional connected graph with n vertices where each vertex is labeled from 1 to n (inclusive). 
The edges in the graph are represented as a 2D integer array edges, where each edges[i] = [u_i, v_i] denotes a bi-directional edge between vertex u_i and vertex v_i. 
Every vertex pair is connected by at most one edge, and no vertex has an edge to itself. 
The time taken to traverse any edge is 'time' minutes.

Each vertex has a traffic signal which changes its color from green to red and vice versa every 'change' minutes. 
All signals change at the same time. 
You can enter a vertex at any time, but can leave a vertex only when the signal is green. 
You cannot wait at a vertex if the signal is green.

The second minimum value is defined as the smallest value strictly larger than the minimum value.

For example the second minimum value of [2, 3, 4] is 3, and the second minimum value of [2, 2, 4] is 4.
Given n, edges, time, and change, return the second minimum time it will take to go from vertex 1 to vertex n.

Notes:
- You can go through any vertex any number of times, including 1 and n.
- You can assume that when the journey starts, all signals have just turned green.

Link:
https://leetcode.com/problems/second-minimum-time-to-reach-destination/?envType=daily-question&envId=2024-07-28

Inspiration:
The LeetCode Editorial for this question was very helpful!
*/
func secondMinimum(n int, edges [][]int, time int, change int) int {
	// First make a connectivity list to represent our graph
	connections := make([][]int, n)
	for _, edge := range edges {
		connections[edge[0]-1] = append(connections[edge[0]-1], edge[1])
		connections[edge[1]-1] = append(connections[edge[1]-1], edge[0])
	}

	type connection struct {
		node int
		cost int
	}

	// Plan of attack: Perform a modified Djikstra's algorithm to second shortest path reaching reaching the target n
	shortest := make([]int, n)
	second_shortest := make([]int, n)
	seen := make([]int, n)
	second_shortest[0] = math.MaxInt
	for i:=1; i<n; i++ {
		shortest[i] = math.MaxInt
		second_shortest[i] = math.MaxInt
	}
	node_heap := heap.NewCustomMinHeap(func(first connection, second connection) bool {
		return first.cost < second.cost
	})
	node_heap.Insert(connection{1, 0})
	for !node_heap.Empty() {
		next := node_heap.Extract()
		seen[next.node-1]++
		if seen[next.node-1] == 2 && next.node == n {
			break
		}
		time_to_here := next.cost
		for _, neighbor := range connections[next.node-1] {
			if seen[neighbor-1] < 2 {
				// We can update this next neighbor
				time_to_neighbor := 0
				if (time_to_here / change) % 2 == 1 {
					// We are currently on RED - we have to wait until the next switch occurs
					time_to_neighbor += change * (time_to_here / change + 1) + time
					// The total time is the total number of changes that must occur before we can traverse this next edge, PLUS the time it takes to traverse this next edge
				} else {
					// We are currently on GREEN - we can just progress onto the next edge to reach neighbor
					time_to_neighbor += time_to_here + time
				}
				// See if we need to update shortest or second shortest
				if shortest[neighbor-1] > time_to_neighbor {
					second_shortest[neighbor-1] = shortest[neighbor-1] 
					shortest[neighbor-1] = time_to_neighbor
					node_heap.Insert(connection{neighbor, time_to_neighbor})
				} else if shortest[neighbor-1] < time_to_neighbor && second_shortest[neighbor-1] > time_to_neighbor {
					// Remember, we need the shortest and second shortest times to be distinct
					second_shortest[neighbor-1] = time_to_neighbor
					node_heap.Insert(connection{neighbor, time_to_neighbor})
				}
				// If the time to reach this neighbor does no updating on the neighbor's behalf, then don't push the neighbor with its same value onto the heap
			}
		}
	}

	// Now just return the second minimum distance associated with node n
    return second_shortest[n-1]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.

Link:
https://leetcode.com/problems/counting-bits/description/
*/
func countBits(n int) []int {
    return []int{}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given an integer n, break it into the sum of k positive integers, where k >= 2, and maximize the product of those integers.

Return the maximum product you can get.

Link:
https://leetcode.com/problems/integer-break/description/
*/
func integerBreak(n int) int {
	return 0
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given two strings s and t, return true if s is a subsequence of t, or false otherwise.

A subsequence of a string is a new string that is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. 
(i.e., "ace" is a subsequence of "abcde" while "aec" is not).

Link:
https://leetcode.com/problems/is-subsequence/description/
*/
func isSubsequence(s string, t string) bool {
    return false
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

