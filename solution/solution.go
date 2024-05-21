package solution

import (
	"leetcode/algorithm"
	"leetcode/binary_tree"
	disjointset "leetcode/disjoint_set"
	"leetcode/euclidean"
	"leetcode/graph"
	"leetcode/heap"
	"leetcode/linked_list"
	"leetcode/list_node"
	"leetcode/modulo"
	"math"
	"sort"
	"strconv"
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
		nodes[idx] = s.MakeNode(n)
	}

	for idx, n := range nums {
		prime_factors := euclidean.GetPrimeFactors(n)
		for _, p := range prime_factors {
			nodes[idx].Join(s.MakeNode(p))
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
				additional = min(additional, topDownPutCamera(root.Right, put_camera, no_camera) + min(topDownPutCamera(root.Left, put_camera, no_camera), topDownNoCamera(root.Left, put_camera, no_camera)))
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
	for i:=0; i<n; i++ {
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
				if (num_children_want_xor % 2 == 0) && nums[root] < xors[root] {
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
				} else if (num_children_want_xor % 2 == 1) && xors[root] < nums[root] {
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
				if (num_children_want_xor % 2 == 0) && (xors[root] > nums[root]) {
					// Then no use exploring forcing the root or any children to be xored/not-xored against its wishes
					node_loss = 0
					child_loss = 0
				} else if (num_children_want_xor % 2 == 1) && (nums[root] > xors[root]) {
					// Same
					node_loss = 0
					child_loss = 0
				}
				dp[1][root] -= int64(min(node_loss, child_loss))

				node_loss = old_node_loss
				child_loss = old_child_loss
				// Similarly, if we will NOT xor the root with its parent
				// Xoring an ODD number of children will leave the root xored - we could add or take away a child xor to change that
				if (num_children_want_xor % 2 == 1) && (xors[root] > nums[root]) {
					// Then no use exploring forcing the root or any children to be xored/not-xored against its wishes
					node_loss = 0
					child_loss = 0
				} else if (num_children_want_xor % 2 == 0) && (nums[root] > xors[root]) {
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

