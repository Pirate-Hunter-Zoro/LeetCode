package solution

import (
	"leetcode/algorithm"
	disjointset "leetcode/disjoint_set"
	"leetcode/euclidean"
	"leetcode/graph"
	"leetcode/heap"
	"leetcode/list_node"
	"leetcode/modulo"
	"leetcode/queue"
	"leetcode/stack"
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
	node_queue := queue.New[int]()
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
		people_nodes[i] = disjointset.New[int](i)
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
	s := disjointset.NewSet[int]()

	if len(nums) == 1 {
		return true
	}

	// Set of primes we have run into so far
	// If we are divisible by any of them, then we can reach a value in the right side of the array from our current index, and have not broken the streak
	nodes := make([]*disjointset.Node[int], len(nums))
	for idx, n := range nums {
		if n == 1 {
			return false
		}
		nodes[idx] = s.New(n)
	}

	seen_nums := make(map[int]bool)

	for idx, n := range nums {
		_, ok := seen_nums[n]
		if !ok { // If we have not yet seen this number
			prime_factors := euclidean.GetPrimeFactors(n)
			for _, p := range prime_factors {
				nodes[idx].Join(s.New(p))
			}
			seen_nums[n] = true
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
	idx_stack := stack.New[int]()

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
	idx_stack := stack.New[int]()
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
	q := queue.New[int]()
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
	st := stack.New[int]()
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
			nodes[i][j] = disjointset.New(coordinate{i, j})
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
			nodes[i][j] = disjointset.New(coordinate{i, j})
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
		underlying_node *graph.Node
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
				underlying_node: &graph.Node{
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

	underlying_nodes := make([]*graph.Node, len(lock_nodes))
	for idx, lock_node := range lock_nodes {
		if lock_node != nil {
			underlying_nodes[idx] = lock_node.underlying_node
		}
	}

	// No need for Djikstra's algorithm here as all edge weights are 1 - BFS will do
	q := queue.New[*graph.Node]()
	lock_nodes[0].underlying_node.IsVisited = true
	q.Enqueue(lock_nodes[0].underlying_node)
	length := 0
	for !q.Empty() {
		empty_this_many := q.Length()
		for i:=0; i<empty_this_many; i++ {
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
