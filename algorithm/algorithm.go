package algorithm

import (
	"cmp"
)

/*
Runs in O(nlog(n))
Source:
https://www.youtube.com/watch?v=S9oUiVYEq7E
*/
func LongestIncreasingSubsequence[T cmp.Ordered](values []T) int {
	length_minus_one := 0
	// The following array stores the index of the minimum possible last value for any increasing subsequence of a particular length
	length_end_indices := make([]int, len(values))
	for i:=1; i<len(values); i++ {
		if values[i] < values[length_end_indices[0]] {
			length_end_indices[0] = i
		} else if values[i] > values[length_end_indices[length_minus_one]] {
			length_minus_one++
			length_end_indices[length_minus_one] = i
		} else {
			this_val := values[i]
			// Binary search for the last (increasing subsequence end for a given length) immediately preceding a value less than this_val
			left := 1
			right := length_minus_one + 1
			for left < right {
				mid := (left + right) / 2
				val := values[length_end_indices[mid]]
				prev_val := values[length_end_indices[mid-1]]
				if val < this_val {
					// Look right
					left = mid + 1
				} else if prev_val < this_val {
					// Stop here
					length_end_indices[mid] = i
					// The kicker is that this might replace the index for the lowest value of the highest length increasing subsequence so far, which could let us extend our record in the future
					break
				} else {
					right = mid
				}
			}
		}
	}

	return length_minus_one + 1
}

/*
Returns the index of the target value in the (sorted) array
*/
func BinarySearch[T cmp.Ordered](values []T, target T) int {
	left := 0
	right := len(values)
	for left < right {
		mid := (left + right) / 2
		if values[mid] < target {
			// look right
			left = mid + 1
		} else if values[mid] > target {
			// look left
			right = mid
		} else {
			// we are done
			return mid
		}
	}

	return -1
}

/*
Returns the index of the greatest element which does not exceed target - values contains unique elements
*/
func BinarySearchMeetOrLower[T cmp.Ordered](values []T, target T) int {
	left := 0
	right := len(values)
	for left < right {
		mid := (left + right) / 2
		if values[mid] < target {
			// we may return this if the next value is greater than target
			if (mid == len(values) - 1 || values[mid+1] > target) {
				return mid
			}
			// Otherwise
			left = mid + 1
		} else if values[mid] > target {
			// look left
			right = mid
		} else {
			// we are done
			return mid
		}
	}

	return -1
}