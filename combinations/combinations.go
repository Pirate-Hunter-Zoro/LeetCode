package combinations

import "leetcode/modulo"

var sols = make([][]int, 1000)

func Choose(n int, k int) int {
	if k > n {
		return 0
	} else {
		if len(sols[n]) < 1000 {
			sols[n] = make([]int, 1000)
		}
		if sols[n][k] == 0 {
			if k == n || k == 0 {
				sols[n][k] = 1
			} else {
				sols[n][k] = Choose(n-1, k-1) + Choose(n-1, k)
			}
		}
		return sols[n][k]
	}
}

var mod_sols = make(map[int]map[int]int)

func ChooseMod(n int, k int) int {
	if n == k || k == 0 {
		return 1
	} else {
		_, ok := mod_sols[n]
		if !ok {
			mod_sols[n] = make(map[int]int)
		}
		_, ok = mod_sols[n][k]
		if !ok {
			// Need to solve this problem
			mod_sols[n][k] = modulo.ModularAdd(ChooseMod(n-1, k-1), ChooseMod(n-1, k))
		}
		return mod_sols[n][k]
	}
}