package prime_numbers

func GCD(n int, k int) int {
	if n == k {
		return n
	} else if n > k {
		if n % k == 0 {
			return k
		} else {
			r := n % k
			return GCD(k, r)
		}
	} else {
		if k % n == 0 {
			return n
		} else {
			r := k % n
			return GCD(n, r)
		}
	}
}