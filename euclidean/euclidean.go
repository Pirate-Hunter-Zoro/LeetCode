package euclidean

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

var prime_sieve = make([]int, 99_998)

var primes = initializePrimes()

func initializePrimes() []int {
	primes := []int{}

	for i := 0; i<len(prime_sieve); i++ {
		prime_sieve[i] = i+2
	}
	curr_idx := 0
	for curr_idx < len(prime_sieve) {
		for prime_sieve[curr_idx] < 0 {
			curr_idx++
			if curr_idx >= len(prime_sieve) {
				break
			}
		}
		if curr_idx < len(prime_sieve) {
			val := prime_sieve[curr_idx]
			primes = append(primes, val)
			next := val * val
			for next < len(prime_sieve) + 2 {
				if prime_sieve[next-2] > 0 {
					prime_sieve[next-2] = -prime_sieve[next-2]
				}
				next += val
			}
		}
		curr_idx++
	}

	return primes
}

var prime_factors = make(map[int][]int)

func GetPrimeFactors(n int) []int {
	if (n == 1) {
		return []int{}
	}
	_, ok := prime_factors[n]
	if !ok {
		// We need to find the prime factors since we have not solved this problem yet
		for _, p := range primes {
			if (n % p) == 0 {
				v := n / p
				for (v % p) == 0 {
					v = v / p
				}
				prev := GetPrimeFactors(v)
				prev = append(prev, p)
				prime_factors[n] = prev
				break
			}
		}
	}

	return prime_factors[n]
}