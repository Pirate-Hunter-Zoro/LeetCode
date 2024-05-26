package modulo

var mod = int64(1000000007)

func ModularAdd(a int, b int) int {
	a_long := int64(a)
	b_long := int64(b)
	return int(((a_long % mod) + (b_long % mod)) % mod)
}

func ModularMultiply(a int, b int) int {
	a_long := int64(a)
	b_long := int64(b)
	return int(((a_long % mod) * (b_long % mod)) % mod)
}

func ModularSubtract(a int, b int) int {
	a_long := int64(a)
	b_long := int64(b)
	a_long = a_long % mod
	b_long = b_long % mod
	if a_long >= b_long {
		return int(a_long - b_long)
	} else {
		return int((a_long + (mod - b_long)) % mod)
	}
}

func ModularPow(base int, power int) int {
	if power == 0 {
		return 1
	}
	base_long := int64(base)
	if power == 1 {
		return int(base_long % mod)
	} else {
		half_power := power / 2
		square_root := ModularPow(base, half_power)

		if power % 2 == 1 {
			return ModularMultiply(ModularMultiply(square_root, square_root), base)
		} else {
			return ModularMultiply(square_root, square_root)
		}
	}
}