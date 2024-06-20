package float_rounding

import "math"

/*
Helper function to round a float

Source:
https://gosamples.dev/round-float/
*/
func RoundFloat(val float64, precision uint) float64 {
	ratio := math.Pow(10, float64(precision))
	return math.Round(val*ratio) / ratio
}