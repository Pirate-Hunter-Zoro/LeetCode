package heap

type Ordered interface {
	Greater(other Ordered) bool
	Equal(other Ordered) bool
	Less(other Ordered) bool
}