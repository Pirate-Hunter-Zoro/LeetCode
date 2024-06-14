package binary_tree

import "testing"

func TestBSTIterator(t *testing.T) {
	root := NewTree([]int{7, 3, 15, NULL, NULL, 9, 20})
	iterator := BSTIteratorConstructor(root)
	v := iterator.Next()
	if v != 3 {
		t.Fatalf("Error - expected 3, but got %d...", v)
	} 
	v = iterator.Next()
	if v != 7 {
		t.Fatalf("Error - expected 7, but got %d...", v)
	}
	h := iterator.HasNext()
	if !h {
		t.Fatalf("Error - expected %t, but got %t...", !h, h)
	}
	v = iterator.Next()
	if v != 9 {
		t.Fatalf("Error - expected 9, but got %d...", v)
	}
	h = iterator.HasNext()
	if !h {
		t.Fatalf("Error - expected %t, but got %t...", !h, h)
	}
	v = iterator.Next()
	if v != 15 {
		t.Fatalf("Error - expected 15, but got %d...", v)
	}
	h = iterator.HasNext()
	if !h {
		t.Fatalf("Error - expected %t, but got %t...", !h, h)
	}
	v = iterator.Next()
	if v != 20 {
		t.Fatalf("Error - expected 20, but got %d...", v)
	}
	h = iterator.HasNext()
	if h {
		t.Fatalf("Error - expected %t, but got %t...", !h, h)
	}
}
