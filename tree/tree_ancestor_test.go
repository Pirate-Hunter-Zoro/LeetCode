package tree

import "testing"

func TestTreeAncestor(t *testing.T) {
	tree_ancestor := Constructor(7, []int{-1,0,0,1,1,2,2})
	a := tree_ancestor.GetKthAncestor(3, 1)
	if a != 1 {
		t.Fatalf("Error - expected %d but got %d", 1, a)
	}
	a = tree_ancestor.GetKthAncestor(5, 2)
	if a != 0 {
		t.Fatalf("Error - expected %d but got %d", 0, a)
	}
	a = tree_ancestor.GetKthAncestor(6, 3)
	if a != -1 {
		t.Fatalf("Error - expected %d but got %d", -1, a)
	}

	tree_ancestor = Constructor(5, []int{-1,0,0,1,2})
	a = tree_ancestor.GetKthAncestor(3, 5)
	if a != -1 {
		t.Fatalf("Error - expected %d but got %d", -1, a)
	}
	a = tree_ancestor.GetKthAncestor(3, 2)
	if a != 0 {
		t.Fatalf("Error - expected %d but got %d", 0, a)
	}
	a = tree_ancestor.GetKthAncestor(2, 2)
	if a != -1 {
		t.Fatalf("Error - expected %d but got %d", -1, a)
	}
	a = tree_ancestor.GetKthAncestor(0, 2)
	if a != -1 {
		t.Fatalf("Error - expected %d but got %d", -1, a)
	}
	a = tree_ancestor.GetKthAncestor(2, 1)
	if a != 0 {
		t.Fatalf("Error - expected %d but got %d", 0, a)
	}
}