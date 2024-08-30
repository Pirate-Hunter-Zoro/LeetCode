package trie

import "testing"

func TestTrie(t *testing.T) {
	trie := Constructor()
	trie.Insert("apple")
	if !trie.Search("apple") {
		t.Fatalf("Error - expected %t when searching trie for %s, but got %t", true, "apple", false)
	}
	if trie.Search("app") {
		t.Fatalf("Error - expected %t when searching trie for %s, but got %t", false, "app", true)
	}
	if !trie.StartsWith("app") {
		t.Fatalf("Error - expected %t when searching trie for %s, but got %t", true, "apple", false)
	}
	trie.Insert("app")
	if !trie.Search("app") {
		t.Fatalf("Error - expected %t when searching trie for %s, but got %t", true, "apple", false)
	}
}