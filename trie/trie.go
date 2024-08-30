package trie

/*
A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. 
There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:
- Trie() Initializes the trie object.
- void insert(String word) Inserts the string word into the trie.
- boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
- boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.

Link:
https://leetcode.com/problems/implement-trie-prefix-tree/description/
*/

// Useful node data structure for the preceding Trie implementation
type trieNode struct {
	id byte
	children map[byte]*trieNode
	end_of_word map[string]bool
}

type Trie struct {
    root *trieNode
}

/*
Initializes the trie object.
*/
func Constructor() Trie {
    return Trie{root: &trieNode{id: '\n', children: make(map[byte]*trieNode)}}
}

/*
Inserts the string word into the trie.
*/
func (trie *Trie) Insert(word string)  {
    current := trie.root
	idx := 0
	for idx < len(word) {
		_, ok := current.children[word[idx]]
		if !ok {
			current.children[word[idx]] = &trieNode{id: word[idx], children: make(map[byte]*trieNode), end_of_word: make(map[string]bool)}
		}
		current = current.children[word[idx]]
		idx++
	}
	current.end_of_word[word] = true
}

/*
Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
*/
func (trie *Trie) Search(word string) bool {
    current := trie.root
	idx := 0
	for idx < len(word) {
		_, ok := current.children[word[idx]]
		if !ok {
			return false
		}
		current = current.children[word[idx]]
		idx++
	}
	// If the current node does not correspond with the END of the given word, then return false - the given word is only a prefix of another insertion
	_, ok := current.end_of_word[word]
	return ok
}

/*
Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.
*/
func (trie *Trie) StartsWith(prefix string) bool {
    current := trie.root
	idx := 0
	for idx < len(prefix) {
		_, ok := current.children[prefix[idx]]
		if !ok {
			return false
		}
		current = current.children[prefix[idx]]
		idx++
	}
	return true
}