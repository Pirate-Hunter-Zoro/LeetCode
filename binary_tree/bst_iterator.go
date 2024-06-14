package binary_tree

/*
Implement the BSTIterator class that represents an iterator over the in-order traversal of a binary search tree (BST):

BSTIterator(TreeNode root) Initializes an object of the BSTIterator class. The root of the BST is given as part of the constructor. The pointer should be initialized to a non-existent number smaller than any element in the BST.
boolean hasNext() Returns true if there exists a number in the traversal to the right of the pointer, otherwise returns false.
int next() Moves the pointer to the right, then returns the number at the pointer.
Notice that by initializing the pointer to a non-existent smallest number, the first call to next() will return the smallest element in the BST.

You may assume that next() calls will always be valid. That is, there will be at least a next number in the in-order traversal when next() is called.

Link:
https://leetcode.com/problems/binary-search-tree-iterator/description/
*/
type BSTIterator struct {
	curr_index  int
	max_index 	int
	values 		[]int
}

func BSTIteratorConstructor(root *TreeNode) BSTIterator {
	values := inorderTraversal(root)
	return BSTIterator{0, len(values)-1, values}
}
// Helper method to traverse a tree inorder
func inorderTraversal(root *TreeNode) []int {
	values := []int{}
	if root.Left != nil {
		values = append(values, inorderTraversal(root.Left)...)
	} 
	values = append(values, root.Val)
	if root.Right != nil {
		values = append(values, inorderTraversal(root.Right)...)
	}
	return values
}

func (it *BSTIterator) Next() int {
	val := it.values[it.curr_index]
	it.curr_index++
	return val
}

func (it *BSTIterator) HasNext() bool {
	return it.curr_index <= it.max_index
}

/*
 * Your BSTIterator object will be instantiated and called as such:
 * obj := Constructor(root);
 * param_1 := obj.Next();
 * param_2 := obj.HasNext();
 */
