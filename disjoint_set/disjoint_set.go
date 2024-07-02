package disjointset

type Node[T comparable] struct {
	val    T
	parent *Node[T]
}

func NewNode[T comparable](t T) *Node[T] {
	return &Node[T]{
		val:    t,
		parent: nil,
	}
}

func (n *Node[T]) Value() T {
	return n.val
}

func (n *Node[T]) RootValue() T {
	n.collapse()
	if n.parent == nil {
		return n.val
	} else {
		return n.parent.val
	}
}

func (n *Node[T]) Isolate() {
	n.parent = nil
}

func (n *Node[T]) Join(other *Node[T]) {
	n.collapse()
	other.collapse()
	if n.RootValue() != other.RootValue() {
		if n.parent == nil && other.parent == nil {
			other.parent = n
		} else if n.parent == nil {
			n.parent = other
			n.collapse()
		} else if other.parent == nil {
			other.parent = n
			other.collapse()
		} else {
			n.Join(other.parent)
			other.collapse()
		}
	}
}

func (n *Node[T]) collapse() {
	if n.parent != nil {
		n.parent.collapse()
		if n.parent.parent != nil {
			n.parent = n.parent.parent
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////

// You may only use Nodes - that's all well and good - but using this SetOfSets object will prohibit a repeat node value which may sometimes be useful

// Use an underlying hash map to prevent repeat nodes from existing
type SetOfSets[T comparable] struct {
	nodes map[T]*Node[T]
}

func NewSetOfSets[T comparable]() *SetOfSets[T] {
	return &SetOfSets[T]{make(map[T]*Node[T])}
}

func (s *SetOfSets[T]) Clear() {
	s.nodes = make(map[T]*Node[T])
}

// Return this node if it exists.
func (s *SetOfSets[T]) GetNode(v T) *Node[T] {
	_, ok := s.nodes[v]
	if !ok {
		return nil
	}
	return s.nodes[v]
}

// Make this node if it does not already exist
func (s *SetOfSets[T]) MakeNode(v T) {
	_, ok := s.nodes[v]
	if !ok {
		s.nodes[v] = NewNode[T](v)
	}
}