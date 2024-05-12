package graph

type Edge struct {
	To     *Node
	Weight int
}

type Node struct {
	Id          int
	IsVisited   bool
	Cost        int
	Connections []*Edge
}