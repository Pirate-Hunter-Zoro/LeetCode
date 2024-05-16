package graph

type Edge struct {
	To     *GraphNode
	Weight int
}

type GraphNode struct {
	Id          int
	IsVisited   bool
	Cost        int
	Connections []*Edge
}
