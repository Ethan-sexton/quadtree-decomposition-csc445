class QuadTree:
    nodes = []
    def __init__(self, nodes):
        self.nodes = [nodes[i] for i in range(len(nodes))]