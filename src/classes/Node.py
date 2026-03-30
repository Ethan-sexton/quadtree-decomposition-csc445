class Node:
    children = []
    def __init__(self, val, children):
        self.value = val
        if len(children) <= 4 and len(children) >= 0:
            self.children = [children[i] for i in range(len(children))]
        else:
            raise ValueError("Invalid number of children given")
    
    def hasChildren(self):
        return len(self.children) > 0