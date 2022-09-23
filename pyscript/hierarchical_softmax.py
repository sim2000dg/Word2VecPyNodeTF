import bisect

# This class is the brick of our hierarchical softmax implementation
class Node:
    def __init__(self, key:str|int, frequency:int, right_child:"None|Node"=None, left_child:"None|Node"=None) -> None:
        self.key = key
        self.frequency = frequency
        self.right_child = right_child
        self.left_child = left_child


# This classic recursive method returns the path in reverse order along with a 
# succession of 1 and 0 integers which represents the Huffman encoding

    def path(self, query:str, log:list[int, str]) -> bool:
        
        if self.key == query: 
            return True
        
        if self.right_child is None and self.left_child is None:
            return False
        
        if self.right_child.path(query, log):
            log.append((1, self.key))
            return True
        
        elif self.left_child.path(query, log):
            log.append((0, self.key))
            return True
        
        return False

# Usual thing when working with trees, class tree with a pointer to the root node
class Tree:
    def __init__(self, root:Node) -> None:
        self. root = root 
        self.lookup_dict = dict()
        
    # The following factory method allows us to build an Huffman Tree
    @classmethod     
    def huffman_builder(cls, word_fr:list[tuple[int, int]]) -> "Tree":
        # We initialize each word as a single Node object. Each of the resuling nodes is put 
        # into a list and the list is sorted according to the frequency of each word (ascending order)
        word_fr = [Node(key = x, frequency = y) for x, y in word_fr]
        word_fr.sort(key = lambda x: x.frequency, reverse = True)
        
        # Now the builder loop starts. The condition for loop  termination is simply len(word_fr) == 1
        i = -1
        while len(word_fr) > 1: 
            couple = word_fr.pop(), word_fr.pop() # Get the last elements from the list, remove them
            # The new node is a node with a unique integer as string key and with the frequency attribute equal to the sum of the frequencies of its children
            new_node = Node(i, couple[0].frequency+couple[1].frequency, couple[1], couple[0]) 
            pointer = new_node # Pointer to the new root
            # The following is a way to reduce time complexity, since the list is already sorted 
            # and we just need to find the right insertion point to keep the sorted order
            bisect.insort(word_fr, new_node, key = lambda x: - x.frequency)
            i -= 1
        return Tree(pointer)
    
    # This is basically a wrapper around the path method of the Node class
    # It also implements a lookup dictionary at object variable level 
    # to save the paths and speed up the retrieval of the paths for subsequent calls to the method with the same query
    def path_finder(self, query:str) -> list[int, int]:
        log = self.lookup_dict.get(query, False)
        if not log:
            log = list()
            self.root.path(query, log)
            log.reverse()
            self.lookup_dict[query] = log
        return log
        
    


if __name__ =="__main__":    
    tree = Tree.huffman_builder([(1, 10), (2, 5), (10, 2), (100, 1)])
    print (tree.path_finder(1), tree.path_finder(2), tree.path_finder(10), tree.path_finder(100))
    
        
        
        
        
        
        
        
        
        
        
        
        
            
