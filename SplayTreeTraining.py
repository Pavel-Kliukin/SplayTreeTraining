class Node:
    def __init__(self, key):
        self.key = key
        self.parent = None
        self.leftChild = None
        self.rightChild = None
        self.height = 0

    def __str__(self):
        return str(self.key) + "(" + str(self.height) + ")"

    def is_leaf(self):
        return self.height == 0

    def max_children_height(self):
        if self.leftChild and self.rightChild:
            return max(self.leftChild.height, self.rightChild.height)
        elif self.leftChild and not self.rightChild:
            return self.leftChild.height
        elif not self.leftChild and self.rightChild:
            return self.rightChild.height
        else:
            return -1

    def balance(self):
        return (self.leftChild.height if self.leftChild else -1) - (self.rightChild.height if self.rightChild else -1)


class AVLTree:
    def __init__(self, *args):
        self.rootNode = None
        self.elements_count = 0
        self.rebalance_count = 0
        if len(args) == 1:
            for i in args[0]:
                self.insert(i)

    def height(self):
        if self.rootNode:
            return self.rootNode.height
        else:
            return 0

    def rebalance(self, node_to_rebalance):
        self.rebalance_count += 1
        A = node_to_rebalance
        F = A.parent  # allowed to be NULL
        if node_to_rebalance.balance() == -2:
            if node_to_rebalance.rightChild.balance() <= 0:
                """ Rebalance, case RRC """
                B = A.rightChild
                C = B.rightChild
                assert (A is not None and B is not None and C is not None)
                A.rightChild = B.leftChild
                if A.rightChild:
                    A.rightChild.parent = A
                B.leftChild = A
                A.parent = B
                if F is None:
                    self.rootNode = B
                    self.rootNode.parent = None
                else:
                    if F.rightChild == A:
                        F.rightChild = B
                    else:
                        F.leftChild = B
                    B.parent = F
                self.recompute_heights (A)
                self.recompute_heights (B.parent)
            else:
                """ Rebalance, case RLC """
                B = A.rightChild
                C = B.leftChild
                assert (A is not None and B is not None and C is not None)
                B.leftChild = C.rightChild
                if B.leftChild:
                    B.leftChild.parent = B
                A.rightChild = C.leftChild
                if A.rightChild:
                    A.rightChild.parent = A
                C.rightChild = B
                B.parent = C
                C.leftChild = A
                A.parent = C
                if F is None:
                    self.rootNode = C
                    self.rootNode.parent = None
                else:
                    if F.rightChild == A:
                        F.rightChild = C
                    else:
                        F.leftChild = C
                    C.parent = F
                self.recompute_heights (A)
                self.recompute_heights (B)
        else:
            assert(node_to_rebalance.balance() == +2)
            if node_to_rebalance.leftChild.balance() >= 0:
                B = A.leftChild
                C = B.leftChild
                """ Rebalance, case LLC """
                assert (A is not None and B is not None and C is not None)
                A.leftChild = B.rightChild
                if A.leftChild:
                    A.leftChild.parent = A
                B.rightChild = A
                A.parent = B
                if F is None:
                    self.rootNode = B
                    self.rootNode.parent = None
                else:
                    if F.rightChild == A:
                        F.rightChild = B
                    else:
                        F.leftChild = B
                    B.parent = F
                self.recompute_heights(A)
                self.recompute_heights(B.parent)
            else:
                B = A.leftChild
                C = B.rightChild
                """ Rebalance, case LRC """
                assert (A is not None and B is not None and C is not None)
                A.leftChild = C.rightChild
                if A.leftChild:
                    A.leftChild.parent = A
                B.rightChild = C.leftChild
                if B.rightChild:
                    B.rightChild.parent = B
                C.leftChild = B
                B.parent = C
                C.rightChild = A
                A.parent = C
                if F is None:
                    self.rootNode = C
                    self.rootNode.parent = None
                else:
                    if F.rightChild == A:
                        F.rightChild = C
                    else:
                        F.leftChild = C
                    C.parent = F
                self.recompute_heights (A)
                self.recompute_heights (B)

    def recompute_heights(self, start_from_node):
        changed = True
        node = start_from_node
        while node and changed:
            old_height = node.height
            node.height = (node.max_children_height() + 1 if (node.rightChild or node.leftChild) else 0)
            changed = node.height != old_height
            node = node.parent

    def add_as_child(self, parent_node, child_node):
        node_to_rebalance = None
        if child_node.key < parent_node.key:
            if not parent_node.leftChild:
                parent_node.leftChild = child_node
                child_node.parent = parent_node
                if parent_node.height == 0:
                    node = parent_node
                    while node:
                        node.height = node.max_children_height() + 1
                        if not node.balance() in [-1, 0, 1]:
                            node_to_rebalance = node
                            break  # we need the one that is furthest from the root
                        node = node.parent
            else:
                self.add_as_child(parent_node.leftChild, child_node)
        else:
            if not parent_node.rightChild:
                parent_node.rightChild = child_node
                child_node.parent = parent_node
                if parent_node.height == 0:
                    node = parent_node
                    while node:
                        node.height = node.max_children_height() + 1
                        if not node.balance() in [-1, 0, 1]:
                            node_to_rebalance = node
                            break  # we need the one that is furthest from the root
                        node = node.parent
            else:
                self.add_as_child(parent_node.rightChild, child_node)

        if node_to_rebalance:
            self.rebalance(node_to_rebalance)

    def insert(self, key):
        new_node = Node(key)
        if self.rootNode is None:
            self.rootNode = new_node
            self.elements_count += 1                       # --------
        else:
            if self.find(key) is None:
                self.elements_count += 1
                self.add_as_child(self.rootNode, new_node)

    def find_biggest(self, start_node):
        node = start_node
        while node.rightChild:
            node = node.rightChild
        return node

    def find_smallest(self, start_node):
        node = start_node
        while node.leftChild:
            node = node.leftChild
        return node

    def inorder_non_recursive(self):
        node = self.rootNode
        retlst = []
        if not node:                    # --------
            return retlst               # --------
        while node.leftChild:
            node = node.leftChild
        while node:
            retlst += [node.key]
            if node.rightChild:
                node = node.rightChild
                while node.leftChild:
                    node = node.leftChild
            else:
                while node.parent and node == node.parent.rightChild:
                    node = node.parent
                node = node.parent
        return retlst

    def preorder(self, node, retlst=None):
        if retlst is None:
            retlst = []
        retlst += [node.key]
        if node.leftChild:
            retlst = self.preorder(node.leftChild, retlst)
        if node.rightChild:
            retlst = self.preorder(node.rightChild, retlst)
        return retlst

    def inorder(self, node, retlst=None):
        if retlst is None:
            retlst = []
        if node.leftChild:
            retlst = self.inorder(node.leftChild, retlst)
        retlst += [node.key]
        if node.rightChild:
            retlst = self.inorder(node.rightChild, retlst)
        return retlst

    def postorder(self, node, retlst=None):
        if retlst is None:
            retlst = []
        if node.leftChild:
            retlst = self.postorder(node.leftChild, retlst)
        if node.rightChild:
            retlst = self.postorder(node.rightChild, retlst)
        retlst += [node.key]
        return retlst

    def as_list(self, pre_in_post):
        if not self.rootNode:
            return []
        if pre_in_post == 0:
            return self.preorder(self.rootNode)
        elif pre_in_post == 1:
            return self.inorder(self.rootNode)
        elif pre_in_post == 2:
            return self.postorder(self.rootNode)
        elif pre_in_post == 3:
            return self.inorder_non_recursive()

    def find(self, key):
        return self.find_in_subtree(self.rootNode, key)

    def find_in_subtree(self, node, key):
        if node is None:
            return None  # key not found
        if key < node.key:
            return self.find_in_subtree(node.leftChild, key)
        elif key > node.key:
            return self.find_in_subtree(node.rightChild, key)
        else:  # key is equal to node key
            return node

    def remove(self, key):
        # first find
        node = self.find(key)

        if node is not None:
            self.elements_count -= 1

            #     There are three cases:
            #
            #     1) The node is a leaf.  Remove it and return.
            #
            #     2) The node is a branch (has only 1 child). Make the pointer to this node
            #        point to the child of this node.
            #
            #     3) The node has two children. Swap items with the successor
            #        of the node (the smallest item in its right subtree) and
            #        delete the successor from the right subtree of the node.
            if node.is_leaf():
                self.remove_leaf(node)
            elif (bool(node.leftChild)) ^ (bool(node.rightChild)):
                self.remove_branch(node)
            else:
                assert node.leftChild and node.rightChild
                self.swap_with_successor_and_remove(node)

    def remove_leaf(self, node):
        parent = node.parent
        if parent:
            if parent.leftChild == node:
                parent.leftChild = None
            else:
                assert parent.rightChild == node
                parent.rightChild = None
            self.recompute_heights(parent)
        else:
            self.rootNode = None
        del node
        # rebalance
        node = parent
        while node:
            if not node.balance() in [-1, 0, 1]:
                self.rebalance(node)
            node = node.parent

    def remove_branch(self, node):
        parent = node.parent
        if parent:
            if parent.leftChild == node:
                parent.leftChild = node.rightChild or node.leftChild
            else:
                assert (parent.rightChild == node)
                parent.rightChild = node.rightChild or node.leftChild
            if node.leftChild:
                node.leftChild.parent = parent
            else:
                assert node.rightChild
                node.rightChild.parent = parent
            self.recompute_heights(parent)
        else:                                                  # ---------
            if node.leftChild:
                node.leftChild.parent = None
                node.leftChild.height = 0
            else:
                assert node.rightChild
                node.rightChild.parent = None
                node.rightChild.height = 0
            self.rootNode = node.rightChild or node.leftChild
            self.elements_count = 1                            # ---------
        del node
        # rebalance
        node = parent
        while node:
            if not node.balance() in [-1, 0, 1]:
                self.rebalance(node)
            node = node.parent

    def swap_with_successor_and_remove(self, node):
        successor = self.find_smallest(node.rightChild)
        self.swap_nodes(node, successor)
        assert (node.leftChild is None)
        if node.height == 0:
            self.remove_leaf(node)
        else:
            self.remove_branch(node)

    def swap_nodes(self, node1, node2):
        assert (node1.height > node2.height)
        parent1 = node1.parent
        leftChild1 = node1.leftChild
        rightChild1 = node1.rightChild
        parent2 = node2.parent
        assert (parent2 is not None)
        assert (parent2.leftChild == node2 or parent2 == node1)
        leftChild2 = node2.leftChild
        assert (leftChild2 is None)
        rightChild2 = node2.rightChild

        # swap heights
        tmp = node1.height
        node1.height = node2.height
        node2.height = tmp

        if parent1:
            if parent1.leftChild == node1:
                parent1.leftChild = node2
            else:
                assert (parent1.rightChild == node1)
                parent1.rightChild = node2
            node2.parent = parent1
        else:
            self.rootNode = node2
            node2.parent = None

        node2.leftChild = leftChild1
        leftChild1.parent = node2
        node1.leftChild = leftChild2  # None
        node1.rightChild = rightChild2
        if rightChild2:
            rightChild2.parent = node1
        if not (parent2 == node1):
            node2.rightChild = rightChild1
            rightChild1.parent = node2

            parent2.leftChild = node1
            node1.parent = parent2
        else:
            node2.rightChild = node1
            node1.parent = node2


T = AVLTree()
s = 0
for i in range(int(input())):
    a, b, *c = input().split()
    b = (int(b) + s) % 1000000001
    if a == "+":
        T.insert(b)
    if a == '-':
        T.remove(b)
    if a == '?':
        print('Found') if T.find(b) is not None else print('Not found')
    if a == 's':
        c = (int(c[0]) + s) % 1000000001
        if T.elements_count != 0:
            s_new = 0
            for elem in T.inorder_non_recursive():
                if b <= elem <= c:
                    s_new += elem
            s = s_new
        else:
            s = 0
        print(s)
