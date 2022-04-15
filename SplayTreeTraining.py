import collections
import collections.abc


class Node:
    """Represents a node of a binary tree"""
    def __init__(self,key,value):
        self.left = None
        self.right = None
        self.parent = None
        self.key = key
        self.value = value


class BSTree:
    """
    BSTree implements an unbalanced Binary Search Tree.
    A Binary Search Tree is an ordered node based tree key structure
    in which each node has at most two children.
    For more information regarding BSTs, see:
    http://en.wikipedia.org/wiki/Binary_search_tree
    Constructors:
    BSTree() -> Creates a new empty Binary Search Tree
    BSTree(seq) -> Creates a new Binary Search Tree from the elements in sequence [(k1,v1),(k2,v2),...,(kn,vn)]
    """
    def __init__(self,*args):

        self.Root = None

        if len(args) == 1:
            if isinstance(args[0],collections.abc.Iterable):
                for x in args[0]:
                    self.insert(x[0],x[1])
            else:
                raise TypeError(str(args[0]) + " is not iterable")

    def inorder(self,*args):
        """
        T.inorder(...) -> Sequence. Produces a sequence of the Nodes
        in T, obtained in inorder.
        """
        if len(args) == 0:
            elements = []
            node = self.Root
        else:
            node = args[0]
            elements = args[1]

        if node.left:
            self.inorder(node.left,elements)

        elements.append(node)

        if node.right:
            self.inorder(node.right,elements)

        return elements

    def get_node(self,key,*args):
        """
        T.get_node(key,...) -> Node. Produces the Node in T  with key
        attribute key. If there is no such node, produces None.
        """
        if len(args) == 0:
            start = self.Root
        else:
            start = args[0]

        if not start:
            return None
        if key == start.key:
            return start
        elif key > start.key:
            return self.get_node(key,start.right)
        else:
            return self.get_node(key,start.left)

    def insert(self,key,value,*args):
        """
        T.insert(key,value...) <==> T[key] = value. Inserts
        a new Node with key attribute key and value attribute
        value into T.
        """
        if not isinstance(key,(int,int,float)):
            raise TypeError(str(key) + " is not a number")
        else:
            if not self.Root:
                self.Root = Node(key,value)
            elif len(args) == 0:
                if not self.get_node(key,self.Root):
                    self.insert(key,value,self.Root)
            else:
                child = Node(key,value)
                parent = args[0]
                if child.key > parent.key:
                    if not parent.right:
                        parent.right = child
                        child.parent = parent
                    else:
                        self.insert(key,value,parent.right)
                else:
                    if not parent.left:
                        parent.left = child
                        child.parent = parent
                    else:
                        self.insert(key,value,parent.left)

    def insert_from(self,seq):
        """
        T.insert_from(seq). For every key, value pair in seq,
        inserts a new Node into T with key and value attributes
        as given.
        """
        if isinstance(seq,collections.Iterable):
            for x in seq:
                self.insert(x[0],x[1])
        else:
            raise TypeError(str(iter) + " is not iterable")

    def get_max(self,*args):
        """
        T.get_max(...) -> Node. Produces the Node that has the maximum
        key attribute in T.
        """
        if len(args) == 0:
            node = self.Root
        else:
            node = args[0]

        if not node.right:
            return node
        else:
            return self.get_max(node.right)

    def get_min(self,*args):
        """
        T.get_min(...) -> Node. Produces the Node that has the minimum
        key attribute in T.
        """
        if len(args) == 0:
            node = self.Root
        else:
            node = args[0]

        if not node.left:
            return node
        else:
            return self.get_min(node.left)

    def get_element_count(self,*args):
        """
        T.get_element_count(...) -> Nat. Produces the number of elements
        in T.
        """
        if len(args) == 0:
            node = self.Root
        else:
            node = args[0]

        left = 0
        right = 0

        if node:
            if node.left:
                left = self.get_element_count(node.left)
            if node.right:
                right = self.get_element_count(node.right)

            return 1 + left + right
        else:
            return 0

    def get_height(self,*args):
        """
        T.get_height(...) -> Nat. Produces the height of T, defined
        as one added to the height of the tallest subtree.
        """
        if len(args) == 0:
            node = self.Root
        else:
            node = args[0]

        if not node or (not node.left and not node.right):
            return 0
        else:
            return 1 + max(self.get_height(node.left), self.get_height(node.right))

    def _delete_leaf(self,node):
        """
        T._delete_leaf(node). Deletes node from T, treating it as a leaf.
        """
        par_node = node.parent

        if par_node:
            if par_node.left == node:
                par_node.left = None
            else:
                par_node.right = None

            del node

    def _delete_leaf_parent(self,node):
        """
        T._delete_leaf_parent(node). Deletes node from T, treating it
        as a node with only one child.
        """
        par_node = node.parent

        if node.key == self.Root.key:
            if node.right:
                self.Root = node.right
                node.right = None
            else:
                self.Root = node.left
                node.left = None

        else:
            if par_node.right == node:
                if node.right:
                    par_node.right = node.right
                    par_node.right.parent = par_node
                    node.right = None
                else:
                    par_node.right = node.left
                    par_node.right.parent = par_node
                    node.left = None
            else:

                if node.right:
                    par_node.left = node.right
                    par_node.left.parent = par_node
                    node.right = None
                else:
                    par_node.left = node.left
                    par_node.left.parent = par_node
                    node.left = None

        del node

    def _switch_nodes(self,node1,node2):
        """
        T._switch_nodes(node1,node2). Switches positions
        of node1 and node2 in T.
        """
        switch1 = node1
        switch2 = node2
        temp_key = switch1.key
        temp_value = switch1.value

        if switch1.key == self.Root.key:
            self.Root.key = node2.key
            self.Root.value = node2.value
            switch2.key = temp_key
            switch2.value = temp_value

        elif switch2.key == self.Root.key:
            switch1.key = self.Root.key
            self.Root.key = temp_key
            self.Root.value = temp_value
        else:
            switch1.key = node2.key
            switch1.value = node2.value
            switch2.key = temp_key
            switch2.value = temp_value

    def _delete_node(self,node):
        """
        T._delete_node(node). Deletes node from T, treating it as
        a node with two children.
        """
        if self.get_height(node.left) > self.get_height(node.right):
            to_switch = self.get_max(node.left)
            self._switch_nodes(node,to_switch)

            if not (to_switch.right or to_switch.left):
                to_delete = self.get_max(node.left)
                self._delete_leaf(to_delete)
            else:
                to_delete = self.get_max(node.left)
                self._delete_leaf_parent(to_delete)
        else:
            to_switch = self.get_min(node.right)
            self._switch_nodes(node,to_switch)

            if not (to_switch.right or to_switch.left):
                to_delete = self.get_min(node.right)
                self._delete_leaf(to_delete)
            else:
                to_delete = self.get_min(node.right)
                self._delete_leaf_parent(to_delete)

    def delete(self,key):
        """T.delete(key) <==> del T[key]. Deletes the node
        with key attribute key from T.
        """
        node = self.get_node(key,self.Root)

        if node:
            if not (node.left or node.right):
                self._delete_leaf(node)

            elif not (node.left and node.right):
                self._delete_leaf_parent(node)

            else:
                self._delete_node(node)


class AVLNode(Node):
    """Represents a node of a balanced AVL Tree"""
    def __init__(self,key,value):
        """Initializes a BST node, then add height and balance attributes"""
        Node.__init__(self,key,value)
        self.height = 0
        self.balance = 0


class AVLTree(BSTree):
    """
    AVLTree implements a self-balancing AVL Tree.
    An AVL Tree is an ordered node based tree key structure
    in which each node has at most two children, and the heights
    of each children of a node differ by at most one.
    For more information regarding AVL Trees, see:
    http://en.wikipedia.org/wiki/Avl_tree
    Constructors:
    AVLTree() -> Creates a new empty AVL Tree
    AVLTree(seq) -> Creates a new AVL Tree from the elements in sequence [(k1,v1),(k2,v2),...,(kn,vn)]
    For further explanation of some functions or their source code, see bstree.py.
    """
    def __init__(self,*args):
        """Initializes tree the same as a BST"""
        BSTree.__init__(self,*args)

    def is_valid(self, *args):
        """
        T.is_valid(...) -> Boolean. Produces True if and only if
        T is a valid AVL Tree. Raises an exception otherwise.
        """
        if len(args) == 0:
            node = self.Root
        else:
            node = args[0]

        if not node:
            return True

        expected_height = self.get_height(node)
        expected_balance = self.get_balance(node)

        if not (node.height == expected_height):
            raise Exception("Height of node " + str(node.key) + " is " + str(node.height) + " and should be " + str(expected_height))

        if not (node.balance == expected_balance):
            raise Exception("Balance of node " + str(node.key) + " is " + str(node.balance) + " and should be " + str(expected_balance))

        if abs(expected_balance) > 1:
            raise Exception("Tree is unbalanced at node " + str(node.key))

        if node.left:
            if not node.left.parent == node:
                raise Exception("Left child of node " + str(node.key) + " is adopted by another node!")

        if node.right:
            if not node.right.parent == node:
                raise Exception("Right child of node " + str(node.key) + " is adopted by another node!")

        if node.parent and node.parent.left == node:
            if node.key > node.parent.key:
                raise Exception("Node " + str(node.key) + " is to the left of " + str(node.parent.key) + " but is larger")

        if node.parent and node.parent.right == node:
            if node.key < node.parent.key:
                raise Exception("Node " + str(node.key) + " is to the right of " + str(node.parent.key) + " but is smaller")

        return (self.is_valid(node.left) and self.is_valid(node.right))

    def insert(self,key,value,*args):
        """
        T.insert(key,value...) <==> T[key] = value. Inserts
        a new Node with key attribute key and value attribute
        value into T. Balances if necessary.
        """
        if not isinstance(key,(int,int,float)):
            raise TypeError(str(key) + " is not a number")
        else:
            if not self.Root:
                self.Root = AVLNode(key,value)
            elif len(args) == 0:
                if not self.get_node(key,self.Root):
                    self.insert(key,value,self.Root)
            else:
                child = AVLNode(key,value)
                parent = args[0]
                if child.key > parent.key:
                    if not parent.right:
                        parent.right = child
                        child.parent = parent
                        self._update_height(parent)
                        self._update_balance(parent)
                        node = child
                        while node and abs(node.balance) <=1:
                            node = node.parent
                        if node:
                            self._balance(node)
                    else:
                        self.insert(key,value,parent.right)
                else:
                    if not parent.left:
                        parent.left = child
                        child.parent = parent
                        self._update_height(parent)
                        self._update_balance(parent)
                        node = child
                        while abs(node.balance) <=1:
                            node = node.parent
                            if not node:
                                break
                        if node:
                            self._balance(node)
                    else:
                        self.insert(key,value,parent.left)

    def get_balance(self,*args):
        """
        T.get_balance(...) -> Nat. Produces the balance of T, defined
        as the height of the right subtree taken away from the height
        of the left subtree.
        """
        if len(args) == 0:
            node = self.Root
        else:
            node = args[0]

        return ((node.left.height if node.left else -1) -
                (node.right.height if node.right else -1))

    def _update_height(self,node):
        """
        T._update_height(node). Updates the height attribute
        of Nodes in T starting from node backtracking up to the root.
        """
        if not node:
            pass
        else:
            new_height = self.get_height(node)
            if node.height == new_height:
                pass
            else:
                node.height = new_height
                self._update_height(node.parent)

    def _update_balance(self,node):
        """
        T._update_balance(node). Updates the balance attribute
        of Nodes in T starting from node backtracking up to the root.
        """
        if not node:
            pass
        else:
            new_balance = self.get_balance(node)
            if node.balance == new_balance:
                pass
            else:
                node.balance = new_balance
                self._update_balance(node.parent)

    def _rotate_left(self,pivot):
        """
        T._rotate_left(pivot). Performs a left tree rotation in T
        around the Node pivot.
        """
        old_root = pivot
        par_node = old_root.parent

        new_root = old_root.right
        temp = new_root.right
        old_root.right = new_root.left

        if old_root.right:
            old_root.right.parent = old_root
        new_root.left = old_root
        old_root.parent = new_root

        if par_node is None:
            self.Root = new_root
            self.Root.parent = None
        else:
            if par_node.right and par_node.right.key == old_root.key:
                par_node.right = new_root
                new_root.parent = par_node
            elif par_node.left and par_node.left.key == old_root.key:
                par_node.left = new_root
                new_root.parent = par_node

        self._update_height(new_root.left)
        self._update_height(par_node)
        self._update_balance(new_root.left)
        self._update_balance(par_node)

    def _rotate_right(self,pivot):
        """
        T._rotate_right(pivot). Performs a right tree rotation in T
        around the Node pivot.
        """
        old_root = pivot
        par_node = old_root.parent

        new_root = old_root.left
        temp = new_root.left
        old_root.left = new_root.right

        if (old_root.left):
            old_root.left.parent = old_root

        new_root.right = old_root
        old_root.parent = new_root

        if par_node is None:
            self.Root = new_root
            self.Root.parent = None
        else:
            if par_node.right and par_node.right.key == old_root.key:
                par_node.right = new_root
                new_root.parent = par_node
            elif par_node.left and par_node.left.key == old_root.key:
                par_node.left = new_root
                new_root.parent = par_node

        self._update_height(new_root.right)
        self._update_height(par_node)
        self._update_balance(new_root.right)
        self._update_balance(par_node)

    def _balance(self,pivot):
        """
        T._balance(pivot). Balances T at Node pivot, performing
        appropriate tree rotations to ensure T remains a valid AVL Tree.
        """
        weight = self.get_balance(pivot)

        if weight == -2:
            if self.get_balance(pivot.right) == -1 or self.get_balance(pivot.right) == 0:
                self._rotate_left(pivot)

            elif self.get_balance(pivot.right) == 1:
                self._rotate_right(pivot.right)
                self._rotate_left(pivot)

        elif weight == 2:
            if self.get_balance(pivot.left) == 1 or self.get_balance(pivot.left) == 0:
                self._rotate_right(pivot)

            elif self.get_balance(pivot.left) == -1:
                self._rotate_left(pivot.left)
                self._rotate_right(pivot)

    def _delete_leaf(self,node):
        """
        T._delete_leaf_parent(node). Deletes node from T, treating it
        as a Node with only one child.
        """
        par_node = node.parent

        if par_node:
            if par_node.left == node:
                par_node.left = None
            else:
                par_node.right = None

            del node

            self._update_height(par_node)
            self._update_balance(par_node)
            to_balance = par_node

            while to_balance and abs(to_balance.balance) <=1:
                to_balance = to_balance.parent
            if to_balance:
                self._balance(to_balance)

        else:
            self.Root = None

    def _delete_leaf_parent(self,node):
        """
        T._delete_leaf_parent(node). Deletes node from T, treating it
        as a Node with only one child.
        """
        par_node = node.parent

        if node.key == self.Root.key:
            if node.right:
                self.Root = node.right
                node.right = None
            else:
                self.Root = node.left
                node.left = None

        else:
            if par_node.right == node:
                if node.right:
                    par_node.right = node.right
                    par_node.right.parent = par_node
                    node.right = None
                else:
                    par_node.right = node.left
                    par_node.right.parent = par_node
                    node.left = None
            else:

                if node.right:
                    par_node.left = node.right
                    par_node.left.parent = par_node
                    node.right = None
                else:
                    par_node.left = node.left
                    par_node.left.parent = par_node
                    node.left = None

        del node

        self._update_height(par_node)
        self._update_balance(par_node)
        to_balance = par_node

        while to_balance and abs(to_balance.balance) <=1:
            to_balance = to_balance.parent
        if to_balance:
            self._balance(to_balance)

    def _switch_nodes(self,node1,node2):
        """
        T._switch_nodes(node1,node2). Switches positions
        of node1 and node2 in T.
        """
        BSTree._switch_nodes(self,node1,node2)

    def _delete_node(self,node):
        """
        T._delete_node(node). Deletes node from T, treating it as
        a Node with two children.
        """
        if self.get_height(node.left) > self.get_height(node.right):
            to_switch = self.get_max(node.left)
            self._switch_nodes(node,to_switch)

            if not (to_switch.right or to_switch.left):
                to_delete = self.get_max(node.left)
                self._delete_leaf(to_delete)
            else:
                to_delete = self.get_max(node.left)
                self._delete_leaf_parent(to_delete)
        else:
            to_switch = self.get_min(node.right)
            self._switch_nodes(node,to_switch)

            if not (to_switch.right or to_switch.left):
                to_delete = self.get_min(node.right)
                self._delete_leaf(to_delete)
            else:
                to_delete = self.get_min(node.right)
                self._delete_leaf_parent(to_delete)

    def delete(self,key):
        """T.delete(key) <==> del T[key]. Deletes the Node
        with key attribute key from T.
        """
        node = self.get_node(key,self.Root)

        if node:
            if not (node.left or node.right):
                self._delete_leaf(node)

            elif not (node.left and node.right):
                self._delete_leaf_parent(node)

            else:
                self._delete_node(node)

    def delete_from(self,seq):
        """
        T.delete_from(seq). For every keyin seq, deletes
        the Node with that key attribute from T.
        """
        if isinstance(seq,collections.Iterable):
            for x in seq:
                self.delete(x)
        else:
            raise TypeError(str(iter) + " is not iterable")


T = AVLTree()
s = 0
el_count = 0
for i in range(int(input())):
    a, b, *c = input().split()
    b = (int(b) + s) % 1000000001
    if a == "+" and not T.get_node(b):
        T.insert(b, 0)
        el_count += 1
    if a == '-' and T.get_node(b):
        T.delete(b)
        el_count -= 1
    if a == '?':
        print('Found') if T.get_node(b) else print('Not found')
    if a == 's':
        if el_count != 0:
            s_new = 0
            c = (int(c[0]) + s) % 1000000001
            for elem in T.inorder():
                if elem.key in range(b, c + 1):
                    s_new += elem.key
            s = s_new
        else:
            s = 0
        print(s)
        