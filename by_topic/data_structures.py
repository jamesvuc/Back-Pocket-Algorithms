# ===========Data Structures=======
class Stack:
	def __init__(self):
		self.stk=[]

	def push(self, x):
		self.stk.append(x)

	def peek(self):
		return self.stk[-1]

	def pop(self):
		if self.empty():
			raise IndexError('Stack is empty!')
		return self.stk.pop()

	def empty(self):
		return len(self.stk)==0

from collections import deque
class Queue:
	def __init__(self):
		self.q=deque()

	def push(self, x):
		self.q.appendleft(x)

	def peek_top(self):
		return self.q[-1]

	def pop(self):
		if self.empty():
			raise IndexError('Queue is empty')
		
		return self.q.pop()

	def empty(self):
		return len(self.q)==0

class LinkedListNode:
	def __init__(self, data):
		self.data=data
		self.next=None
		self.prev=None

class DoubleLinkedList:
	def __init__(self):
		self.head=None
		self.tail=None

	def insert_front(self, node):
		if self.empty():
			self.head, self.tail = node, node
			node.next, node.prev = None, None
		else:
			node.next, node.prev = self.head, None
			self.head.prev=node
			self.head=node

	def insert_back(self, node):
		if self.empty():
			self.head, self.tail= node, node
			node.next, node.prev = None, None
		else:
			node.prev, node.next = self.tail, None
			self.tail.next = node
			self.tail = node

	def remove_node(self, node):
		if self.empty():
			raise Exception('List is empty!')

		if node.next is None and node.prev is None:#removing a singleton
			self.head, self.tail = None, None

		elif node.next is None:#removing the tail
			node.prev.next=None
			self.tail=node.prev
			node.prev=None

		elif node.prev is None:#removing the head
			node.next.prev=None
			self.head=node.next
			node.next=None
		
		else:#removing an interior node
			node.next.prev=node.prev
			node.prev.next=node.next

	def empty(self):
		return self.head is None

	def print_list(self):
		if self.empty():
			print('Empty.')
		else:
			node = self.head
			while node.next is not None:
				print(node.data,'->', end='')
				node = node.next
			print(node.data)

class TrieNode:
	def __init__(self):
		self.children={}#keys are characters, values are nodes
		self.data=None
		self.isword = False #data is optional => include an isword flag

class Trie:
	def __init__(self):
		self.root=TrieNode()

	def insert(self, string, data=None):
		node=self.root
		idx_last_found=None
		
		for char_idx, char in enumerate(string):
			if char in node.children:
				node=node.children[char]
			else:
				idx_last_found=char_idx
				break

		if idx_last_found is not None:
			for char in string[idx_last_found:]:
				node.children[char]=TrieNode()
				node=node.children[char]

		node.data=data 
		node.isword = True

	def search(self, key):
		node=self.root
		for char in key:	
			if char in node.children:
				node=node.children[char]
			else:
				return None

		if not node.isword: print(key,'not in Trie!')
		return node.data if node.isword else None

class BSTNode:
	def __init__(self, key, value=None):
		self.key=key
		self.value=value
		self.left=None
		self.right=None
		self.parent=None

class BST:
	def __init__(self):
		self.root = None

	def insert(self, key, value=None):
		node=BSTNode(key, value=value)
		
		if self.root is None:
			self.root = node
		else:
			curr_node = self.root
			prev_node = self.root
			while curr_node is not None:
				prev_node = curr_node
				if node.key > curr_node.key:
					curr_node = curr_node.right
				else:
					curr_node = curr_node.left

			node.parent = prev_node
			if node.key > prev_node.key:
				prev_node.right = node
			else:
				prev_node.left = node

		return node

	def search(self, key):
		curr_node = self.root

		while curr_node is not None:
			if curr_node.key == key: 
				return node

			if key > curr_node.key:
				curr_node = curr_node.right
			else:
				curr_node = curr_node.left

		return None

	def _transplant(self, old_root, new_root):
			if old_root == self.root:
				self.root = new_root
			elif old_root == old_root.parent.left:
				old_root.parent.left = new_root
			else:
				old_root.parent.right = new_root

			if new_root is not None:
				new_root.parent = old_root.parent

	def remove(self, node):
		if node.left is None and node.right is None: #is leaf
			if node.parent is not None: #not the root
				if node == node.parent.left:
					node.parent.left = None
				else:
					node.parent.right = None
			else:
				self.root = None

		elif node.left is None and node.right is not None:
			self._transplant(node, node.right)
		elif node.right is None and node.left is not None:
			self._transplant(node, node.left)
		else:
			min_node=self._find_min(node.right)
			if min_node.parent != node:
				self._transplant(min_node, min_node.right)
				min_node.right = node.right
				min_node.right.parent = min_node

			self._transplant(node, min_node)
			min_node.left = node.left
			min_node.left.parent = min_node

		node.left, node.right, node.parent = None, None, None
		return node

	def _find_min(self, node):
		tmp = node
		while tmp.left is not None:
			tmp = tmp.left

		return tmp

	def succ(self, node):
		if node.right is not None:
			return self._find_min(node.right)
		else:
			tmp=node.parent
			while tmp is not None and node==tmp.right:
				node=tmp
				tmp=tmp.parent

			return tmp

	def _find_max(self, node):
		tmp = node
		while tmp.right is not None:
			tmp = tmp.right
		
		return tmp

	def pred(self, node):
		if node.left is not None:
			return self._find_max(copy(node.left))
		else:
			tmp=node.parent
			while tmp is not None and node==tmp.left:
				node=tmp
				tmp=tmp.parent
			return tmp

	def inorder(self, node, nodefunc=lambda x:print('(',x.key, x.value,')', end='')):
		#left, node, right
		if node.left is not None:
			self.inorder(node.left, nodefunc=nodefunc)

		nodefunc(node)

		if node.right is not None:
			self.inorder(node.right, nodefunc=nodefunc)

	def preorder(self, node, nodefunc=lambda x:print('(',x.key, x.value,')', end='')):
		# node, left right
		stack = [node]
		while stack:
			n = stack.pop()
			nodefunc(n)
			if n.right is not None:
				stack.append(n.right)
			if n.left is not None:
				stack.append(n.left)

	def postorder(self, node, nodefunc=lambda x:print('(',x.key, x.value,')', end='')):
		#left, right, node
		stack = []
		do = True
		while do: #do-while loop in python
			while node is not None: #descend to the left, storing right children
				if node.right is not None:
					stack.append(node.right)
				stack.append(node)
				node = node.left

			node = stack.pop()
			if node.right is not None and stack and stack[-1] == node.right: #get the right child
				stack.pop()
				stack.append(node)
				node = node.right
			else:
				nodefunc(node)
				node = None

			do = bool(stack)

	def empty(self):
		return self.root is None

import heapq

class MinHeap:
	"""
	Provides a binary min-heap.
	You can do key-value stores with heapq by:
		1. wrapping your key-value in a class with overridden __lt__(self, other) 
			method; OR
		2. store values in the heap as (key,value) tuples. Comparisons are done using
			the first element's comparator;
		3. store the values in a dict and the keys (i.e hanldes) in the heap.
	"""
	def __init__(self, L=[]):
		self.heap = L
		heapq.heapify(self.heap)
		self.size = len(L)

	def push(self, x):
		heapq.heappush(self.heap, x)
		self.size += 1
	
	def pop(self):
		self.size -= 1
		return heapq.heappop(self.heap)
	
	def replace(self, x):
		return heapq.heapreplace(self.heap, x)
		#no change to size
	
	def peek(self):
		return self.heap[0]

	def empty(self):
		return self.size == 0

class MaxHeap:
	def __init__(self, L=[]):
		self.heap = L
		#need to reverse sign on heap elements for max-heap using heapq
		self.heap = [-x for x in self.heap]
		heapq.heapify(self.heap)
		self.size = len(L)

	def push(self, x):
		heapq.heappush(self.heap, -x)
		self.size += 1
	
	def pop(self):
		self.size -= 1
		return -heapq.heappop(self.heap)
	
	def replace(self, x):
		return -heapq.heapreplace(self.heap, -x)
		#no change to size
	
	def peek(self):
		return -self.heap[0]

from math import floor
class CustomHeap:
	def __init__(self, arr):
		#max heap 
		self.H = arr
		for i in range(floor(0.5 * len(self.H)) - 1, -1, -1):
			self._heapify(i)

	def _heapify(self, idx):
		l = 2 * idx + 1
		r = l + 1
		maxidx = idx
		if l < len(self.H) and self.H[l] > self.H[maxidx]:
			maxidx = l
		if r < len(self.H) and self.H[r] > self.H[maxidx]:
			maxidx = r

		if maxidx != idx:
			self.H[idx], self.H[maxidx] = self.H[maxidx], self.H[idx]
			self._heapify(maxidx) #percolate downwards

	def _increase_key(self, idx, new_key):
		#assume new_key > self.H[idx]
		self.H[idx] = new_key
		parent = floor(0.5 * idx) - 1

		#while heap property is violated, percolate up
		while parent >= 0 and self.H[idx] > self.H[parent]: 
			self.H[idx], self.H[parent] = self.H[parent], self.H[idx]

			idx = parent
			parent = floor(0.5 * idx) - 1 

	def push(self, x):
		self.H.append(x)
		self._increase_key(len(self.H) - 1, x)

	def pop(self):
		self.H[0], self.H[-1] = self.H[-1], self.H[0]
		tmp = self.H.pop()
		self._heapify(0)
		return tmp

	def peek(self):
		return self.H[0]

	def empty(self):
		return len(self.H) == 0

if __name__ == '__main__':
	if False: #Double LL
		L = DoubleLinkedList()
		nodes = dict()
		nums = [5,4,6,3,7,8,2,9,1,0]
		for num in nums:
			nodes[num] = LinkedListNode((num,chr(ord('a') + num)))
			L.insert_front(nodes[num])

		L.print_list()
		for num in nums[3:8]:
			print('removing',num)
			L.remove_node(nodes[num])

		L.print_list()

	if False: #Trie
		words = ['hello', 'hell', 'hella', 'halla','good', 'god', 'bye', 'by', 'be']
		t = Trie()
		for word in words:
			t.insert(word, data=True)

		for word in words+['helo','he','go','b','hi','him','byo']:
			print(word,'in Trie?',t.search(word) != None)

	if False: #BST
		B=BST()

		import random
		random.seed(0)
		keys = [random.randint(0,50) for _ in range(100)]
		nodes=[]

		nodes=[B.insert(k, value=chr(ord('a') + k)) for k in keys]

		inorder_trav = []
		func = lambda node : inorder_trav.append(node.key)
		B.inorder(B.root, nodefunc = func)
		print('inorder pass?',inorder_trav == sorted(keys))

		for n in nodes:
			s=B.succ(n)
			if s is not None:
				print('key=',n.key,'succ=', s.key)
			else:
				print(n.key, s)

		for n in nodes:
			p=B.pred(n)
			if p is not None:
				print('key=',n.key,'pred=', p.key)
			else:
				print(n.key, p)

		i=0
		while not B.empty():
			n=B.remove(nodes[::-1][i])
			print('removed', n.key)
			i+=1

	if False: #Heaps & heapsort
		#using heapq
		m = MinHeap()
		M = MaxHeap()

		nums=[5,4,6,3,7,3,8,2,9,1,0]
		for num in nums:
			m.push(num); M.push(num)
			print(m.peek(), M.peek())

		while not m.empty():
			print(m.pop(), M.pop())

		#using custom heap
		nums=[5,4,6,3,7,3,8,2,9,1,0]
		M = CustomHeap(nums)
		for x in [10,11,12,15,30]:
			M.push(x)

		print('peek=',M.peek())
		while not M.empty():
			print(M.pop())