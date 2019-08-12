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

from collections import deque, defaultdict
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

# ========Sorting & Selecting=======

def heapsort(arr):
	newarr = []
	heapq.heapify(arr)
	while arr:
		newarr.append(heapq.heappop(arr))
	arr[:] = newarr[:]
	return arr

def heapselect(arr, k):
	"""
	Does order statistic selection in O(n log k) using O(k) extra storage
	"""
	H = [] #will be a max-heap of size k
	for x in arr:
		if len(H) < k:
			heapq.heappush(H, -x) #use negation for max-heap keys
		else:
			if x < - H[0]: #use negation for max-heap keys
				heapq.heapreplace(H, -x)

	return -H[0]

def merge(arr1, arr2):
	new_arr = []
	i,j = 0,0
	while i < len(arr1) and j < len(arr2):
		if arr1[i] < arr2[j]:
			new_arr.append(arr1[i])
			i += 1
		else:
			new_arr.append(arr2[j])
			j += 1

	while i < len(arr1):
		new_arr.append(arr1[i])
		i+=1
	
	while j < len(arr2):
		new_arr.append(arr2[j])
		j+=1

	return new_arr

def _mergesort(arr):
	"""
	Implements a not in-place mergesort.
	Runs in O(nlgn)
	"""
	if len(arr) == 1:
		return arr
	
	mid = int(len(arr)/2)
	
	left = _mergesort(arr[:mid])
	right = _mergesort(arr[mid:])

	return merge(left, right)

def mergesort(arr):
	if len(arr) > 1:
		arr[:] = _mergesort(arr)

def partition(arr, lo, hi):
	pivot = arr[(lo + hi)//2]
	i, j = lo, hi
	while True:
		while i < j and arr[i] < pivot:
			i += 1
		while i < j and arr[j] > pivot:
			j -= 1

		if i >= j: return j
		arr[i], arr[j] = arr[j], arr[i]

def _quicksort(arr, lo, hi):
	if lo < hi:
		p = partition(arr, lo, hi)
		_quicksort(arr, lo, p-1)
		_quicksort(arr, p+1, hi)

def quicksort(arr):
	#in-place
	#add randomization?
	_quicksort(arr, 0, len(arr)-1)

def _quickselect(arr, lo, hi, k):
	if lo == hi:
		return arr[lo]

	p = partition(arr, lo, hi)

	if k == p:
		return arr[p]
	elif k < p:
		return _quickselect(arr, lo, p-1, k)
	else:
		return _quickselect(arr, p+1, hi, k)

def quickselect(arr, k):
	return _quickselect(arr, 0, len(arr)-1, k)

def bisection_search(arr, x, lo=0, hi=None):
	"""
	Assmues arr is sorted in *ascending* order.
	Returns
		if x in arr, 
			x
		else:
			the index where x should go if we insert by 
			moving everything after idx to the right
	
	This is the same as bisect.bisect_left(arr, x)
	"""
	if not hi:
		hi = len(arr)
	while lo < hi:
		mid = (lo + hi) // 2 #// is integer division
		if x < arr[mid]: hi = mid
		else: lo = mid + 1

	return lo

def insort(arr, x):
	"""
	Inserts x into the pre-sorted array arr.
	"""
	idx = bisection_search(arr, x)
	arr.append(arr[-1])
	for i in range(len(arr)-1, idx, -1):
		arr[i] = arr[i-1]

	arr[idx] = x

def get_ith_digit(x, i): #lsd is at i=0
    return (x//(10**i)%10)

def counting_sort(arr, digit):
	"""
	counting sort for integers in base 10
	"""
	counts = [0] * 10
	for num in arr:
		counts[get_ith_digit(num, digit)] += 1
    
	for i in range(1, 10):
		counts[i] += counts[i-1]
		
	res = [0]*len(arr)
	""" need to build the result in reverse order for stable a sort
		because elements get placed further in res first due to 
		to counts[d] decreasing, not increasing."""
	for num in reversed(arr):
		d = get_ith_digit(num, digit)
		res[counts[d]-1] = num
		counts[d] -= 1

	arr[:] = res
    
def radix_sort(arr, max_num_digs=10):
	"""
	radix-sort for integers <= 10*max_num_digs - 1 in O(n * max_num_digs)
		When max_num_digs = 10, this includes all 32-bit 
		signed & unsigned integers.
	"""
	for i in range(max_num_digs):
		counting_sort(arr, i)

def insertion_sort(arr):
	for i in range(1, len(arr)):
		for j in range(i, 0, -1):# takes the ith element and inserts 
								#  it in sorted order for arr[:i+1]
			if arr[j] < arr[j-1]: 
				arr[j], arr[j-1] = arr[j-1], arr[j]

def bubble_sort(arr):
	i = len(arr)
	while True:
		swapped = False
		for j in range(1, i):
			if arr[j-1] > arr[j]:
				arr[j-1], arr[j] = arr[j], arr[j-1]
				swapped = True
		#largest element is now at i-1
		if not swapped: break
		i -= 1

#=======Graphs & Sets==========
from copy import copy, deepcopy

class Graph:
	def __init__(self, vertices, edges, weights = None, is_directed=False):
		"""
		vertices is an iterable of hashable vertices
		edges is an iterable of pairs of hashable vertices [u,v]
		weights (if not None) is an iterable of tuples (u,v,w)
		"""
		self._graph = defaultdict(list)
		for u in vertices:
			self._graph[u] #add all the vertices to the graph:

		for u,v in edges:
			self._graph[u].append(v)

		#initialize a 2-d array of weights which default to 1
		self._weights = defaultdict(lambda : defaultdict(lambda :1))
		if weights is not None:
			for u,v,w in weights:
				self._weights[u][v] = w

	""" Traversals"""
	def BFS(self, src):
		#initialize the attribute maps
		colours = defaultdict(lambda :'w')
		parents = defaultdict(lambda : None)
		dists = defaultdict(lambda : float('inf'))

		dists[src] = 0 	#the distance to the source is zero
		Q = deque() #initialize queue
		Q.appendleft(src)
		while Q:
			u = Q.pop()					#dequeue a node to visit
			colours[u] = 'g'
			for v in self._graph[u]:
				if colours[v] == 'w': 	#if the colour is white (unseen)
					dists[v] = dists[u] +1 #dist to v is dist to u + 1
					parents[v] = u 		#v was discovered via u => u is v's parent
					Q.appendleft(v) 			#push v onto the queue for processing

			colours[u] = 'b' 			#set the colour of u, all of whose children are either
										# processed or queued, to black 
		return dists, parents

	def DFS(self):
		#set up variables
		not_seen = set(self._graph.keys())
		colours = defaultdict(lambda :'w')
		parents = defaultdict(lambda: None)
		t_disc, t_fin = dict(), dict()
		t = 0
		
		self.has_cycle = False
		self.topo_sort_res = deque()

		def _DF_visit(self, u): #do pseudo-inheritance
			nonlocal t
			t += 1 						#increment the global 'time'
			t_disc[u] = t 				#record the time u was discovered
			colours[u] = 'g' 			#make u grey
			not_seen.remove(u)
			for v in self._graph[u]:			
				if colours[v] == 'w': 	#first time discovering v
					parents[v] = u 		#set the parent (first discoverer) of v to be u
					_DF_visit(self, v) 	#visit v 
				elif colours[v] == 'g': #if you've already visited v, there's a cycle
					print(u,v)
					self.has_cycle = True

			colours[u] = 'b'
			t += 1
			t_fin[u] = t

			# add the nodes to the topological sort list in the reverse order
			# they are compelted.
			self.topo_sort_res.appendleft(u)

		#do the main search
		while not_seen:
			node = not_seen.pop(); not_seen.add(node)
			if colours[node] == 'w': #necessary?
				_DF_visit(self, node)

		return parents, t_disc, t_fin

	""" Other Algs """
	def min_span_tree_PRIM(self):
		costs = defaultdict(lambda: float('inf'))
		parents = defaultdict(lambda: None)
		not_seen = set(self._graph.keys())
		
		r = not_seen.pop(); not_seen.add(r)
		costs[r] = 0
		while not_seen:
			
			#extract_min from not_seen (can't use heapq bc we decrease keys)
			min_u = not_seen.pop(); not_seen.add(min_u)
			for u in not_seen:
				if costs[u] < costs[min_u]:
					min_u = u
			not_seen.remove(min_u)

			#for all the edges leaving from u, greedily update their parents
			for v in self._graph[min_u]:
				if v in not_seen and self._weights[min_u][v] < costs[v]:
					parents[v] = min_u
					costs[v] = self._weights[min_u][v]

		return parents

	def min_span_tree_KRUK(self):
		MST_edges = []
		dsu = DisjointSetUnion(self._graph.keys())

		srt_edges = []
		for u in self._graph:
			for v in self._graph[u]:
				srt_edges.append((u,v,self._weights[u][v]))

		srt_edges.sort(key = lambda x:x[2])

		for u,v,w in srt_edges:
			if dsu.find(u) != dsu.find(v):
				MST_edges.append((u,v))
				dsu.union(u,v)

		return MST_edges

	def topo_sort(self):
		if not hasattr(self, 'has_cycle'):
			#the DFS has not been run
			self.DFS()

		if self.has_cycle:
			#ordering is only defined for acyclic graphs
			print('Graph has a cycle!')
			return None

		return list(self.topo_sort_res)

	def detect_cycle(self): ###UNTESTED###
		not_visited = set(self._graph.keys())
		colours = {node : 'w' for node in self._graph}

		def _helper(u):
			if colours[u] != 'w':
				return colours[u] == 'g'

			colours[u] = 'g'
			not_seen.remove(u)
			for v in self._graph[u]:
				if colours[v] == 'g' or _helper(v):
					return True

			colours[u] = 'b'
			return False

		while not_visited:
			x = not_visited.pop(); not_visited.add(x)
			if _helper(x):
				return True

		return False

	def find_tour(self, src):
		""" 
		Use find an Eulerian tour through the nodes connected to src 
		using Hierholzer’s algorithm.
		"""
		graph_copy = deepcopy(self._graph) # Hierholzer’s algorithm destroys the graph
		colours = defaultdict(lambda :'w')
		tour = deque()
		
		def DF_visit(u):
			colours[u] = 'g'
			while graph_copy[u]:
				v = graph_copy[u].pop()
				if colours[v] == 'w':
					DF_visit(v)

			tour.appendleft(u)

		DF_visit(src)

		return list(tour)

	""" Relaxation-based Shortest Path """
	def _init_relaxation(self, src):
		dists, parents = defaultdict(lambda:float('inf')), defaultdict(lambda:None)
		dists[src] = 0
		return dists, parents

	def _relax(self, u, v, dists=None, parents=None):
		#if the path to src -> u -> v is less than cur path to v
		if dists[u] + self._weights[u][v] < dists[v]:
			dists[v] = dists[u] + self._weights[u][v]
			parents[v] = u

	def bellman_ford(self, src):
		dists, parents = self._init_relaxation(src) # initialize
		for i in range(len(self._graph)-1):
			for u in self._graph:					# iterate through vertices
				for v in self._graph[u]:			# iterate through edges
					self._relax(u, v, dists=dists, parents=parents) # relax
			
		#check for negative weight cycles which result in undefined min path lengths
		for u in self._graph:						# iterate through vertices
			for v in self._graph[u]:					# iterate through edges
				if dists[v] > dists[u] + self._weights[u][v]:
					print('Undefined path length!')

		return dists, parents

	def DAG_shortest_path(self, src):
		topo_srt_nodes = self.topo_sort()
		dists, parents = self._init_relaxation(src)
		for u in topo_srt_nodes:
			for v in self._graph[u]:
				self._relax(u, v, dists=dists, parents=parents)

		return dists, parents

	def dijkstra(self, src):
		dists, parents = self._init_relaxation(src)
		to_search = set(self._graph.keys())

		#while unvisted notes (bool(nonempty set) = True)
		while to_search:
			# find the min element from to_search
			min_u = to_search.pop(); to_search.add(min_u)#initialize an element
			min_d = dists[min_u]						#initialize a dist
			for u in to_search: 						#use a linear min-search for simplicity
				if dists[u] < min_d:
					min_u, min_d = u, dists[u]
			to_search.remove(min_u)
			
			#relax the edges departing from u
			for v in self._graph[min_u]:				
				self._relax(min_u, v, dists, parents) # relax the edge u->v

		return dists, parents

class DisjointSetUnion:
	def __init__(self, elems):
		self.reps = dict(zip(elems, elems))
		self.sizes = defaultdict(lambda : 1)
	
	def find(self, x):
		if self.reps[x] != x: #traverse the chain until you get to the parent
			self.reps[x] = self.find(self.reps[x]) #path compression
		return self.reps[x]
	
	def union(self, x, y): #union by size
		#should store sizes ideally to merge the smaller w/ larger
		rep_x = self.find(x)
		rep_y = self.find(y)
		if rep_x == rep_y: return # sets are the same

		if self.sizes[rep_x] < self.sizes[rep_y]:
			rep_x, rep_y = rep_y, rep_x #swap the reps

		#S(y) is now smaller than S(x)
		self.sizes[rep_x] += self.sizes[rep_y] #S(x) eats S(y)
		self.reps[rep_y] = rep_x # x now represents S(y)

# ========Testing=======

def main():
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
		#keys have lots of duplicates (on purpose)
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

		#heapsort
		import random
		random.seed(0)
		nums = [random.randint(0,2**31-1) for _ in range(1000)]
		srt_nums = sorted(nums)
		nums = heapsort(nums)
		print('sorted?', nums == srt_nums)

		#using custom heap
		nums=[5,4,6,3,7,3,8,2,9,1,0]
		M = CustomHeap(nums)
		for x in [10,11,12,15,30]:
			M.push(x)

		print('peek=',M.peek())
		while not M.empty():
			print(M.pop())

	if False: #mergesort
		import random
		random.seed(0)
		nums = [random.randint(0,2**31-1) for _ in range(1000)]
		srt_nums = sorted(nums)
		mergesort(nums)
		print('sorted?', nums == srt_nums)

	if False: #quicksort
		import random
		random.seed(0)
		nums = [random.randint(0,2**31-1) for _ in range(1000)]
		srt_nums = sorted(nums)
		quicksort(nums)
		print('sorted?', nums == srt_nums)

	if False: #radix-sort
		import random
		random.seed(0)
		nums = [random.randint(0,2**31-1) for _ in range(1000)]
		radix_sort(nums)
		print('Is sorted?', nums == sorted(nums))

	if False: #O(n^2) sorts
		import random
		random.seed(0)
		
		nums = [random.randint(0,2**31-1) for _ in range(1000)]
		srt_nums = sorted(nums)
		insertion_sort(nums)
		print('Insertion sorted?', nums == srt_nums)

		nums = [random.randint(0,2**31-1) for _ in range(1000)]
		srt_nums = sorted(nums)
		bubble_sort(nums)
		print('Bubble sorted?', nums == srt_nums)

	if False: #selection using quickselect & heapselect
		import random
		random.seed(0)
		nums = [random.randint(0,2**31-1) for _ in range(1000)]
		srt_nums = sorted(nums)
		N = 100

		heapselected_nums = [heapselect(nums, k) for k in range(1, N)]
		print('heapselect pass?', heapselected_nums == srt_nums[:N-1])
		
		quickselected_nums = [quickselect(nums, k-1) for k in range(1, N)]
		print('quickselect pass?', quickselected_nums == srt_nums[:N-1])

	if False: #Binary searching
		arr = sorted([5,4,6,3,7,3,8,2,9,1,0])
		print(arr)
		for x in arr:
			print(x,bisection_search(arr, x))

		for i in range(2,8):
			insort(arr, i)
		print(arr)

	if False: #Graph algorithms
		if False: #DAG
			vertices = range(11)
			#this is a DAG
			edges = [[0,1], [0,2], [1,4], [4,5], [1,3], [3,6],
					 [5,7], [2,7], [7,8], [8,9], [2,4], [7,3],
					]# [9,0]]

			g = Graph(vertices, edges)

			dists, parents = g.BFS(0)
			for v in vertices:
				print('v=',v,'dist from 0=',dists[v],'BF parent=',parents[v])
			
			parents, disc_times, fin_times = g.DFS()
			for v in vertices:
				print(v,':', disc_times[v],'-',fin_times[v])

			print('Has cycle?',g.has_cycle)
			print(g.topo_sort())

			min_path_dists, min_path_parents = g.bellman_ford(0)
			dijkstra_dists, dijkstra_parents = g.dijkstra(0)
			for v in vertices:
				bf_dist = min_path_dists[v]
				d_dist = dijkstra_dists[v]
				print('v=',v,'bellman dist=',bf_dist, 'dijkstra dist=',d_dist)
		
			print(g.find_tour(0))

		if True:
			vertices = range(11)
			edges = [[0,1], [0,2], [1,4], [4,5], [1,3], [3,6],
					 [5,7], [2,7], [7,8], [8,9], [2,4], [7,3],
					]# [9,0]]
			import random
			random.seed(0)
			wgts = [(u,v,random.randint(1,10)) for u,v in edges]
			
			#this is an undirected graph
			edges += [[v,u] for u,v in edges]
			wgts += [(v,u,w) for u,v,w in wgts]
			
			g = Graph(vertices, edges, weights=wgts)

			mst_parents = g.min_span_tree_PRIM()
			for u in vertices:
				p = mst_parents[u]
				print(p,'->',u,'w=', g._weights[p][u])

			mst_edges = g.min_span_tree_KRUK()
			print(sorted(mst_edges))

	if False: #Disjoint sets with union
		elems = range(10)
		dsu = DisjointSetUnion(elems)
		#make sets of even and odd numbers
		for x in elems:
			if x % 2 == 0:
				dsu.union(0,x)
			else:
				dsu.union(1,x)
		
		print(dsu.reps)

if __name__=='__main__':
	main()
