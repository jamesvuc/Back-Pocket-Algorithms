"""
DO NOT OVERWITE THIS!
"""

class Stack:
	def __init__(self):
		pass
	def push(self, x):
		pass
	def peek(self):
		pass
	def pop(self):
		pass
	def empty(self):
		pass

class Queue:
	def __init__(self):
		pass
	def push(self, x):
		pass
	def peek_top(self):
		pass
	def pop(self):
		pass
	def empty(self):
		pass

class LinkedListNode:
	def __init__(self, data):
		pass

class DoubleLinkedList:
	def __init__(self):
		pass
	def insert_front(self, node):
		pass
	def insert_back(self, node):
		pass
	def remove_node(self, node):
		pass
	def empty(self):
		pass

class TrieNode:
	def __init__(self):
		pass

class Trie:
	def __init__(self):
		pass
	def insert(self, word):
		pass
	def search(self, word):
		pass

class BSTNode:
	def __init__(self, key, value=None):
		pass

class BST:
	def __init__(self):
		pass
	def insert(self, node):
		pass
	def search(self, key):
		pass
	# def _find_min(self, node):
		# pass
	# def _transplant(self, old_node, new_node):
	# 	pass
	def remove(self, node):
		pass
	def succ(self, node, process_node = print):
		pass
	def pred(self, node, process_node = print):
		pass
	def inorder(self, node):
		pass
	def preorder(self, node):
		pass
	def postorder(self, node):
		pass
	def empty(self):
		pass

class MinHeap:
	def __init__(self):
		pass
	def push(self, x):
		pass
	def pop(self):
		pass
	def replace(self, x):
		pass
	def peek(self):
		pass
	def empty(self):
		pass

class MaxHeap:
	def __init__(self):
		pass
	def push(self, x):
		pass
	def pop(self):
		pass
	def replace(self, x):
		pass
	def peek(self):
		pass

class CustomHeap:
	def __init__(self):
		pass
	# def _heapify(self, idx):
	# 	pass
	# def _increase_key(self, idx, newkey):
	# 	pass
	def push(self, x):
		pass
	def pop(self):
		pass
	def peek(self):
		pass
	def empty(self):
		pass

def heapsort(arr):
	pass

def heapselect(arr, k):
	pass
# def merge(arr):
# 	pass
# def _mergesort():
# 	pass
def mergesort(arr):
	pass
# def partition():
# 	pass
# def _quicksort():
# 	pass
def quicksort(arr):
	pass
# def _quickselect():
# 	pass
def quickselect(arr, k):
	pass
def bisection_search(arr, x, lo=0, hi=None):
	pass
def insort(arr, x):
	pass
# def get_ith_digit():
# 	pass
def counting_sort(arr):
	pass
def radix_sort(arr):
	pass
def insertion_sort(arr):
	pass
def bubble_sort(arr):
	pass

class Graph:
	def __init__(self):
		pass
	def BFS(self, src):
		pass
	def DFS(self):
		pass
	def min_span_tree(self):
		pass
	def topo_sort(self):
		pass
	def detect_cycle(self):
		pass
	def find_tour(self, src):
		pass
	def bellman_ford(self, src):
		pass
	def DAG_shortest_path(self, src):
		pass
	def dijkstra(self, src):
		pass

class DisjointSetUnion:
	def __init__(self):
		pass
	def find(self, x):
		pass
	def union(self, x, y):
		pass