# Back Pocket Algorithms

### A collection of fundamental data structures and algorithms

## What is this?
This is a collection of implementations of fundamental data structures and algorithms; I do not claim claim it is exhaustive, but it has been a useful tool and reference when preparing for software interviews and (subsequently) when working as a software engineer at Microsoft Research. 

## Contents
All algorithms have at least some unit tests, which are in the ``if __name__ == __main__:`` section of the files below.

``by_topic/data_structures.py``
1. Stack
2. Queue
3. Linked List
4. Trie
5. Binary Search Tree
6. Heaps

``by_topic/searching_sorting.py``
1. heapsort & heapselect
2. mergesort
3. quicksort & quickselect
4. bisection search & insort
5. counting sort & radix sort
6. insertion sort & bubble sort

``by_topic/graphs_sets.py``
1. Simple graph object with methods for
	1. BFS & DFS
	2. min-spanning tree (Prim's and Kruksal's algorithms)
	3. topological sort
	4. Bellman-Ford algorithm
	5. Djikstra's algorithm
	6. DAG shortes-path algorithm
2. Disjoint-Set-Union object with methods for
	1. find
	2. union

``combined.py``
- All the algorithms above, but in one file. Just 'cause.

``template.py``
- The function signatures for the above algorithms (and their helpers) but without any of the implementation code. 
- This is a good self-test: sit down and fill in the template. You will quickly find out which algorithms you don't know but thought you did. Then iterate.

- what's not in here: 
	- notably, **dynamic programming, greedy algorithms, etc are not here.** These problems are much more varied and than their data structures and algorithms cousins, so it's harder to systematically compile them into a self-contained list without simply enumerating examples. That's for another day. 
	- Interesting but complicated data structures and algorithms such as Red-Black trees, B-Trees, Fibonacci heaps, Van Emde Boas trees, etc. There are better things to do than memorize these ones. 

## Design Principles

- **Optimization vs Pragmatism**: There exist many variations and optimized versions of these fundamental algorithms. The goal was to find implementations that are 'in your back pocket', to be implemented in minimal time, with good readability, and good performance. For a more comprehensive, albeit less pragmatic, view see CLRS.

- **Readable but Not Verbose**: At \~ 1000 total lines, there's a enough to digest on without reading long comments. I've tried to be concise with my syntax and comments, without impacting useability. 

- **Don't Re-invent the Wheel**: To the extent possible, I've used useful syntactic and algorithmic features of Python where appropriate, since good command of the builtin libraries is important. For instance, if you need a *custom* binary heap implementation (for an application or for an interivew) then writing ``import heapq`` doesn't count. However, if you're implementing k-way merge (in Python?), using ``heapq`` is probably fine.

- **Self-Contained**: These implementations are generally self-contained, using at most components from the Python Standard Library (things you should be able to use in an interview, depending on the task), or perhaps a small helper class/method (think ``LinkedListNode(...)`` for ``LinkedList(...)``). Batteries included.

## Why
I initially created these as a way to prepare for software engineering interviews. My opinion is that one should have all \~850 lines of these implementations **memorized**. We can debate the connections between *understanding* and *memorization* but, in an interview, not having to re-derive a binary heap or quicksort can save time for solving the actual tricky problems. In the wild, memorization is perhaps less important, but having these algorithms and *ideas* close at hand will ensure you use the right tool for the job. 
