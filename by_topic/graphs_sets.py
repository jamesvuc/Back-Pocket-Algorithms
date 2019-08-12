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
			u = Q.pop() #dequeue a node to visit
			colours[u] = 'g'
			for v in self._graph[u]:
				if colours[v] == 'w': #if the colour is white (unseen)
					dists[v] = dists[u] +1 #dist to v is dist to u + 1
					parents[v] = u #v was discovered via u => u is v's parent
					Q.appendleft(v) #push v onto the queue for processing

			colours[u] = 'b' #set the colour of u, all of whose children are either
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
			for u in self._graph:						# iterate through vertices
				for v in self._graph[u]:				# iterate through edges
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