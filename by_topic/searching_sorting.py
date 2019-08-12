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

