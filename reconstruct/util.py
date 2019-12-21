import tensorflow as tf
import contextlib
import time
import math
import pdb
import sys
import re


### Constructing scope names.

_scope_stack = []
@contextlib.contextmanager
def scope(name):
	global _scope_stack
	_scope_stack.append(name)
	with tf.name_scope(name):
		yield scope_name()
	_scope_stack.pop()

def scope_name(postfix=None):
	global _scope_stack
	if postfix:
		return "_".join(_scope_stack) + "_" + postfix
	else:
		return "_".join(_scope_stack)


### Reading tuples from a generator.

def in_tuples(n, generator, pad_with_none=False):
	'''Groups outputs of a generator into collections of n.
	   If out of data during a tuple, fills the missing elements with None if pad_with_none is True.
	   Otherwise discards the whole tuple with missing elements.'''
	it = iter(generator)
	
	while True:
		elements = []
		try:
			# Read n elements.
			for i in range(n):
				elements.append(next(it))
				
			# Got a full tuple.
			yield elements
		
		except StopIteration:
			# Out of data.
			if elements:
				# Still an unfinished tuple to handle.
				
				if pad_with_none:
					# Pad with None for the last time.
					elements.extend([None] * (n - len(elements)))
					yield elements
				
			# Done.
			raise StopIteration()
	

### Profiling.

class Timer:
	'''A simple timer for profiling.'''
	
	def __init__(self, skip_first=0):
		self.total_time = 0.0
		self.total_count = 0
		self.begin_time = None
		self.end_time = 0.0
		
		self.skip = skip_first
		
		
	def begin(self):
		self.begin_time = time.time()
	
	def end(self):
		self._validate()
	
		self.end_time = time.time()
		
		if self.skip > 0:
			self.skip -= 1
		else:
			self.total_time += self.end_time - self.begin_time
			self.total_count += 1
	
	def total(self):
		self._validate()
		return self.total_time
	
	def count(self):
		self._validate()
		return self.total_count
	
	def mean(self):
		self._validate()
		return self.total() / float(self.count())
		
	def in_use(self):
		return self.begin_time is not None
	
	def _validate(self):
		if self.begin_time is None:
			raise Exception('Timer never started.')		


class TimerCollection:
	'''A collection of timers for profiling execution times.
	
	   Supports skipping a number of first timings which often take a longer time in TF.'''
	
	def __init__(self, skip_first=0):
		self.skip_first = skip_first
		self.custom_skips = {}
		
		self.timers = {}
	
	def setSkipFirst(self, name, skip_first):
		self.custom_skips[name] = skip_first
		
	def getTimer(self, name, allow_creation=False):
		if name in self.timers:
			return self.timers[name]
		
		if not allow_creation:
			return None
		
		skip_first = self.custom_skips[name] if name in self.custom_skips else self.skip_first
		timer = Timer(skip_first=skip_first)
		
		self.timers[name] = timer
		return timer
	
	def __getitem__(self, key):
		return self.getTimer(key, allow_creation=True)
	
	def begin(self, name):
		self.getTimer(name, allow_creation=True).begin()
	
	def end(self, name):
		self.getTimer(name).end()
		
	def total(self, name):
		return self.getTimer(name).total()

	def count(self, name):
		return self.getTimer(name).count()

	def mean(self, name):
		return self.getTimer(name).mean()
	
	def report(self):
		names = []
		total_times = []
		percentages = []
		
		total_time = self.total('total')
		
		# Add other benchmarks.
		for name, timer in sorted(self.timers.items()):
			if name == 'total' or not timer.in_use():
				continue
			
			names.append(name)
			total_times.append(timer.total())
			percentages.append(100.0 * timer.total() / total_time if total_time > 0 else float('nan'))
		
		# Add total.
		names.append('total')
		total_times.append(total_time)
		percentages.append(100.0)
		
		row_format = "{:>30}{:>15}{:>15}"
		print(row_format.format('NAME', 'TOTAL TIME', 'PERCENTAGE'))
		for name, total, percentage in zip(names, total_times, percentages):
			print(row_format.format(name, "{:.1f} s".format(total), "{:.1f}".format(percentage)))
		
			
		
def benchmark_iteration(generator, timer):
	'''Wraps a generator inside a timer.'''
	
	it = iter(generator)
	
	try:
		while True:
			timer.begin()
			next_value = next(it)
			timer.end()
			yield next_value
	
	except StopIteration:
		raise StopIteration()
	
	
def int_arg(name, default_value=None):
	'''Reads the integer value of a given command line parameter.'''
	result = default_value
	
	for arg in sys.argv[1:]:
		match = re.match('--{}=([0-9]+)'.format(name), arg)
		if match:
			result = int(match.groups()[0])
	
	return result
	
	
def digit_reversed_range(maximum, base):
	'''Return a list of integers from 0 to maximum-1 ordered in reverse digit order.'''
	
	def next_pow_k(i, k):
		'''Returns the smallest power of k that is greater than i.'''
		n = 1
		while n <= i:
			n *= k
		return n
	
	next_pow = next_pow_k(maximum - 1, base)
	result = []
	
	for i in range(next_pow):
		it = base
		reverse_it = next_pow // base
		j = 0
		n = i
		while reverse_it > 0:
			j += (n % it) * reverse_it
			n //= base
			reverse_it //= base
		
		if j < maximum:
			result.append(j)
	
	return result
