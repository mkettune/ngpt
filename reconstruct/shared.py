import multiprocessing
import ctypes
import struct
import pdb

import numpy as np


# Some basic structures for fast communication between processes.


_ctypes = {np.uint32 : ctypes.c_uint32,
		   np.uint16 : ctypes.c_uint16,
		   np.ubyte : ctypes.c_uint8,
		   np.float16 : ctypes.c_uint16,
		   np.float32 : ctypes.c_uint32}

class CircularBuffer:
	'''A circular buffer implemented using numpy arrays, shareable with multiple processes by the multiprocessing module.
	   Use .lock.acquire() and .lock.release() before calling any functions.'''
	
	def __init__(self, dtype=None, dimensions=None, from_shared=None):									
		if from_shared:
			self.memory, self.lock, self.begin, self.end, self.size, self.dtype, self.dimensions = from_shared
		else:
			self.size = dimensions[0]
			self.dtype = dtype
			self.dimensions = dimensions

			self.memory = multiprocessing.Array(_ctypes[dtype], np.prod(dimensions, dtype=np.uint64).item())
			self.lock = multiprocessing.Lock()
			self.begin = multiprocessing.Value('i', 0)
			self.end = multiprocessing.Value('i', 0)

		self.np_data = np.frombuffer(self.memory.get_obj(), dtype=self.dtype).reshape(*self.dimensions)

	def getSharedMemory(self):
		'''Returns the shared memory used by the circular buffer.
		   This can be transmitted to another process by using multiprocessing, to construct
		   a view to this CircularBuffer for that process.'''
		return (self.memory, self.lock, self.begin, self.end, self.size, self.dtype, self.dimensions)
		
	def getDataSize(self):
		return self.end.value - self.begin.value
	
	def getFreeSize(self):
		return self.size - self.end.value + self.begin.value
		
	def append(self, data):				
		if self.getFreeSize() <= 0:
			# Full.
			return False
		
		# Add.
		self.np_data[self.end.value % self.size] = data
		self.end.value += 1

		return True
		
	def consume(self):
		if self.getDataSize() <= 0:
			# Empty.
			return None
		
		# Consume.
		data = self.np_data[self.begin.value]
		self.begin.value += 1
		
		if self.begin.value >= self.size:
			self.begin.value -= self.size
			self.end.value -= self.size
		
		return data

class BytesArray:
	'''A contiguous block of memory that maps integers to variable length byte arrays that may be written or read from different processes. Internally, holds a dictionary of where the data of each index is located.
	
	Use .lock.acquire() and .lock.release() before calling any functions.
	
	Example:
		bytes_list[3] = b'foo'
	'''

	def __init__(self, memory_size=None, from_shared=None, allow_empty_bytes=False):
		if memory_size:
			self.memory_size = memory_size
			self.memory = multiprocessing.Array(ctypes.c_uint8, memory_size)
			self.lock = multiprocessing.Lock()
			self.list_size = multiprocessing.Value('L', 0)

			self.data_offset = multiprocessing.Value('L', 0)
			self.dictionary_offset = multiprocessing.Value('L', self.memory_size)
			
			self.allow_empty_bytes = allow_empty_bytes
		elif from_shared:
			self.memory_size, self.memory, self.lock, self.data_offset, self.dictionary_offset, self.allow_empty_bytes = from_shared
		else:
			raise Exception("Invalid BytesArray parameters")

		self.memory_view = np.frombuffer(self.memory.get_obj(), dtype=np.uint8)
	
	def clear(self):
		'''Prepares the BytesArray for reuse.'''
		# Clear dictionary.
		self.memory_view[self.dictionary_offset.value:] = np.zeros(self.memory_size - self.dictionary_offset.value, dtype=np.uint8)
			
		self.list_size.value = 0
		self.data_offset.value = 0
		self.dictionary_offset.value = self.memory_size
		
	def getSharedMemory(self):
		'''Returns the shared memory used by the BytesArray.
		   This can be transmitted to another process by using multiprocessing, to construct
		   a view to this BytesArray for that process.'''
		return (self.memory_size, self.memory, self.lock, self.data_offset, self.dictionary_offset, self.allow_empty_bytes)

	def __len__(self):
		return (self.memory_size - self.dictionary_offset.value) // 8
	
	def __getitem__(self, key):
		if key < 0 or key >= len(self):
			raise Exception("Index out of range in BytesArray")
		
		dictionary_offset = self.memory_size - (1 + key) * 8
				
		data_offset, data_length = struct.unpack('<2I', self.memory_view[dictionary_offset : dictionary_offset + 8].tobytes())
		
		if not self.allow_empty_bytes and data_length == 0:
			raise Exception("Accessing unwritten index {} in BytesArray".format(key))
		
		return self.memory_view[data_offset : data_offset + data_length].tobytes()

	def __setitem__(self, key, item):
		dictionary_offset = self.memory_size - 8 * (key + 1)
		if (dictionary_offset < 0 or dictionary_offset > self.memory_size or 
				self.data_offset.value + len(item) + 8 > self.dictionary_offset.value or
				self.data_offset.value + len(item) + 8 > dictionary_offset):
			raise Exception("Out of memory in BytesArray")
		
		if not self.allow_empty_bytes and len(item) == 0:
			raise Exception("Assigning empty bytes not allowed in BytesArray configuration")
			
		# Add to dictionary.
		self.memory_view[dictionary_offset : dictionary_offset + 8] = np.frombuffer(struct.pack('<2I', self.data_offset.value, len(item)), dtype=np.uint8)
		
		# Set memory.
		self.memory_view[self.data_offset.value : self.data_offset.value + len(item)] = np.frombuffer(item, dtype=np.uint8)
		
		# Update internal data.
		self.data_offset.value += len(item)
		self.dictionary_offset.value = min(self.dictionary_offset.value, dictionary_offset)


if __name__ == '__main__':
	# TODO: Add more tests.
	b = BytesArray(1000)
	b[0] = b'103'
	b[2] = b'1007'

	assert b[0] == b'103'
	assert b[2] == b'1007'
	assert len(b) == 3
	
	c = BytesArray(from_shared=b.getSharedMemory())
	assert c[0] == b'103'
	assert c[2] == b'1007'
	assert len(c) == 3
	