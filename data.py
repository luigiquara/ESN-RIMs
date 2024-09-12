import torch
import struct
import numpy as np
import gzip
import cv2

def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


#if __name__ == '__main__':
#	a = read_idx('mnist/train-labels-idx1-ubyte.gz')
#	print(a.shape)


class MnistData:
	def __init__(self, batch_size, size, k, subset_size=None):
		#self.train_data = read_idx('/content/drive/My Drive/Recurrent Independent Mechanisms/mnist/train-images-idx3-ubyte.gz')
		#self.train_labels = read_idx('/content/drive/My Drive/Recurrent Independent Mechanisms/mnist/train-labels-idx1-ubyte.gz')
		#self.val_data = read_idx('/content/drive/My Drive/Recurrent Independent Mechanisms/mnist/t10k-images-idx3-ubyte.gz')
		#self.val_labels = read_idx('/content/drive/My Drive/Recurrent Independent Mechanisms/mnist/t10k-labels-idx1-ubyte.gz')


		self.train_data = read_idx('mnist/train-images-idx3-ubyte.gz')
		self.train_labels = read_idx('mnist/train-labels-idx1-ubyte.gz')
		self.val_data = read_idx('mnist/t10k-images-idx3-ubyte.gz')
		self.val_labels = read_idx('mnist/t10k-labels-idx1-ubyte.gz')

		if subset_size > 0:
			train_idxs = np.random.choice(self.train_data.shape[0], subset_size, replace=False)
			val_idxs = np.random.choice(self.val_data.shape[0], size=subset_size, replace=False)
			self.train_data = self.train_data[train_idxs]
			self.train_labels = self.train_labels[train_idxs]
			self.val_data = self.val_data[val_idxs]
			self.val_labels = self.val_labels[val_idxs]

		train_data_ = np.zeros((self.train_data.shape[0], size[0] * size[1]))
		val_data_ = np.zeros((self.val_data.shape[0], size[0] * size[1]))
		val_data_24 = np.zeros((self.val_data.shape[0], (size[0]  + 10)* (size[1] + 10)))
		val_data_19 = np.zeros((self.val_data.shape[0], (size[0] + 5) * (size[1] + 5)))
		val_data_16 = np.zeros((self.val_data.shape[0], (size[0] + 2) * (size[1] + 2)))

		for i in range(self.train_data.shape[0]):
			img = self.train_data[i, :]
			img = cv2.resize(img, size, interpolation = cv2.INTER_NEAREST)

			_, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY) 
			
			img = np.reshape(img, (-1))
			train_data_[i, :] = img

		# resize val images with given sizes
		for i in range(self.val_data.shape[0]):
			img = self.val_data[i, :]

			# standard imgs, as in training
			img0 = cv2.resize(img, (size), interpolation = cv2.INTER_NEAREST)
			_, img0 = cv2.threshold(img0, 120, 255, cv2.THRESH_BINARY) 
			img0 = np.reshape(img0, (-1))
			val_data_[i, :] = img0

			# resize to 24x24
			img1 = cv2.resize(img, (size[0] + 10, size[1] + 10), interpolation = cv2.INTER_NEAREST)
			_, img1 = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY) 
			img1 = np.reshape(img1, (-1))
			val_data_24[i, :] = img1

			# resize to 19x19
			img2 = cv2.resize(img, (size[0] + 5, size[1] + 5), interpolation = cv2.INTER_NEAREST)
			_, img2 = cv2.threshold(img2, 120, 255, cv2.THRESH_BINARY) 
			img2 = np.reshape(img2, (-1))
			val_data_19[i, :] = img2

			# resize to 16x16
			img3 = cv2.resize(img, (size[0] + 2, size[1] + 2), interpolation = cv2.INTER_NEAREST)
			_, img3 = cv2.threshold(img3, 120, 255, cv2.THRESH_BINARY) 
			img3 = np.reshape(img3, (-1))
			val_data_16[i, :] = img3
			
		self.train_data = train_data_
		self.val_data = val_data_
		self.val_data24 = val_data_24
		self.val_data19 = val_data_19
		self.val_data16 = val_data_16

		del train_data_
		del val_data_

		self.train_data = np.reshape(self.train_data, (self.train_data.shape[0], self.train_data.shape[1], 1))
		self.val_data = np.reshape(self.val_data, (self.val_data.shape[0], self.val_data.shape[1], 1))
		self.val_data24 = np.reshape(self.val_data24, (self.val_data24.shape[0], self.val_data24.shape[1], 1))
		self.val_data19 = np.reshape(self.val_data19, (self.val_data19.shape[0], self.val_data19.shape[1], 1))
		self.val_data16 = np.reshape(self.val_data16, (self.val_data16.shape[0], self.val_data16.shape[1], 1))

		self.train_data = [self.train_data[i:i + batch_size] for i in range(0, self.train_data.shape[0], batch_size)]
		self.val_data = [self.val_data[i:i + 512] for i in range(0, self.val_data.shape[0], 512)]
		self.val_data24 = [self.val_data24[i:i + 512] for i in range(0, self.val_data24.shape[0], 512)]
		self.val_data19 = [self.val_data19[i:i + 512] for i in range(0, self.val_data19.shape[0], 512)]
		self.val_data16 = [self.val_data16[i:i + 512] for i in range(0, self.val_data16.shape[0], 512)]
		self.train_labels = [self.train_labels[i:i + batch_size] for i in range(0, self.train_labels.shape[0], batch_size)]
		self.val_labels = [self.val_labels[i:i + 512] for i in range(0, self.val_labels.shape[0], 512)]

	def train_len(self):
		return len(self.train_labels)

	def val_len(self):
		return len(self.val_labels)

	def train_get(self, i):
		return self.train_data[i], self.train_labels[i]
	
	def val_get(self, i):
		return self.val_data[i], self.val_labels[i]

	def val_get24(self, i):
		return self.val_data24[i], self.val_labels[i]

	def val_get19(self, i):
		return self.val_data19[i], self.val_labels[i] 

	def val_get16(self, i):
		return self.val_data16[i], self.val_labels[i]	
