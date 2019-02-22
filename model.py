# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, iterators, Chain, optimizers
from chainer.training import updaters, Trainer, extensions
from chainer.backends import cuda


def compose(x, funcs:list):
	y = x
	for f in funcs:
		y = f(y)
		
	return y


N = 32
class Discriminator(Chain):
	def __init__(self):
		super().__init__()
		kwds = {
			"ksize": 4,
			"stride": 2,
			"pad": 1,
			"nobias":True
		}
		with self.init_scope():
			self.conv1 = L.Convolution2D(1, N, **kwds)	# 14
			self.bn1 = L.BatchNormalization(N)
			self.conv2 = L.Convolution2D(N, N*2, **kwds)	# 7
			self.bn2 = L.BatchNormalization(N*2)
			self.conv3 = L.Convolution2D(N*2, N*4, ksize=2, stride=1, pad=0, nobias=True)	# 6
			self.bn3 = L.BatchNormalization(N*4)
			self.conv4 = L.Convolution2D(N*4, N*8, **kwds)	# 3
			self.bn4 = L.BatchNormalization(N*8)
			self.conv_out = L.Convolution2D(N*8, 1, ksize=1, stride=1, pad=0)
			
	def __call__(self, x):
		return self.forward_with_feature_map(x)[0]
		
	def forward_with_feature_map(self, x):
		assert len(x.shape) == 4
		fmap = compose(x, [
			self.conv1, self.bn1, F.relu,
			self.conv2, self.bn2, F.relu,
			self.conv3, self.bn3, F.relu,
			self.conv4, self.bn4, F.relu
		])
		
		h = compose(fmap, [
			self.conv_out,
			lambda x:F.mean(x, axis=(1,2,3))
		])
		return h, fmap

M = 5
class Generator(Chain):
	def __init__(self, zdim=20):
		super().__init__()
		kwds = {
			"ksize": 4,
			"stride": 2,
			"pad": 1,
			"nobias": True
		}
		with self.init_scope():
			self.fc_in = L.Linear(zdim, 3*3*M)	# GAPに対応
			self.conv_in = L.Convolution2D(M, N*8, ksize=1, stride=1, pad=0)	# conv_outに対応 -> 3
			self.bn_in = L.BatchNormalization(N*8)
			self.deconv1 = L.Deconvolution2D(N*8, N*4, **kwds)	# -> 6
			self.bn1 = L.BatchNormalization(N*4)
			self.deconv2 = L.Deconvolution2D(N*4, N*2, ksize=2, stride=1, pad=0, nobias=True)	# -> 7
			self.bn2 = L.BatchNormalization(N*2)
			self.deconv3 = L.Deconvolution2D(N*2, N, **kwds)	# -> 14
			self.bn3 = L.BatchNormalization(N)
			self.deconv4 = L.Deconvolution2D(N, 1, ksize=4, stride=2, pad=1)	# -> 28
			
	def __call__(self, z):
		assert len(z.shape) == 2
		h = compose(z, [
			self.fc_in,
			lambda x:F.reshape(x, (-1,M,3,3)),
			self.conv_in, self.bn_in, F.relu,
			self.deconv1, self.bn1, F.relu,
			self.deconv2, self.bn2, F.relu,
			self.deconv3, self.bn3, F.relu,
			self.deconv4, F.sigmoid
		])
		return h
		