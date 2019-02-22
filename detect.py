# coding: utf-8

import numpy as np
import cv2, tqdm

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, iterators, Chain, optimizers, training, serializers
from chainer.training import updaters, Trainer, extensions
from chainer.backends import cuda

import matplotlib
matplotlib.use("Agg")

import argparse, pprint, json
from pathlib import Path

from model import Discriminator, Generator
from train_gan import get_mnist_num

def parse_args():
	import argparse
	desc = """
	Performs anomaly detection
	"""
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument("-s", "--setting", required=True, help="setting file used for training GAN")
	parser.add_argument("--gen", required=True, help="Generator model file trained by train_gan.py")
	parser.add_argument("--dis", required=True, help="Discriminator model trained by train_gan.py")
	parser.add_argument("-r", "--result", required=True, help="result directory")
	parser.add_argument("-g", "--gpu", type=int, default=-1, help="GPU")
	parser.add_argument("-i", "--iteration", type=int, default=1000, help="iteration number of optimizing z")
	parser.add_argument("-n", "--num", type=int, default=10, help="number of images")
	parser.add_argument("--lam", type=float, default=0.1, help="balance between residual loss and feature matching loss")
	args = parser.parse_args()
	
	pprint.pprint(vars(args))
	main(args)
	
def main(args):
	with open(args.setting, "r") as f:
		setting = json.load(f)
	pprint.pprint(setting)
	
	try:
		Path(args.result).mkdir(parents=True)
	except FileExistsError:
		pass
	
	chainer.config.user_gpu = args.gpu
	if args.gpu >= 0:
		cuda.get_device_from_id(args.gpu).use()
		print("GPU mode")
		
	# sample test data
	normal_data, abnormal_data = get_random_data(setting["normal_dig"], args.num)
	
	# load models
	generator = Generator(setting["zdim"])
	discriminator = Discriminator()
	serializers.load_npz(args.gen, generator)
	serializers.load_npz(args.dis, discriminator)
	if args.gpu >= 0:
		generator.to_gpu()
		discriminator.to_gpu()
		
	dig = np.concatenate([normal_data, abnormal_data], axis=0)
	dig = Variable(dig)
	if args.gpu >= 0:
		dig.to_gpu()
	
	_, x_gen = detect(dig, generator, discriminator, args.lam, args.iteration, setting["zdim"])
	
	x_gen = x_gen.array
	dig = dig.array
	if args.gpu >= 0:
		x_gen = generator.xp.asnumpy(x_gen)
		dig = generator.xp.asnumpy(dig)
	
	# saves generated images from optimized z and diccefence between original images and them
	result_path = Path(args.result)
	for i, t in enumerate(zip(x_gen, dig)):
		img, orig = t
		img = img.squeeze()
		diff_img = np.abs(img - orig.squeeze())
		diff_img = 255.0*diff_img / diff_img.max()
		
		img = (img*255).astype(np.uint8)
		diff_img = diff_img.astype(np.uint8)
		p1 = result_path / "gen_{:03d}.png".format(i)
		p2 = result_path / "diff_{:03d}.png".format(i)
		cv2.imwrite(str(p1), img)
		cv2.imwrite(str(p2), diff_img)
	
def get_random_data(normal_dig:list, num:int) -> tuple:
	normal_data = get_mnist_num(normal_dig, test=True)
	indices = np.random.choice(len(normal_data), size=num, replace=False)
	normal_data = np.array(normal_data[indices])
	
	abnormal_dig = set(range(10)) - set(normal_dig)
	abnormal_data = get_mnist_num(abnormal_dig, test=True)
	indices = np.random.choice(len(abnormal_data), size=num, replace=False)
	abnormal_data = np.array(abnormal_data[indices])
	
	return normal_data, abnormal_data
	
def detect(dig:Variable, gen, dis, lam, iteration, zdim):
	def compute_loss(z):
		with chainer.using_config("train", False):
			x_gen = gen(z)
			d_gen = dis.forward_with_feature_map(x_gen)[1]
			d_real = dis.forward_with_feature_map(dig)[1]
			
		res_loss = F.mean_absolute_error(x_gen, dig)
		feature_loss = F.mean_absolute_error(d_gen, d_real)
		total_loss = (1.0-lam)*res_loss + lam*feature_loss
		
		return total_loss, x_gen
	
	assert len(dig.shape) == 4, dig.shape
	
	z = np.random.rand(len(dig), zdim).astype(np.float32)
	z = chainer.links.Parameter(z)
	
	if chainer.config.user_gpu >= 0:
		z.to_gpu()
	
	opt = optimizers.Adam()
	opt.setup(z)
	
	for i in tqdm.trange(iteration):
		total_loss = compute_loss(z())[0]
		
		for m in (gen, dis, z):
			m.cleargrads()
		total_loss.backward()
		opt.update()
		
		if i%250 == 0:
			tqdm.tqdm.write(str(total_loss.array))
		
	return compute_loss(z())
	
	
if __name__ == "__main__":
	parse_args()
	