# coding: utf-8

import numpy as np
import cv2

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, iterators, Chain, optimizers, training
from chainer.training import updaters, Trainer, extensions
from chainer.backends import cuda

import matplotlib
matplotlib.use("Agg")

import argparse, pprint, json
from pathlib import Path

from model import Discriminator, Generator


def parse_args():
	import argparse
	desc = """
	trains GAN to be used for anomaly detection
	"""
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument("-s", "--setting", required=True, help="setting file")
	parser.add_argument("-r", "--result", required=True, help="result directory")
	parser.add_argument("-g", "--gpu", type=int, default=-1, help="GPU")
	args = parser.parse_args()
	
	pprint.pprint(vars(args))
	main(args)
	
def main(args):
	with open(args.setting, "r") as f:
		setting = json.load(f)
	pprint.pprint(setting)
		
	chainer.config.user_gpu = args.gpu
	if args.gpu >= 0:
		cuda.get_device_from_id(args.gpu).use()
		print("GPU mode")
		
	train_iter = get_iterator(setting)
	
	gen = Generator(setting["zdim"])
	dis = Discriminator()
	if args.gpu >= 0:
		gen.to_gpu()
		dis.to_gpu()
	gen_opt = optimizers.Adam(**setting["optimizer"])
	dis_opt = optimizers.Adam(**setting["optimizer"])
	gen_opt.setup(gen)
	dis_opt.setup(dis)
	if setting["weight_decay"] > 0:
		gen_opt.add_hook(chainer.optimizer_hooks.WeightDecay(setting["weight_decay"]))
		dis_opt.add_hook(chainer.optimizer_hooks.WeightDecay(setting["weight_decay"]))
		
	updater = GANUpdater(train_iter, gen_opt, dis_opt, setting["zdim"])
	trainer = training.Trainer(updater, (setting["epoch"], "epoch"), out=args.result)
	trainer.extend(extensions.LogReport())
	trainer.extend(extensions.PrintReport(["epoch", "generator/loss", "discriminator/loss"]))
	trainer.extend(extensions.ProgressBar(update_interval=1))
	trainer.extend(extensions.PlotReport(("generator/loss","discriminator/loss"),"epoch", file_name="loss_plot.pdf"))
	trainer.extend(ext_save_img(gen, setting["zdim"], 10, args.result), trigger=(1, "epoch"))
	trainer.extend(extensions.dump_graph(root_name="generator/loss", out_name="gen_cg.dot"))
	trainer.extend(extensions.dump_graph(root_name="discriminator/loss", out_name="dis_cg.dot"))
	trainer.extend(extensions.snapshot_object(gen, 'gen_epoch_{.updater.epoch:04d}.model'), trigger=(1,"epoch"))
	trainer.extend(extensions.snapshot_object(dis, 'dis_epoch_{.updater.epoch:04d}.model'), trigger=(1,"epoch"))
	
	trainer.run()
	
def get_iterator(setting):
	mnist_dataset = get_mnist_num(setting["normal_dig"])
	return iterators.SerialIterator(mnist_dataset, setting["batch_size"], shuffle=True, repeat=True)

def get_mnist_num(dig_list:list, test=False) -> np.ndarray:
	mnist_dataset = chainer.datasets.get_mnist(ndim=3)[1 if test else 0]	# MNISTデータ取得
	# 指定された数字だけ抽出
	mnist_dataset = [img for img,label in mnist_dataset[:] if label in dig_list]
	mnist_dataset = np.stack(mnist_dataset)
	return mnist_dataset
	
	
# 生成画像を保存するextension
def ext_save_img(generator, zdim:int, n:int, out:str):
	out_dir_path = Path(out, "out_images")
	try:
		out_dir_path.mkdir(parents=True)
	except FileExistsError:
		pass

	@chainer.training.make_extension(trigger=(1,"epoch"))
	def _ext_save_img(trainer):
		z = np.random.rand(n, zdim).astype(np.float32)
		z = Variable(z)
		if chainer.config.user_gpu >= 0:
			z.to_gpu()
		with chainer.using_config("train", False):
			x = generator(z).array.squeeze(axis=1)
		if chainer.config.user_gpu >= 0:
			x = generator.xp.asnumpy(x)
		
		epoch = trainer.updater.epoch
		for i,_x in enumerate(x):
			img = (_x*255).astype(np.uint8)
			assert len(img.shape) == 2, (img.shape, img.dtype)
			filename = out_dir_path / "out_epoch{:03d}_{:03d}.png".format(epoch, i)
			cv2.imwrite(str(filename), img)

	return _ext_save_img

	

class GANUpdater(updaters.StandardUpdater):
	def __init__(self, iterator, gen_opt, dis_opt, zdim:int):
		opt_dict = {
			"generator": gen_opt,
			"discriminator": dis_opt
		}
		super().__init__(iterator, opt_dict)
		self.zdim = zdim
		
	def update_core(self):
		gen_opt = self.get_optimizer("generator")
		dis_opt = self.get_optimizer("discriminator")
		gen = gen_opt.target
		dis = dis_opt.target
		
		x_real = self.get_batch()
		assert len(x_real.shape) == 4
		x_real.name = "real images"
		
		batch_size = len(x_real)
		
		z = np.random.rand(batch_size, self.zdim).astype(np.float32)
		z = Variable(z, name="latent vector")
		if chainer.config.user_gpu >= 0:
			z.to_gpu()
			
		# update discriminator
		# 本物に対して大きな値，偽物に対して小さな値を出力させる
		x_fake = gen(z)
		x_fake.name = "fake images"
		y_fake = dis(x_fake)
		y_fake.name = "score for fake images"
		y_real = dis(x_real)
		y_real.name = "score for real images"
		dis_loss = F.mean(F.softplus(y_fake)+F.softplus(-y_real))
		dis_loss.name = "discriminator loss"
		
		dis.cleargrads()
		dis_loss.backward()
		dis_opt.update()
		
		# update generator
		x_fake = gen(z)
		x_fake.name = "fake images"
		y_fake = dis(x_fake)
		y_fake.name = "score for fake images"
		gen_loss = F.mean(-F.softplus(y_fake))
		gen_loss.name = "generator loss"
		
		dis.cleargrads()
		gen.cleargrads()
		gen_loss.backward()
		gen_opt.update()
		
		chainer.report({
			"generator/loss": gen_loss,
			"discriminator/loss": dis_loss
		})
		
		
	def get_batch(self) -> Variable:
		iterator = self.get_iterator("main")
		batch = iterator.next()
		batch = Variable(np.stack(batch))
		if chainer.config.user_gpu >= 0:
			batch.to_gpu()
			
		return batch

if __name__ == "__main__":
	parse_args()
	