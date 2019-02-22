# Chainer implementation for AnoGAN
[Schlegl, Thomas, et al. "Unsupervised anomaly detection with generative adversarial networks to guide marker discovery." International Conference on Information Processing in Medical Imaging. Springer, Cham, 2017.](https://arxiv.org/abs/1703.05921)

## Requirements
Chainer, OpenCV, NumPy, Matplotlib

```bash
$ pip install chainer opencv-python numpy matplotlib
```

## How to run
### `train_gan.py`
Trains GAN for anomaly detection.

Digits that specified in `setting.json` are generated.

```bash
$ python train_gan.py [options]
```

You can read help with `-h` option.

```bash
$ python train_gan.py -h
usage: train_gan.py [-h] -s SETTING -r RESULT [-g GPU]

trains GAN to be used for anomaly detection

optional arguments:
  -h, --help            show this help message and exit
  -s SETTING, --setting SETTING
                        setting file
  -r RESULT, --result RESULT
                        result directory
  -g GPU, --gpu GPU     GPU
```

### `detect.py`
Performs anomaly detection.

Digits that specified in `setting.json` are regarded as normal data.  
The other digits are regarded as abnormal.

Images generated from optimized latent vector **z** and difference with original images are saved.

```bash
$ python detect.py [options]
```

`-h` option can show help.

```bash
$ python detect.py -h
usage: detect.py [-h] -s SETTING --gen GEN --dis DIS -r RESULT [-g GPU]
                 [-i ITERATION] [-n NUM] [--lam LAM]

Performs anomaly detection

optional arguments:
  -h, --help            show this help message and exit
  -s SETTING, --setting SETTING
                        setting file used for training GAN
  --gen GEN             Generator model file trained by train_gan.py
  --dis DIS             Discriminator model trained by train_gan.py
  -r RESULT, --result RESULT
                        result directory
  -g GPU, --gpu GPU     GPU
  -i ITERATION, --iteration ITERATION
                        iteration number of optimizing z
  -n NUM, --num NUM     number of images
  --lam LAM             balance between residual loss and feature matching
                        loss
```
