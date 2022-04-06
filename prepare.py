import argparse
import glob
import h5py
import numpy as np
import scipy.io as sio
import PIL.Image as pil_image
from utils import convert_rgb_to_y, Gaussnoise_func


def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
        # hr:原始图片 lr：卷积后的图片

        # 不使用.convert('RGB')进行转换的话，读出来的图像是RGBA四通道的，A通道为透明通道
        hr = pil_image.open(image_path).convert('RGB')
        # //：向下取整的整数除法
        # 保证图片大小是scale的整数倍
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale

        #卷积得到低分辨率图片
        # 读取H并保存为array
        matrix = sio.loadmat('sink2.mat')
        kernal = np.array(matrix["sink2"])
        imageAfterConv = np.dot(kernal, hr)
        imageAfterGauss = Gaussnoise_func(imageAfterConv)

        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        lr = pil_image.fromarray(imageAfterGauss).resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        #array和asarray都可以将数组转化为ndarray对象。
        #当参数为一般数组时，两个函数结果相同；当参数本身就是ndarray类型时，array会新建一个ndarray对象作为参数的副本，asarray不会新建与参数共享同一个内存
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)


        # 把原始图片截成14*14
        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    # dataset是类似于数组的数据集，而group是类似文件夹一样的容器，存放dataset和其他group
    # 创建给定形状和数据类型的空dataset,用现有的Numpy数组来初始化
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale

        # 卷积得到低分辨率图片
        # 读取H并保存为array
        matrix = sio.loadmat('sink2.mat')
        kernal = np.array(matrix["sink2"])
        imageAfterConv = np.dot(kernal, hr)
        imageAfterGauss = Gaussnoise_func(imageAfterConv)

        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        lr = pil_image.fromarray(imageAfterGauss).resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)

        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)
        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


if __name__ == '__main__':
    #声明一个parser，用于解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--stride', type=int, default=14)
    # upscale 放大系数
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--eval', action='store_true')
    #读取参数
    args = parser.parse_args()

    #train(args)

    eval(args)
