import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import scipy.io as sio
import cv2
from models import SRCNN
import matplotlib.pyplot as plt
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr, Gaussnoise_func

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    #image = pil_image.open(args.image_file).convert('L')
    image = pil_image.open(args.image_file).convert('L')
    image = image.crop((0, 0, 300, 300))

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    # 卷积得到低分辨率图片
    # 读取H并保存为array
    matrix = sio.loadmat('sink2.mat')
    kernal = np.array(matrix["sink2"])
    imageAfterConv = np.dot(kernal, image)
    image = Gaussnoise_func(imageAfterConv)
    image = pil_image.fromarray(np.uint8(imageAfterConv))
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    #归一化，否则显示有问题
    image = np.array(image).astype(np.float64)
    image = image / np.max(image) * 255
    image = np.abs(image)
    image = pil_image.fromarray(np.uint8(image))

    #灰度化，否则保存有问题
    image.convert("L").show()
    image.save()
    sio.savemat('/Users/leeefn/PycharmProjects/imageAfterConv.mat', {'image': image})

    #image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    #image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)

    #image = np.array(image).astype(np.float32)

    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    psnr = calc_psnr(y, preds)
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output = output.convert("L")
    output.save('image_afterSrcnn_x{}.tif')
