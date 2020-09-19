import os
from glob import glob
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
from modeling.deeplab import DeepLab



def load_model(args):
    model = DeepLab(num_classes=args.nclass,
                    backbone=args.backbone,
                    output_stride=args.out_stride)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def main(args):
    model = load_model(args)
    X1 = torch.rand((1, 3) + eval(args.size))
    X2 = torch.rand((1, args.nclass) + eval(args.size))
    inputs = {'forward' : X1, 'decode' : X2}
    traced_script_module = torch.jit.trace_module(model, inputs)
    traced_script_module.save(args.model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus export for inference")
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--nclass', type=int, default=21,
                        help='number of classes(incluce background class)')
    parser.add_argument('--checkpoint', type=str, default=r"C:\Users\CUONG\Desktop\model_best.pth.tar",
                        help='Checkpoint file path')
    parser.add_argument('--model_path', type=str, default=r"C:\Users\CUONG\Desktop\model.pt",
                        help='Model file path')
    parser.add_argument('--size', type=str, default='(513, 513)',
                        help='Input size')

    args = parser.parse_args()
    main(args)