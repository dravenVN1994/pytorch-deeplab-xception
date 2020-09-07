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

def get_mask(predict):
    predict[predict > 0] = 255
    predict = np.stack([predict, np.zeros_like(predict), np.zeros_like(predict)], axis=-1)
    predict = predict.astype(np.uint8)
    return predict

def blend_mask(image, mask):
    blend = cv2.addWeighted(image, 0.8, mask, 0.2, 0.0)
    return blend

def inference_file(model, args, image_path):
    img_origin = cv2.imread(image_path)
    img = Image.fromarray(img_origin)
    img = img.resize(eval(args.size), Image.BILINEAR)
    img = np.array(img).astype(np.float32)
    img /= 255.0
    img -= (0.485, 0.456, 0.406)
    img /= (0.229, 0.224, 0.225)
    img = np.array(img).astype(np.float32).transpose((2, 0, 1))
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    output = model(img)
    output = output[0]
    predict = torch.argmax(output, dim=0)
    predict = predict.cpu().numpy()
    mask = get_mask(predict)
    img_origin = cv2.resize(img_origin, predict.shape[: 2], cv2.INTER_LINEAR)
    blended = blend_mask(img_origin, mask)
    return blended

def inference_dir(model, args):
    image_files = glob(os.path.join(args.dir_path, '*.*'))
    if args.max_file:
        image_files = image_files[: args.max_file]
    for image_file in image_files:
        print("Predicting: ", image_file)
        blended = inference_file(model, args, image_file)
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(image_file)), blended)
    
def main(args):
    model = load_model(args)
    inference_dir(model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Inference")
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--nclass', type=int, default=21,
                        help='number of classes(incluce background class)')
    parser.add_argument('--checkpoint', type=str, default=r"C:\Users\CUONG\Desktop\model_best.pth.tar",
                        help='Checkpoint file path')
    parser.add_argument('--dir_path', type=str, default=None,
                        help='Path to input images')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to output dir')
    parser.add_argument('--max_file', type=int, default=10,
                        help='Max number of files to predict')
    parser.add_argument('--size', type=str, default='(513, 513)',
                        help='Input size')

    args = parser.parse_args()
    main(args)


