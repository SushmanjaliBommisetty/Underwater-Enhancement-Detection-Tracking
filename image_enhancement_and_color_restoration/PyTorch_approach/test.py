'''
Author:Xuelei Chen(chenxuelei@hotmail.com)
Usgae:
python test.py --checkpoint CHECKPOINTS_PATH
'''
import os
import torch
import numpy as np
from PIL import Image
from model import PhysicalNN
import argparse
from torchvision import transforms
import datetime
import math


def main(checkpoint, imgs_path, result_path):

    ori_dirs = []
    for image in os.listdir(imgs_path):
        ori_dirs.append(os.path.join(imgs_path, image))

    # Check for GPU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = PhysicalNN()
    model = torch.nn.DataParallel(model).to(device)
    print("=> loading trained model")
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(checkpoint['epoch']))
    model = model.module
    model.eval()

    testtransform = transforms.Compose([
                transforms.ToTensor(),
            ])
    unloader = transforms.ToPILImage()

    starttime = datetime.datetime.now()
    for imgdir in ori_dirs:
        img_name = (imgdir.split('/')[-1]).split('.')[0]
        img = Image.open(imgdir)
        inp = testtransform(img).unsqueeze(0)
        inp = inp.to(device)
        out = model(inp)

        corrected = unloader(out.cpu().squeeze(0))
        dir = '{}/results_{}'.format(result_path, checkpoint['epoch'])
        if not os.path.exists(dir):
            os.makedirs(dir)
        corrected.save(dir+'/{}corrected.png'.format(img_name))

        # Save also to D:/project/dataset/restored_frames
        restored_dir = r'D:/project/dataset/restored_frames'
        os.makedirs(restored_dir, exist_ok=True)
        corrected.save(os.path.join(restored_dir, f'{img_name}_corrected.png'))
    endtime = datetime.datetime.now()
    print(endtime-starttime)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='checkpoints/model_best_2842.pth.tar', required=True)
    parser.add_argument(
            '--images', help='test images folder', default='./test_img/')
    parser.add_argument(
            '--result', help='results folder', default='./results/')
    args = parser.parse_args()
    
    project_root = r"D:/project/image_enhancement_and_color_restoration"

    checkpoint_path_absolute = os.path.join(project_root, "PyTorch_approach", args.checkpoint.replace("./", ""))
    images_path_absolute = os.path.join(project_root, "PyTorch_approach", args.images.replace("./", ""))
    results_base_path_absolute = os.path.join(project_root, "PyTorch_approach", args.result.replace("./", ""))
    
    os.makedirs(results_base_path_absolute, exist_ok=True)
    print(f"Ensuring output base directory exists: {results_base_path_absolute}")

    main(checkpoint=checkpoint_path_absolute, imgs_path=images_path_absolute, result_path=results_base_path_absolute)

