import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from src.pix2pix_turbo import Pix2Pix_Turbo
import torch.nn.functional as F


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--dataset_folder", default='/data/hn/DataSets/CelebAMask-HQ/v', type=str)
    parser.add_argument("--mask_folder", default="/data/hn/DataSets/mask/liu-mask/t", type=str)
    parser.add_argument("--attr_random_rate", default=0.4, type=int)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--pretrained_model_name_or_path",default='/data/hn/code/img2img-turbo-liumask/sd-turbo_net')

    parser.add_argument('--model_path', type=str,default='./output/pretrained/celebahq.pkl', help='path to a sd-turbo_net state dict to be used')
    parser.add_argument('--output_dir', type=str, default='./', help='the directory to save the outputs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    args = parser.parse_args()

    transform = transforms.ToTensor()
    # # initialize the sd-turbo_net
    model = Pix2Pix_Turbo(
                   pretrained_path=args.model_path,
                    pretrained_model_name_or_path=args.pretrained_model_name_or_path,)
    model.set_eval()
    captions = ['This woman feel anxious','This guy does not wear glasses','This guy is angry', 'This woman seems very happy',]

    idx = 0

    for filename in os.listdir(args.dataset_folder):
        if filename.endswith(('.png', '.jpg', 'jpeg')):
            gt = Image.open(os.path.join(args.dataset_folder, filename))
            mask = Image.open(os.path.join(args.mask_folder, filename))
            gt = transform(gt).unsqueeze(0)
            mask = transform(mask).unsqueeze(0)

            gt = F.interpolate(gt, size=(256,256), mode='bilinear', align_corners=False)
            mask = F.interpolate(mask, size=(256,256), mode='bilinear', align_corners=False)
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1

            mask = 1 - mask
            gt_keep_mask = gt * mask

            print(filename+": ",captions[idx])
            output_image_batch,pred_image,fusion_image = model(gt_keep_mask.cuda(),mask=mask.cuda(), prompt=captions[idx])

            output_image = output_image_batch[0]
            pred_image = pred_image[0]
            fusion_image = fusion_image[0]

            output_image = transforms.ToPILImage()(output_image.cpu())
            gt_mask = transforms.ToPILImage()(gt_keep_mask[0].cpu())
            pred_image = transforms.ToPILImage()(pred_image.cpu())
            fusion_image = transforms.ToPILImage()(fusion_image.cpu())

            # save the outputs image
            # os.makedirs(args.output_dir+'/pred', exist_ok=True)
            os.makedirs(args.output_dir+'/gt_mask', exist_ok=True)
            os.makedirs(args.output_dir+'/output', exist_ok=True)
            # os.makedirs(args.output_dir+'/fusion', exist_ok=True)

            # pred_image.save(os.path.join(args.output_dir,'pred', filename[idx]))
            pred_image.save(os.path.join(args.output_dir,'output', filename))
            gt_mask.save(os.path.join(args.output_dir,'gt_mask', filename))
            # fusion_image.save(os.path.join(args.output_dir,'fusion', filename[idx]))
            idx = idx + 1


