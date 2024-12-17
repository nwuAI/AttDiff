import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats
from pytorch_fid import fid_score
from src.pix2pix_turbo import Pix2Pix_Turbo
from src.utils.datasets import Dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--dataset_folder", default='/data/hn/DataSets/CelebAMask-HQ', type=str)
    parser.add_argument("--attr", default="/data/hn/DataSets/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt", type=str)
    parser.add_argument("--mask_folder", default="/data/hn/DataSets/mask/liu-mask/testing_mask_dataset", type=str)
    parser.add_argument("--attr_random_rate", default=0.4, type=int)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--pretrained_model_name_or_path", default='/data/hn/code/img2img-turbo-liumask/sd-turbo_net')
    parser.add_argument('--model_path', type=str,default='./output/pretrained/celebahq.pkl', help='path to a sd-turbo_net state dict to be used')
    parser.add_argument('--output_dir', type=str, default='../outputs', help='the directory to save the outputs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_path == '':
        raise ValueError('Either model_name or model_path should be provided')

    # # initialize the sd-turbo_net
    model = Pix2Pix_Turbo(
                    pretrained_path=args.model_path, pretrained_model_name_or_path=args.pretrained_model_name_or_path,)
    model.set_eval()

    #val dataset
    dataset_val = Dataset(img_size=args.img_size,tokenizer=model.tokenizer, dataset_folder = args.dataset_folder,
                          attr=args.attr, attr_random_rate=args.attr_random_rate, mask_folder=args.mask_folder,valing=True)

    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # caption_filepath = "./captions1020"

    for step, batch_val in enumerate(dl_val):
        gt_keep_mask_batch = batch_val["gt_keep_mask"].cuda()
        gt_batch = batch_val["ground_truth"].cuda()
        mask_batch = batch_val["mask"].cuda()
        filename = batch_val["filename"]
        caption = batch_val['caption']

        print(step)
        output_image_batch,pred_image,fusion_image = model(gt_keep_mask_batch,mask_batch)

        for idx in range(args.batch_size):

            gt_keep_mask = gt_keep_mask_batch[idx]
            output_image = output_image_batch[idx]
            pred_image = pred_image[idx]
            fusion_image = fusion_image[idx]

            output_image = transforms.ToPILImage()(output_image.cpu())
            gt_mask = transforms.ToPILImage()(gt_keep_mask.cpu())
            pred_image = transforms.ToPILImage()(pred_image.cpu())
            fusion_image = transforms.ToPILImage()(fusion_image.cpu())

            # save the outputs image
            os.makedirs(args.output_dir+'/pred', exist_ok=True)
            os.makedirs(args.output_dir+'/gt_mask', exist_ok=True)
            os.makedirs(args.output_dir+'/output', exist_ok=True)
            os.makedirs(args.output_dir+'/fusion', exist_ok=True)

            pred_image.save(os.path.join(args.output_dir,'pred', filename[idx]))
            output_image.save(os.path.join(args.output_dir,'output', filename[idx]))
            gt_mask.save(os.path.join(args.output_dir,'gt_mask', filename[idx]))
            fusion_image.save(os.path.join(args.output_dir,'fusion', filename[idx]))

