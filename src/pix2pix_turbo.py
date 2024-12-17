import os

import numpy as np
import requests
import sys
import copy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd, FusionBlock


class TwinConv(torch.nn.Module):
    def __init__(self, convin_pretrained, convin_curr):
        super(TwinConv, self).__init__()
        self.conv_in_pretrained = copy.deepcopy(convin_pretrained)
        self.conv_in_curr = copy.deepcopy(convin_curr)
        self.r = None

    def forward(self, x):
        x1 = self.conv_in_pretrained(x).detach()
        x2 = self.conv_in_curr(x)
        return x1 * (1 - self.r) + x2 * (self.r)

class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(self, pretrained_path=None, pretrained_model_name_or_path=None, lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(pretrained_model_name_or_path)

        #mode from stablityai/sd-turbo
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)

        vae.decoder.ignore_skip = False
        vae.decoder.mask = None
        fusionblock = FusionBlock(3,3)

        unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])

            #-----------------------VAE----------------------#
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            self.target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                                       "to_k", "to_q", "to_v", "to_out.0", ]
            self.lora_rank_vae = lora_rank_vae

            #-----------------------Unet---------------------#
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)
            self.target_modules_unet = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
                                        "conv_shortcut", "conv_out","proj_in",
                                        "proj_out", "ff.net.2", "ff.net.0.proj"]
            self.lora_rank_unet = lora_rank_unet

            #-----------------------AFM---------------------#
            _sd_fusion = fusionblock.state_dict()
            for k in sd["stata_dict_fusionblock"]:
                _sd_fusion[k] = sd["stata_dict_fusionblock"][k]
            fusionblock.load_state_dict(_sd_fusion)

        elif pretrained_path is None:
            print("Initializing sd-turbo_net with random weights")
            #-----------------------VAE---------------------

            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                                  "to_k", "to_q", "to_v", "to_out.0"]
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

            #-----------------------Unet---------------------
            target_modules_unet = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
                                   "conv_shortcut", "conv_out", "proj_in",
                                   "proj_out", "ff.net.2", "ff.net.0.proj"]
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",target_modules=target_modules_unet)
            unet.add_adapter(unet_lora_config)

            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet

        self.unet, self.vae = unet.to("cuda"), vae.to("cuda")
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.vae.decoder.gamma = 1
        self.text_encoder.requires_grad_(False)
        # Timestep = 50
        self.timesteps = torch.tensor([50], device="cuda").long()
        self.fusionblock = fusionblock.to("cuda")

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.fusionblock.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.fusionblock.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.fusionblock.train()

        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        for n, _p in self.fusionblock.named_parameters():
             _p.requires_grad = True

    def forward(self, gt_keep_mask, mask, prompt=None, prompt_tokens=None):

        if prompt is None:
            prompt = [' '] * gt_keep_mask.size(0)
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"

        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        encoded_control = self.vae.encode(gt_keep_mask).latent_dist.sample() * self.vae.config.scaling_factor

        # add noise
        noise_in = torch.randn_like(encoded_control).cuda()
        noised_latent = self.noise_scheduler.add_noise(encoded_control, noise_in, self.timesteps)
        # downsampling mask
        mask_latent = F.interpolate(mask, size=(encoded_control.shape[2], encoded_control.shape[3]), mode='bilinear',align_corners=False)

        # For the hidden layer features, we use the mask to replace unreconstructed areas with features from the original image, adding noise based on the current step.
        # The unreconstructed areas are determined by the input image, while the remaining areas are left unchanged and reconstructed using noise calculated by the U-Net.
        encoded_control = noised_latent * (mask_latent) + encoded_control * (1-mask_latent)

        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc, ).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample

        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        pred_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        pred_image = pred_image.clamp(0, 1)

        fusion_image = self.fusionblock(gt_keep_mask, pred_image).clamp(0,1)
        output_image = gt_keep_mask * mask + fusion_image * (1 - mask)

        return output_image, pred_image, fusion_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        sd["stata_dict_fusionblock"] = {k: v for k, v in self.fusionblock.state_dict().items()}

        torch.save(sd, outf)