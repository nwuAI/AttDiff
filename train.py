import os
import gc
import ssl
import yaml
import wandb
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from tqdm.auto import tqdm
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats
import argparse
import clip
import lpips

# Diffusers library
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

# Local imports
from src.pix2pix_turbo import Pix2Pix_Turbo
from src.utils import misc
from src.utils.loss import StyleLoss
from src.utils.datasets import Dataset
import src.utils.misc as misc


def main(args):
    accelerator = Accelerator(
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    mixed_precision=args.mixed_precision, log_with=args.report_to,)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "pretrained"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    if args.pretrained_model_name_or_path == '/data/hn/code/AttrDiff/sd-turbo_net':
        net_pix2pix = Pix2Pix_Turbo(
                        lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae,
                        pretrained_model_name_or_path=args.pretrained_model_name_or_path, pretrained_path=args.model_url)
        net_pix2pix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pix2pix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gan_disc_type == "vagan_clip":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")

    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    net_clip, _ = clip.load("ViT-B/32", device="cuda")
    net_clip.requires_grad_(False)
    net_clip.eval()

    net_style = StyleLoss().cuda()
    net_style.requires_grad_(False)


    # make the optimizer
    layers_to_opt = []
    for n, _p in net_pix2pix.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_pix2pix.unet.conv_in.parameters())

    for n, _p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)

    for n, _p in net_pix2pix.fusionblock.named_parameters():
        assert _p.requires_grad
        layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    optimizer_disc = torch.optim.AdamW(
        net_disc.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    dataset_train = Dataset(
        img_size=args.resolution,
        tokenizer=net_pix2pix.tokenizer,
        dataset_folder=args.train_dataset_folder,
        attr=args.attr,
        attr_random_rate=args.attr_random_rate,
        mask_folder=args.mask_folder,
        training=True
    )
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers
    )

    dataset_val = Dataset(
        img_size=args.resolution,
        tokenizer=net_pix2pix.tokenizer,
        dataset_folder=args.train_dataset_folder,
        attr=args.attr,
        attr_random_rate=args.attr_random_rate,
        mask_folder=args.mask_folder,
        valing=True
    )
    dl_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # Prepare everything with our `accelerator`.
    net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    net_clip, net_lpips = accelerator.prepare(net_clip, net_lpips)
    net_lpips = accelerator.prepare(net_lpips)

    # renorm with image net statistics
    t_clip_renorm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networks to device and cast to weight_dtype
    net_pix2pix.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_clip.to(accelerator.device, dtype=weight_dtype)
    net_style.to(accelerator.device, dtype=weight_dtype)


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
                        disable=not accelerator.is_local_main_process, )

    # turn off eff. attn for the discriminator
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    # compute the reference stats for FID tracking
    if accelerator.is_main_process and args.track_val_fid:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)

        def fn_transform(x):
            x_pil = Image.fromarray(x)
            out_pil = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS)(x_pil)
            return np.array(out_pil)

        ref_stats = get_folder_features(
                        args.val_dataset_folder,
                        model=feat_model, num_workers=0,
                        num=None,shuffle=False, seed=0,
                        batch_size=8, device=torch.device("cuda"),
                        mode="clean", custom_image_tranform=fn_transform,
                        description="", verbose=True)


    # start the training loop
    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_pix2pix, net_disc]
            with accelerator.accumulate(*l_acc):
                gt_keep_mask = batch["gt_keep_mask"]
                gt = batch["ground_truth"]
                mask = batch["mask"]

                B, C, H, W = gt_keep_mask.shape
                # forward pass
                x_tgt_pred, pred_image,fusion_image = net_pix2pix(gt_keep_mask, mask)

                # Reconstruction loss
                loss_l2 = F.mse_loss(x_tgt_pred.float(), gt.float(), reduction="mean") * args.lambda_l2
                loss_l2_pred = F.mse_loss(pred_image.float(), gt.float(), reduction="mean") * args.lambda_l2
                loss_l2_fusion = F.mse_loss(fusion_image.float(), gt.float(), reduction="mean") * args.lambda_l2

                loss_lpips = net_lpips(x_tgt_pred.float(), gt.float()).mean() * args.lambda_lpips
                loss_lpips_pred = net_lpips(pred_image.float(), gt.float()).mean() * args.lambda_lpips
                loss_lpips_fusion = net_lpips(fusion_image.float(), gt.float()).mean() * args.lambda_lpips

                loss_style = net_style(pred_image, gt) * args.lambda_style

                loss = loss_l2 + loss_lpips + loss_l2_pred + loss_lpips_pred + loss_style + loss_l2_fusion + loss_lpips_fusion
                # CLIP similarity loss
                if args.lambda_clipsim > 0:
                    x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred)
                    x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear",
                                                      align_corners=False)
                    caption_tokens = clip.tokenize(batch["caption"], truncate=True).to(x_tgt_pred.device)
                    clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                    loss_clipsim = (1 - clipsim.mean() / 100)
                    loss += loss_clipsim * args.lambda_clipsim
                accelerator.backward(loss, retain_graph=False)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                Generator loss: fool the discriminator
                """
                x_tgt_pred, pred_image, fusion_image = net_pix2pix(gt_keep_mask, mask)
                lossG = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan
                accelerator.backward(lossG)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                Discriminator loss: fake image vs real image
                """
                # real image
                lossD_real = net_disc(gt.detach(), for_real=True).mean() * args.lambda_gan
                accelerator.backward(lossD_real.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                # fake image
                lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(lossD_fake.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                lossD = lossD_real + lossD_fake

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["lossG"] = lossG.detach().item()
                    logs["lossD"] = lossD.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_l2_pred"] = loss_l2_pred.detach().item()
                    logs["loss_l2_fusion"] = loss_l2_fusion.detach().item()

                    logs["loss_lpips"] = loss_lpips.detach().item()
                    logs["loss_lpips_pred"] = loss_lpips_pred.detach().item()
                    logs["loss_lpips_fusion"] = loss_lpips_fusion.detach().item()

                    logs["loss_style"] = loss_style.detach().item()

                    if args.lambda_clipsim > 0:
                        logs["loss_clipsim"] = loss_clipsim.detach().item()
                    progress_bar.set_postfix(**logs)

                    # viz some images
                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/source": [wandb.Image(gt_keep_mask[idx].float().detach().cpu(), caption=f"idx={idx}")for idx in range(B)],
                            "train/target": [wandb.Image(gt[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/model_pred": [wandb.Image(pred_image[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/model_fusion": [wandb.Image(fusion_image[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/model_output": [wandb.Image(x_tgt_pred[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # checkpoint the sd-turbo_net
                    if global_step % args.checkpointing_steps == 0 or step % dl_train.total_dataset_length == 0:
                        outf = os.path.join(args.output_dir, "pretrained", f"model{epoch}-{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)

                    # compute validation set FID, L2, LPIPS, CLIP-SIM
                    if global_step % args.eval_freq == 1:
                        l_l2, l_l2_pred, l_l2_fusion, l_lpips, l_lpips_pred, l_lpips_fusion, l_style,l_clipsim = [], [], [], [], [], [], [],[]
                        if args.track_val_fid:
                            os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)

                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break

                            gt_keep_mask = batch_val["gt_keep_mask"].cuda()
                            gt = batch_val["ground_truth"].cuda()
                            mask = batch_val["mask"].cuda()
                            B, C, H, W = gt_keep_mask.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # forward pass
                                x_tgt_pred,pred_image,fusion_image = accelerator.unwrap_model(net_pix2pix)(gt_keep_mask, mask)
                                # compute the reconstruction losses
                                loss_l2 = F.mse_loss(x_tgt_pred.float(), gt.float(), reduction="mean")
                                loss_l2_pred = F.mse_loss(pred_image.float(), gt.float(), reduction="mean")
                                loss_l2_fusion = F.mse_loss(fusion_image.float(), gt.float(),reduction="mean")

                                loss_lpips = net_lpips(x_tgt_pred.float(), gt.float()).mean()
                                loss_lpips_pred = net_lpips(pred_image.float(), gt.float()).mean()
                                loss_lpips_fusion = net_lpips(fusion_image.float(), gt.float()).mean()

                                loss_style = net_style(pred_image, gt)

                                # compute clip similarity loss
                                x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred)
                                x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear",align_corners=False)
                                caption_tokens = clip.tokenize(batch_val["caption"], truncate=True).to(x_tgt_pred.device)
                                clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                                clipsim = clipsim.mean()

                                l_l2.append(loss_l2.item())
                                l_l2_pred.append(loss_l2_pred.item())
                                l_l2_fusion.append(loss_l2_fusion.item())

                                l_lpips.append(loss_lpips.item())
                                l_lpips_pred.append(loss_lpips_pred.item())
                                l_lpips_fusion.append(loss_lpips_fusion.item())

                                l_style.append(loss_style.item())
                                l_clipsim.append(clipsim.item())


                            # save outputs images to file for FID evaluation
                            if args.track_val_fid:
                                output_pil = transforms.ToPILImage()(x_tgt_pred[0].cpu())
                                outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"val_{step}.png")
                                output_pil.save(outf)

                        if args.track_val_fid:
                            curr_stats = get_folder_features(
                                os.path.join(args.output_dir, "eval", f"fid_{global_step}"), model=feat_model,
                                num_workers=0, num=None,
                                shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                                mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)
                            fid_score = fid_from_feats(ref_stats, curr_stats)
                            logs["val/clean_fid"] = fid_score
                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/l2_pred"] = np.mean(l_l2_pred)
                        logs["val/l2_fusion"] = np.mean(l_l2_fusion)

                        logs["val/lpips"] = np.mean(l_lpips)
                        logs["val/lpips_pred"] = np.mean(l_lpips_pred)
                        logs["val/lpips_fusion"] = np.mean(l_lpips_fusion)

                        logs["val/style"] = np.mean(l_style)
                        logs["val/clipsim"] = np.mean(l_clipsim)
                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/celebahq.yml')

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)

    args = parser.parse_args()
    config = misc.get_config(args.config)
    main(config)