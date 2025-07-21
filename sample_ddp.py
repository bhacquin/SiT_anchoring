# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train import create_logger
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import sys
import wandb_utils
from datetime import timedelta
import logging
import wandb



def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(mode, args):
    """
    Run sampling.
    """
    # Variables d'environnement SLURM
    rank = int(os.environ.get("RANK", int(os.environ.get("SLURM_PROCID", 0))))
    local_rank = int(os.environ.get("LOCAL_RANK", int(os.environ.get("SLURM_LOCALID", 0))))
    world_size = int(os.environ.get("WORLD_SIZE", int(os.environ.get("SLURM_NTASKS", 1))))
    
    print(f"🔧 [Process {rank}] Initialisation - Local rank: {local_rank}, World size: {world_size}")
    
    # CRITIQUE: Vérifier et configurer le device AVANT l'init DDP
    device_count = torch.cuda.device_count()
    print(f"🔧 [Process {rank}] Devices disponibles: {device_count}")
    
    if local_rank >= device_count:
        raise RuntimeError(f"Local rank {local_rank} >= device count {device_count}")
    
    # Configurer le device AVANT toute autre opération CUDA
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"🔧 [Process {rank}] Device configuré: {device}")
    
    # Test device
    try:
        test_tensor = torch.randn(10, device=device)
        print(f"✅ [Process {rank}] Device test réussi: {test_tensor.device}")
    except Exception as e:
        print(f"❌ [Process {rank}] Device test échoué: {e}")
        raise

    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    torch.set_grad_enabled(False)

    # Setup DDP:
    # dist.init_process_group("nccl")
    # Initialiser le process group seulement si nécessaire
    if world_size > 1:
        if not dist.is_initialized():
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = os.environ.get("MASTER_PORT", "12355")
            
            print(f"🔧 [Process {rank}] Init process group: {master_addr}:{master_port}")
            
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=7200)  # 2h timeout
            )
            
            print(f"✅ [Process {rank}] Process group initialisé")

    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"🌱 Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    if rank == 0:
        logger = create_logger("/capstor/scratch/cscs/vbastien/SiT_anchoring", level=logging.INFO)
    else:
        logger = create_logger(None)

    if args.ckpt is not None:
        ckpt_path = args.ckpt
        try:
            checkpoint = torch.load(ckpt_path, weights_only=True, map_location='cpu')
            if rank == 0:
                if logger:
                    logger.info("✅ Checkpoint loaded with weights_only=True")
        except Exception as e:
            if rank == 0 and logger:
                logger.warning(f"⚠️ Failed to load with weights_only=True: {str(e)}")
                logger.warning("Falling back to weights_only=False. Ensure checkpoint is trusted.")
            checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cpu')
        
        cfg = checkpoint["cfg"]  # Load the config from the checkpoint
    
    if args.ckpt is None:
        assert args.model == "SiT-XL/2", "Only SiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
        assert args.image_size == 256, "512x512 models are not yet available for auto-download." # remove this line when 512x512 models are available
        learn_sigma = args.image_size == 256
    else:
        learn_sigma = False
    print("Time", cfg.get('use_time', True), "contrastive", getattr(cfg, 'use_contrastive', False), "batch_size", getattr(cfg, 'global_batch_size', 256))
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=cfg.get('num_classes', 1000),
        use_time=cfg.get('use_time', True),  # Use time embeddings if specified
        encoder_depth=cfg['encoder_depth'],
        use_projectors=cfg['use_projectors'],
        z_dims=cfg['z_dims'],
        learn_sigma=cfg.get('learn_sigma', True)
    )
    model = model.to(device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()  # Important to set the model to evaluation mode
    """
    Initialisation de wandb (seulement sur le processus principal)
    """
    if rank == 0:
        try:
            if cfg['wandb_api_key'] is not None:
                os.environ["WANDB_API_KEY"] = cfg['wandb_api_key']
            else:
                print(f"⚠️ No Wandb api key.")
                os.environ["WANDB_API_KEY"] = '7054f94a0dfd9c1584b29282bae968073f5139f7'

            project_name = "Sampling SiT" 
            run_name = getattr(cfg, 'wandb_run_name', getattr(cfg, 'run_name', 'ddp-native-test'))
            
            wandb.init(project=project_name, name=run_name, config=dict(cfg))
            if logger:
                logger.info(f"✅ Wandb initialisé - Projet: {project_name}, Run: {run_name}")
            print(f"✅ Wandb initialisé - Projet: {project_name}, Run: {run_name}")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Erreur lors de l'initialisation de Wandb: {e}")
            print(f"⚠️ Erreur lors de l'initialisation de Wandb: {e}")
            print("🔄 Continuer sans Wandb")
    # requires_grad(model, False)
    model.eval()  # important!
    
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)
    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse
            )
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    if mode == "ODE":
        folder_name = f"{model_string_name}-{ckpt_string_name}-" \
                  f"cfg-{args.cfg_scale}-{args.per_proc_batch_size}-"\
                  f"{mode}-{args.num_sampling_steps}-{args.sampling_method}"
    elif mode == "SDE":
        folder_name = f"{model_string_name}-{ckpt_string_name}-" \
                    f"cfg-{args.cfg_scale}-{args.per_proc_batch_size}-"\
                    f"{mode}-{args.num_sampling_steps}-{args.sampling_method}-"\
                    f"{args.diffusion_form}-{args.last_step}-{args.last_step_size}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    num_samples = len([name for name in os.listdir(sample_folder_dir) if (os.path.isfile(os.path.join(sample_folder_dir, name)) and ".png" in name)])
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    done_iterations = int( int(num_samples // dist.get_world_size()) // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    
    for i in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        if args.unsupervised:
            y = torch.tensor([1000] * n, device=device)
        else:
            y = torch.randint(0, args.num_classes, (n,), device=device)
        
        # Setup classifier-free guidance:
        if using_cfg and not args.unsupervised:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            model_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            model_fn = model.forward

        samples = sample_fn(z, model_fn, **model_kwargs)[-1]
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample

        if rank==0 and i <= 2:
            wandb_utils.log_image(samples)

        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size
        dist.barrier()

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"

    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-B/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="Samples/SiT-B/TimeConditioned/Contrastive_256_epoch49/")
    parser.add_argument("--per-proc-batch-size", type=int, default=256)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--unsupervised", action='store_true')
    parser.add_argument("--cfg-scale",  type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default="/capstor/scratch/cscs/vbastien/SiT_anchoring/outputs/SiT-B/2/JEPAlike_False/Time_Cond_True/2025-07-16/Contrast_True__DivFalse_L2_False/checkpoints/epoch_finished_49_step0250200.pt",
                        help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE

    args = parser.parse_known_args()[0]
    main(mode, args)
