# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import logging
import os
from datetime import timedelta

import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset

from utils import get_config_info, get_layer_by_name
from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from train_utils import init_wandb, get_layer_output_dim, create_scheduler, get_layer_by_name
from diffusers.models import AutoencoderKL
from losses import info_nce_loss
import wandb_utils


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def setup_distributed():
    """
    Configuration de l'environnement distribué pour PyTorch DDP
    """
    # Variables d'environnement définies par torchrun
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "12355")
    
    # Définir le device local AVANT l'initialisation du processus group
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Initialiser le processus group 
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=60)
        )
        # Ne pas faire de barrier ici - cela cause des conflits de GPU mapping
    
    return rank, local_rank, world_size, device


def cleanup():
    """
    End DDP training.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################
from utils import get_config_info
# Get config path and name from command line or environment variables
config_dir, config_name = get_config_info()

@hydra.main(version_base=None, config_path=config_dir, config_name=config_name)
def main(cfg: DictConfig):
    """
    Trains a new SiT model using Hydra configuration.
    """
    # assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # rank, local_rank, world_size, device = setup_distributed()
    # # Setup DDP:
    # dist.init_process_group("nccl")
    # assert cfg.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # assert rank == dist.get_rank()
    # rank = dist.get_rank()
    # device = rank % torch.cuda.device_count()
    # seed = cfg.global_seed * dist.get_world_size() + rank
    # torch.manual_seed(seed)
    # torch.cuda.set_device(device)
    # print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    # local_batch_size = int(cfg.global_batch_size // dist.get_world_size())
    # ========== INITIALISATION DDP CORRIGÉE ==========
    
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
    
    # Seed configuration
    seed = cfg.global_seed * world_size + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    print(f"🌱 [Process {rank}] Seed configuré: {seed}")
    
    # Vérifications de cohérence
    local_batch_size = int(cfg.global_batch_size // world_size)
    assert cfg.global_batch_size % world_size == 0, f"Batch size must be divisible by world size."
    
    print(f"📊 [Process {rank}] Batch size local: {local_batch_size}")
    # Setup an experiment folder using Hydra:
    if rank == 0:
        # Hydra automatically creates the output directory
        experiment_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        # if cfg.wandb:
        #     entity = os.environ.get("ENTITY", "default_entity")
        #     project = os.environ.get("PROJECT", "SiT_training")
        #     experiment_name = os.path.basename(experiment_dir)
        #     wandb_utils.initialize(cfg, entity, experiment_name, project)
    else:
        logger = create_logger(None)

    # Initialisation de wandb
    init_wandb(cfg, rank, logger)

    # Create model:
    assert cfg.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = cfg.image_size // 8
    model = SiT_models[cfg.model](
        input_size=latent_size,
        num_classes=cfg.num_classes
    )

    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    model.to(device)
    if cfg.ckpt is not None:
        ckpt_path = cfg.ckpt
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        # Note: optimizer will be created after this, so we'll load its state later
        # cfg will be used as is (no overriding from checkpoint)

    requires_grad(ema, False)
    
    model = DDP(model, device_ids=[local_rank])
    transport = create_transport(
        cfg.path_type,
        cfg.prediction,
        cfg.loss_weight,
        cfg.train_eps,
        cfg.sample_eps
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{cfg.vae}").to(device)
    print(f"🤖 [Process {rank}] Modèle créé et configuré")
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0)
    

    # Setup mixed precision
    mixed_precision = getattr(cfg, 'mixed_precision', 'fp32')
    if mixed_precision == "fp16":
        scaler = GradScaler()
        autocast_dtype = torch.float16
        use_amp = True
        if rank == 0:
            logger.info("🔧 Using mixed precision: fp16 with GradScaler")
    elif mixed_precision == "bf16":
        scaler = None  # bf16 doesn't need gradient scaling
        autocast_dtype = torch.bfloat16
        use_amp = True
        if rank == 0:
            logger.info("🔧 Using mixed precision: bf16")
    else:  # fp32 or any other value
        scaler = None
        autocast_dtype = None
        use_amp = False
        if rank == 0:
            logger.info("🔧 Using full precision: fp32")

    # Setup contrastive loss si activé
    contrastive_loss_fn = None
    activation_feature_dim = None
    layer_outputs = []
    hook_handle = None
    use_contrastive = getattr(cfg, 'use_contrastive_loss', False)   
    def hook_fn(module, input, output):
        layer_outputs.append(output)
    # print("[DEBUG] cfg.use_contrastive_loss", use_contrastive)
    if use_contrastive:
        # Configurer le hook pour capturer les activations
        target_layer_name = getattr(cfg, 'contrastive_layer', 'blocks.8')  # Couche du milieu par défaut
        target_layer = get_layer_by_name(model.module, target_layer_name)
        hook_handle = target_layer.register_forward_hook(hook_fn)
        
        # Déterminer automatiquement la dimension des activations
        if rank == 0:
            logger.info(f"🔍 Determining activation dimension for layer: {target_layer_name}")
        
        activation_feature_dim = get_layer_output_dim(
            model.module, 
            target_layer_name, 
            input_shape=(1, 4, latent_size, latent_size)
        )
        
        if rank == 0:
            logger.info(f"✅ Detected activation dimension: {activation_feature_dim}")
        
        # Configurer la perte contrastive
        # contrastive_loss_fn = SimpleInfoNCE(
        #     temperature=getattr(cfg, 'contrastive_temperature', 0.5)
        # )
        
        if rank == 0:
            logger.info(f"🔥 Contrastive loss enabled on layer: {target_layer_name}")
            logger.info(f"   - Activation dimension: {activation_feature_dim}")
            logger.info(f"   - Temperature: {getattr(cfg, 'contrastive_temperature', 0.5)}")
            logger.info(f"   - Num noisy versions: {getattr(cfg, 'contrastive_num_noisy_versions', 1)}")
            logger.info(f"   - Include clean as positive: {getattr(cfg, 'contrastive_include_clean', True)}")
            logger.info(f"   - Contrastive weight: {getattr(cfg, 'contrastive_weight', 0.1)}")

    # Load checkpoint states if resuming
    if cfg.ckpt is not None:
        ckpt_path = cfg.ckpt
        # Load optimizer state
        if "opt" in checkpoint:
            opt.load_state_dict(checkpoint["opt"])
            if rank == 0:
                logger.info("Loaded optimizer state from checkpoint")
                
        # Load scaler state if it exists and we're using fp16
        if scaler is not None and "scaler" in checkpoint and checkpoint["scaler"] is not None:
            scaler.load_state_dict(checkpoint["scaler"])
            if rank == 0:
                logger.info("Loaded GradScaler state from checkpoint")

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, cfg.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    # Check if data_path is provided, otherwise use HuggingFace datasets
    if not hasattr(cfg, 'data_path') or cfg.data_path is None or cfg.data_path == "":
        logger.warning("data_path not provided or empty. Using HuggingFace datasets as fallback.")
        logger.info(f"Loading dataset: {cfg.dataset_name} (split: {cfg.dataset_split})")
        
        hf_dataset = load_dataset(
            cfg.dataset_name,
            split=cfg.dataset_split,
            streaming=False,
            cache_dir=cfg.get("cache_dir", None),
            trust_remote_code=True
        )
        
        # Convert HuggingFace dataset to ImageFolder-like format
        # This assumes the dataset has 'image' and 'label' fields
        class HFDatasetWrapper(torch.utils.data.Dataset):
            def __init__(self, hf_dataset, transform=None):
                self.hf_dataset = hf_dataset
                self.transform = transform
                
            def __len__(self):
                return len(self.hf_dataset)
                
            def __getitem__(self, idx):
                item = self.hf_dataset[idx]
                image = item['image']
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                label = item['label']
                
                if self.transform:
                    image = self.transform(image)
                    
                return image, label
        
        dataset = HFDatasetWrapper(hf_dataset, transform=transform)
        logger.info(f"Loaded HuggingFace dataset with {len(dataset):,} images")
    else:
        # Use local ImageFolder dataset
        dataset = ImageFolder(cfg.data_path, transform=transform)
        logger.info(f"Dataset contains {len(dataset):,} images ({cfg.data_path})")
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=cfg.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Setup scheduler
    total_steps = len(loader) * cfg.epochs
    lr_scheduler, scheduler_update_mode = create_scheduler(opt, cfg, total_steps)
    if rank == 0 and lr_scheduler:
        logger.info(f"✅ Learning rate scheduler enabled (total steps: {total_steps})")
        if getattr(cfg, 'use_warmup', False):
            logger.info(f"   - Warmup for {cfg.warmup_steps} steps, from {cfg.warmup_init_lr} to {cfg.learning_rate}")
        if getattr(cfg, 'use_scheduler', False):
            logger.info(f"   - Main scheduler: {cfg.scheduler_type}")
    # Load checkpoint states if resuming
    if cfg.ckpt is not None:
        # ... (code de chargement du modèle, ema, opt) ...
        # Load scheduler state
        if lr_scheduler and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
            if rank == 0:
                logger.info("Loaded LR scheduler state from checkpoint")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Labels to condition the model with (feel free to change):
    ys = torch.randint(1000, size=(local_batch_size,), device=device)
    use_cfg = cfg.cfg_scale > 1.0
    # Create sampling noise:
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Setup classifier-free guidance:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1000] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=cfg.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    logger.info(f"Training for {cfg.epochs} epochs...")
    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            # VAE encoding (toujours en fp32)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                # VAE sempre en fp32 pour plus de fiabilité
                x_clean = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            # ========== PERTE DE DIFFUSION + CAPTURE D'ACTIVATIONS ==========
            model_kwargs = dict(y=y)
            
            # Clear the layer outputs before forward pass
            layer_outputs.clear()
            
            # Déterminer le nombre d'échantillons pour le contrastive
            k_samples = getattr(cfg, 'contrastive_num_noisy_versions', 1) if use_contrastive else 1
            # Forward pass avec capture d'activations
            if use_amp:
                with autocast(dtype=autocast_dtype):
                    loss_dict = transport.training_losses(
                        model, x_clean, model_kwargs, k=k_samples
                    )
                    diffusion_loss = loss_dict["loss"]
            else:
                loss_dict = transport.training_losses(
                    model, x_clean, model_kwargs, k=k_samples
                )
                diffusion_loss = loss_dict["loss"]
            
            # Copier les activations capturées
            # all_activations = layer_outputs.copy() if use_contrastive else []
            # print(f"[DEBUG] all_activations length: {len(all_activations)}, shape of first activation: {all_activations[0].shape}")
            # ========== PERTE CONTRASTIVE (si activée) ==========
            contrastive_loss = 0.0
            if use_contrastive and k_samples > 1:
                # Utiliser directement les activations capturées
                # all_activations[0] contient toutes les activations de forme (batch_size * k_samples, feature_dim)
                features = layer_outputs[0]
                layer_outputs.clear()
                batch_size = x_clean.size(0)
                # Vérification précoce des activations du modèle
                if not torch.isfinite(features).all():
                    if logger:
                        logger.error(f"Non-finite activations from model! NaN count: {torch.isnan(features).sum()}, Inf count: {torch.isinf(features).sum()}")
                        logger.error(f"Activation stats: min={features.min():.6f}, max={features.max():.6f}, mean={features.mean():.6f}")
                    # Nettoyer les activations
                    features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
                    # Optionnel: clamper pour éviter des valeurs extrêmes
                    features = torch.clamp(features, -10.0, 10.0)
                # Debug: vérifier les gradients avant la perte contrastive
                if cfg.get('debug_contrastive', False):
                    print(f"[DEBUG] features.requires_grad: {features.requires_grad}")
                    print(f"[DEBUG] features shape: {features.shape}")
                    print(f"[DEBUG] batch_size: {batch_size}, k_samples: {k_samples}")
                    print(f"[DEBUG] features is_leaf: {features.is_leaf}")
                    print(f"[DEBUG] features.grad_fn: {features.grad_fn}")
                
                # S'assurer que les features gardent les gradients
                if not features.requires_grad:
                    print(f"[WARNING] Features don't require grad! Enabling...")
                    features = features.requires_grad_(True)
                
                # Calculer la perte contrastive avec la nouvelle implémentation simple
                # contrastive_loss = contrastive_loss_fn(features, batch_size, k_samples)

                contrastive_loss, mean_pos_sim, mean_neg_sim = info_nce_loss(features, temperature = cfg.contrastive_temperature, logger = logger, 
                                                            sample_weights = None, use_divergent_only = cfg.use_divergent_only) 
                # Debug: vérifier la perte contrastive
                if cfg.get('debug_contrastive', False) and train_steps < 2:
                    print(f"[DEBUG] contrastive_loss: {contrastive_loss.item():.6f}")
                    print(f"[DEBUG] mean_pos_sim: {mean_pos_sim:.6f}, mean_neg_sim: {mean_neg_sim:.6f}")
                    print(f"[DEBUG] contrastive_loss.requires_grad: {contrastive_loss.requires_grad}")
                    print(f"[DEBUG] contrastive_loss.grad_fn: {contrastive_loss.grad_fn}")
                      
            # ========== COMBINAISON DES PERTES ==========
            contrastive_weight = getattr(cfg, 'contrastive_weight', 0.1)
            total_loss = diffusion_loss + contrastive_weight * contrastive_loss
                           
            # Vérification finale que total_loss est un scalaire
            if total_loss.numel() != 1:
                if rank == 0:
                    logger.warning(f"⚠️ total_loss is not scalar! Shape: {total_loss.shape}, taking mean...")
                total_loss = total_loss.mean()

            # ========== BACKWARD PASS ==========
            opt.zero_grad()
            
            # Backward pass with gradient scaling if using fp16
            if scaler is not None:  # fp16 case
                scaler.scale(total_loss).backward()
                scaler.step(opt)
                scaler.update()
            else:  # bf16 or fp32 case
                total_loss.backward()
                opt.step()

            # ========== SCHEDULER STEP ==========
            if lr_scheduler and scheduler_update_mode == "step":
                lr_scheduler.step()
                
            update_ema(ema, model.module)

            # ========== LOGGING ==========
            running_loss += diffusion_loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % cfg.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_diffusion_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_diffusion_loss, op=dist.ReduceOp.SUM)
                avg_diffusion_loss = avg_diffusion_loss.item() / dist.get_world_size()
                
                # LOG UNIQUEMENT DEPUIS RANK 0
                if rank == 0:
                    log_msg = f"(step={train_steps:07d}) Diffusion Loss: {avg_diffusion_loss:.4f}"
                    log_dict = {"train/diffusion_loss": avg_diffusion_loss, "train/steps_per_sec": steps_per_sec}
                    current_lr = opt.param_groups[0]['lr']
                    log_dict["train/lr"] = current_lr
                    if isinstance(contrastive_loss, torch.Tensor):
                        contrastive_val = contrastive_loss.item()
                        total_val = total_loss.item()
                        log_msg += f", Contrastive Loss: {contrastive_val:.4f}, Total Loss: {total_val:.4f}"
                        log_dict.update({
                            "train/contrastive_loss": contrastive_val,
                            "train/total_loss": total_val
                        })
                    log_msg += f", LR: {current_lr:.2e}"
                    logger.info(log_msg)
                    
                    if cfg.wandb:
                        wandb_utils.log(log_dict, step=train_steps)
                
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint:
            if train_steps % cfg.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "cfg": cfg,
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "scheduler": lr_scheduler.state_dict() if lr_scheduler else None # ✅ SAUVEGARDER LE SCHEDULER
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            
            if train_steps % cfg.sample_every == 0 and train_steps > 0:
                if rank == 0:
                    logger.info("Generating EMA samples...")
                
                # Nombre réduit d'images pour sampling (au lieu de global_batch_size)
                num_sample_images = getattr(cfg, 'num_sample_images', 8)
                
                # Créer un batch plus petit pour sampling
                sample_labels = torch.randint(0, cfg.num_classes, (num_sample_images,), device=device)
                use_cfg = cfg.cfg_scale > 1.0
                
                if use_cfg:
                    # Pour CFG, on double le batch (uncond + cond)
                    z_sample = torch.randn(num_sample_images * 2, 4, latent_size, latent_size, device=device)
                    y_null = torch.tensor([1000] * num_sample_images, device=device)
                    sample_labels_cfg = torch.cat([sample_labels, y_null], 0)
                    sample_model_kwargs_local = dict(y=sample_labels_cfg, cfg_scale=cfg.cfg_scale)
                    model_fn_local = ema.forward_with_cfg
                else:
                    z_sample = torch.randn(num_sample_images, 4, latent_size, latent_size, device=device)
                    sample_model_kwargs_local = dict(y=sample_labels)
                    model_fn_local = ema.forward
                
                # S'assurer que le modèle EMA est en mode évaluation
                ema.eval()
                
                with torch.no_grad():
                    # Sampling with mixed precision
                    sample_fn = transport_sampler.sample_ode() # default to ode sampling
                    if use_amp:
                        with autocast(dtype=autocast_dtype):
                            samples = sample_fn(z_sample, model_fn_local, **sample_model_kwargs_local)[-1]
                    else:
                        samples = sample_fn(z_sample, model_fn_local, **sample_model_kwargs_local)[-1]
                    
                    dist.barrier()

                    if use_cfg: # Remove null samples
                        samples, _ = samples.chunk(2, dim=0)
                    
                    # VAE decode sempre en fp32 pour plus de fiabilité
                    samples = vae.decode(samples / 0.18215).sample
                
                # Gather samples from all processes (mais maintenant beaucoup plus petit!)
                out_samples = torch.zeros((num_sample_images * dist.get_world_size(), 3, cfg.image_size, cfg.image_size), device=device)
                dist.all_gather_into_tensor(out_samples, samples)
                
                if rank == 0:
                    # Prendre seulement les premières images pour logging (éviter duplications)
                    log_samples = out_samples[:num_sample_images]
                    
                    if cfg.wandb:
                        wandb_utils.log_image(log_samples, train_steps)
                    logger.info(f"Generated and logged {num_sample_images} EMA samples.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    # Cleanup
    if hook_handle:
        hook_handle.remove()

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    main()
