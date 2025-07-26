# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
import subprocess
import sys
import tempfile
from pathlib import Path
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import torch.nn.functional as F
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

import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
import signal
import os
from datetime import datetime, timedelta

from utils import get_config_info, get_layer_by_name, compute_entropy, visualize_pca_as_rgb, capture_intermediate_activations,run_pca_visualization_on_test_set
from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from train_utils import init_wandb, get_layer_output_dim, create_scheduler, get_layer_by_name
from diffusers.models import AutoencoderKL
from losses import dispersive_info_nce_loss, paired_info_nce_loss
import wandb_utils
import wandb
import matplotlib.pyplot as plt




#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def setup_timeout_signal_handler(checkpoint_dir, model, ema, opt, scaler, lr_scheduler, logger, cfg):
    """Configure un signal handler pour sauvegarder et relancer automatiquement"""
    
    def timeout_handler(signum, frame):
        import sys
        debug_file = f"{checkpoint_dir}/signal_debug.log"
        
        try:
            with open(debug_file, 'w') as f:
                f.write(f"Signal handler called: {signum}\n")
                f.write(f"Time: {datetime.now().isoformat()}\n")
                f.flush()
            
            sys.stderr.write(f"⏰ SIGNAL {signum} RECEIVED - Starting emergency save\n")
            sys.stderr.flush()
            
        except Exception as e:
            sys.stdout.write(f"Signal handler error: {e}\n")
            sys.stdout.flush()
        
        # ✅ RÉCUPÉRER LES INFOS WANDB AVANT DE SAUVEGARDER
        wandb_run_id = None
        wandb_run_name = None
        try:
            with open(debug_file, 'a') as f:
                f.write("Getting wandb info\n")
                f.flush()
            
            if cfg.get('wandb', False):
                import wandb
                if wandb.run is not None:
                    wandb_run_id = wandb.run.id
                    wandb_run_name = wandb.run.name
                    
                    with open(debug_file, 'a') as f:
                        f.write(f"✅ Wandb run found: {wandb_run_id} ({wandb_run_name})\n")
                        f.flush()
                    
                    # Mark preempting et finir proprement
                    try:
                        wandb.run.mark_preempting()
                        wandb.finish(quiet=True)
                        
                        with open(debug_file, 'a') as f:
                            f.write("✅ Wandb finished successfully\n")
                            f.flush()
                    except:
                        with open(debug_file, 'a') as f:
                            f.write("⚠️ Wandb finish failed, continuing\n")
                            f.flush()
                else:
                    with open(debug_file, 'a') as f:
                        f.write("❌ No active wandb run found\n")
                        f.flush()
            
        except Exception as e:
            with open(debug_file, 'a') as f:
                f.write(f"Wandb info retrieval failed: {e}\n")
                f.flush()
        
        # ✅ SAUVEGARDER LE CHECKPOINT AVEC LES INFOS WANDB
        try:
            with open(debug_file, 'a') as f:
                f.write("Starting checkpoint save\n")
                f.flush()
            
            current_train_steps = globals().get('train_steps', 0)
            emergency_path = f"{checkpoint_dir}/emergency_checkpoint_{current_train_steps}.pt"
            
            # ✅ INCLURE LES INFOS WANDB DANS LE CHECKPOINT
            checkpoint = {
                "model": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
                "train_steps": current_train_steps,
                "timestamp": datetime.now().isoformat(),
                "emergency_save": True,
                "wandb_run_id": wandb_run_id,  # ✅ AJOUTER ICI
                "wandb_run_name": wandb_run_name  # ✅ AJOUTER ICI
            }
            
            torch.save(checkpoint, emergency_path)
            
            with open(debug_file, 'a') as f:
                f.write(f"✅ Checkpoint saved: {emergency_path}\n")
                f.write(f"✅ Wandb info saved: {wandb_run_id}\n")
                f.flush()
            
            sys.stderr.write(f"💾 Emergency checkpoint saved: {emergency_path}\n")
            sys.stderr.flush()
            
        except Exception as e:
            with open(debug_file, 'a') as f:
                f.write(f"❌ Checkpoint save failed: {e}\n")
                f.flush()
            sys.stderr.write(f"❌ Checkpoint save failed: {e}\n")
            sys.stderr.flush()
        
        # ✅ RELANCER AVEC LES BONNES INFOS
        try:
            with open(debug_file, 'a') as f:
                f.write("Starting relaunch\n")
                f.flush()
            
            relaunch_training_simple(emergency_path, cfg, wandb_run_id, wandb_run_name, debug_file)
            
        except Exception as e:
            with open(debug_file, 'a') as f:
                f.write(f"Relaunch failed: {e}\n")
                f.flush()
            sys.stderr.write(f"❌ Relaunch failed: {e}\n")
            sys.stderr.flush()
        
        sys.stderr.write("🔄 Exiting process\n")
        sys.stderr.flush()
        os._exit(0)
    
    # Configurer les signaux
    signal.signal(signal.SIGTERM, timeout_handler)
    signal.signal(signal.SIGUSR1, timeout_handler)
    
    debug_file = f"{checkpoint_dir}/signal_debug.log"
    try:
        with open(debug_file, 'w') as f:
            f.write(f"Signal handler setup at: {datetime.now().isoformat()}\n")
            f.write(f"Checkpoint dir: {checkpoint_dir}\n")
            f.flush()
        print(f"✅ Signal handler configured, debug file: {debug_file}")
    except Exception as e:
        print(f"❌ Could not create debug file: {e}")

def relaunch_training_simple(emergency_checkpoint_path, original_cfg, wandb_run_id, wandb_run_name, debug_file):
    """Version simplifiée pour debug"""
    
    try:
        with open(debug_file, 'a') as f:
            f.write("Creating config file\n")
            f.flush()
        
        # ✅ CHARGER LE CHECKPOINT POUR RÉCUPÉRER LES INFOS WANDB
        checkpoint = torch.load(emergency_checkpoint_path, weights_only=False, map_location='cpu')
        
        # Utiliser les IDs passés en paramètre, sinon fallback sur ceux du checkpoint
        if not wandb_run_id:
            wandb_run_id = checkpoint.get('wandb_run_id', None)
            wandb_run_name = checkpoint.get('wandb_run_name', None)
        
        with open(debug_file, 'a') as f:
            f.write(f"Wandb run_id from params: {wandb_run_id}\n")
            f.write(f"Wandb run_name from params: {wandb_run_name}\n")
            f.flush()
        
        # Créer la config modifiée
        from omegaconf import OmegaConf
        # Convertir en dict puis recréer une DictConfig non-structurée
        config_dict = OmegaConf.to_container(original_cfg, resolve=True)
        new_cfg = OmegaConf.create(config_dict)
        
        # ✅ DÉSACTIVER LE MODE STRUCT POUR PERMETTRE L'AJOUT DE NOUVELLES CLÉS
        OmegaConf.set_struct(new_cfg, False)
        
        # Maintenant on peut ajouter les nouvelles clés
        new_cfg.ckpt = emergency_checkpoint_path
        
        # ✅ AJOUTER EXPLICITEMENT LES INFOS WANDB SI ELLES EXISTENT
        if wandb_run_id:
            new_cfg.wandb_resume_id = wandb_run_id
            new_cfg.wandb_resume_name = wandb_run_name
            
            with open(debug_file, 'a') as f:
                f.write(f"✅ Added wandb_resume_id to config: {wandb_run_id}\n")
                f.write(f"✅ Added wandb_resume_name to config: {wandb_run_name}\n")
                f.flush()
        else:
            with open(debug_file, 'a') as f:
                f.write("❌ No wandb run_id found, will create new run\n")
                f.flush()
        
        # Sauvegarder la config
        checkpoint_dir = Path(emergency_checkpoint_path).parent
        config_filename = f"resume_config_{Path(emergency_checkpoint_path).stem}.yaml"
        config_path = checkpoint_dir / config_filename
        
        with open(config_path, 'w') as f:
            OmegaConf.save(new_cfg, f)
        
        with open(debug_file, 'a') as f:
            f.write(f"Config saved: {config_path}\n")
            f.flush()
        
        # ✅ VÉRIFIER QUE LES CLÉS SONT BIEN DANS LE FICHIER
        try:
            # Relire le fichier pour vérifier
            reloaded_cfg = OmegaConf.load(config_path)
            if wandb_run_id:
                with open(debug_file, 'a') as f:
                    f.write(f"✅ Verification: wandb_resume_id in saved config = {getattr(reloaded_cfg, 'wandb_resume_id', 'NOT_FOUND')}\n")
                    f.write(f"✅ Verification: wandb_resume_name in saved config = {getattr(reloaded_cfg, 'wandb_resume_name', 'NOT_FOUND')}\n")
                    f.flush()
        except Exception as verify_e:
            with open(debug_file, 'a') as f:
                f.write(f"❌ Verification failed: {verify_e}\n")
                f.flush()
        
        # Paramètres SLURM
        num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        gpus_per_node = int(os.environ.get('SLURM_GPUS_ON_NODE', 4))
        
        with open(debug_file, 'a') as f:
            f.write(f"SLURM params: {num_nodes} nodes, {gpus_per_node} gpus\n")
            f.flush()
        
        # Commande sbatch
        cmd = [
            "sbatch",
            "--nodes", str(num_nodes),
            "--gpus-per-node", str(gpus_per_node),
            "--cpus-per-task", "32",  # ✅ Réduit pour éviter CPU binding
            "--ntasks-per-node", str(gpus_per_node),
            "--time", "12:00:00",
            "--job-name", "sit_anchoring_resume",
            "--account", "a144",
            "--exclude", "nid005687,nid005911,nid005364,nid005247,nid005227,nid005236,nid005462,nid005340",
            "--output", "logs/training_cluster_%j_%x_.out",
            "--error", "logs/training_cluster_%j_%x.err",
            "run_test.sh",
            f"--config-file={config_path}"
        ]
        
        with open(debug_file, 'a') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.flush()
        
        # Lancer la commande
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        with open(debug_file, 'a') as f:
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"Stdout: {result.stdout}\n")
            f.write(f"Stderr: {result.stderr}\n")
            f.flush()
        
    except Exception as e:
        with open(debug_file, 'a') as f:
            f.write(f"Relaunch exception: {e}\n")
            import traceback
            f.write(traceback.format_exc())
            f.flush()


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


def create_logger(logging_dir, level=logging.INFO):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=level,
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
#                                  Training MAIN                                #
#################################################################################
from utils import get_config_info
# Get config path and name from command line or environment variables
config_dir, config_name = get_config_info()

@hydra.main(version_base=None, config_path=config_dir, config_name=config_name)
def main(cfg: DictConfig):
    """
    Trains a new SiT model using Hydra configuration.
    """
    global train_steps
    train_steps = 0
    # ========== INITIALISATION DDP CORRIGÉE ==========
    if hasattr(cfg, 'wandb_resume_id'):
        print(f"📊 Found wandb_resume_id: {cfg.wandb_resume_id}")
        print(f"📊 Found wandb_resume_name: {cfg.wandb_resume_name}")
    else:
        print("📊 No wandb resume info found - will create new run")
    
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
        print(f"cfg.logging_level: {cfg.logging_level}")
        if cfg.logging_level == 'debug':
            logger = create_logger(experiment_dir, level=logging.DEBUG)
        elif cfg.logging_level == 'info':
            logger = create_logger(experiment_dir, level=logging.INFO)
        elif cfg.logging_level == 'warning':
            logger = create_logger(experiment_dir, level=logging.WARNING)
        elif cfg.logging_level == 'error':
            logger = create_logger(experiment_dir, level=logging.ERROR)
        else:
            logger = create_logger(experiment_dir, level=logging.INFO)
        
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    else:
        logger = create_logger(None)

    # Initialisation de wandb
    wandb_initialised = init_wandb(cfg, rank, logger)

    # Create model:
    assert cfg.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = cfg.image_size // 8

    encoder_depth = cfg.get('encoder_depth',[3])
    use_projectors = cfg.get('use_projectors', [True])
    z_dims = cfg.get('z_dims', [768])

    model = SiT_models[cfg.model](
                input_size=latent_size,
                num_classes=cfg.num_classes,
                use_time=cfg.get('use_time', True),  # Use time embeddings if specified
                encoder_depth=encoder_depth,
                use_projectors=use_projectors,
                z_dims=z_dims,
                learn_sigma=cfg.get('learn_sigma', True)
            )
    
    if logger:
        logger.info(f"Using time as a condition: {cfg.get('use_time', True) }")
    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    model.to(device)
    
    if cfg.ckpt is not None:
        ckpt_path = cfg.ckpt
        try:
            # Essayer d'abord avec weights_only=True (sécurisé)
            checkpoint = torch.load(ckpt_path, weights_only=True, map_location='cpu')
            if rank == 0:
                if logger:
                    logger.info("✅ Checkpoint loaded with weights_only=True")
        except Exception as e:
            # Si ça échoue (à cause de cfg), fallback vers weights_only=False
            if rank == 0 and logger:
                logger.warning(f"⚠️ Failed to load with weights_only=True: {str(e)}")
                logger.warning("Falling back to weights_only=False. Ensure checkpoint is trusted.")
            checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cpu')
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        # Note: optimizer will be created after this, so we'll load its state later

    requires_grad(ema, False)    
    model = DDP(model, device_ids=[local_rank])
    transport = create_transport(
        cfg.path_type,
        cfg.prediction,
        cfg.loss_weight,
        cfg.train_eps,
        cfg.sample_eps,
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
    use_contrastive_loss = getattr(cfg, 'use_contrastive_loss', False) 
    use_dispersive_loss = getattr(cfg, 'use_dispersive_loss', False)  

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

    # ✅ SETUP VALIDATION TRANSFORM (no random flip)
    test_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, cfg.image_size)),
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
        test_hf_dataset = load_dataset(
            cfg.dataset_name,
            split='train',
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
        test_dataset = HFDatasetWrapper(test_hf_dataset, transform=test_transform)
        logger.info(f"Loaded HuggingFace dataset with {len(dataset):,} images")
    else:
        # Use local ImageFolder dataset
        dataset = ImageFolder(cfg.data_path, transform=transform)
        logger.info(f"Dataset contains {len(dataset):,} images ({cfg.data_path})")
    # Train
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
    # Test
    # ✅ Test loader (only on rank 0 to save resources)
    test_loader = None
    test_iter = None
    if rank == 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=min(getattr(cfg, 'test_batch_size', 10), local_batch_size),  # Smaller batch for visualization
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True
        )
        test_iter = iter(test_loader)  # ✅ Initialize iterator
        logger.info(f"Test dataset loaded: {len(test_dataset):,} images")
    # Setup scheduler
    total_steps = len(loader) * cfg.epochs
    lr_scheduler, scheduler_update_mode = create_scheduler(opt, cfg, total_steps)
    if rank == 0 and lr_scheduler:
        logger.info(f"✅ Learning rate scheduler enabled (total steps: {total_steps})")
        if getattr(cfg, 'use_warmup', False):
            logger.info(f"   - Warmup for {cfg.warmup_steps} steps, from {cfg.warmup_init_lr} to {cfg.learning_rate}")
        if getattr(cfg, 'use_scheduler', False):
            logger.info(f"   - Main scheduler: {cfg.scheduler_type}")
    # Si on charge un checkpoint, récupérer le train_steps d'abord
    if cfg.ckpt is not None and "train_steps" in checkpoint:
        train_steps = checkpoint["train_steps"]
        if rank == 0 and logger:
            logger.info(f"✅ Resuming training from step {train_steps}")
    else:
        train_steps = 0

    # ✅ CALCULER L'EPOCH DE DÉPART
    steps_per_epoch = len(loader)
    start_epoch = train_steps // steps_per_epoch
    remaining_steps_in_epoch = train_steps % steps_per_epoch
    if rank == 0 and logger:
        logger.info(f"📊 Dataset info:")
        logger.info(f"   - Total samples: {len(dataset):,}")
        logger.info(f"   - Steps per epoch: {steps_per_epoch}")
        logger.info(f"   - Starting from epoch: {start_epoch}")
        logger.info(f"   - Remaining steps in current epoch: {remaining_steps_in_epoch}")

    # Avancer le scheduler au bon endroit si on resume
    if cfg.ckpt is not None and train_steps > 0:
        if lr_scheduler and scheduler_update_mode == "step":
            # Avancer le scheduler de train_steps
            for _ in range(train_steps):
                lr_scheduler.step()
            if rank == 0 and logger:
                logger.info(f"✅ Scheduler advanced to step {train_steps}")
        elif lr_scheduler:
            # Pour les schedulers basés sur epochs
            completed_epochs = train_steps // len(loader)
            for _ in range(completed_epochs):
                lr_scheduler.step()
            if rank == 0 and logger:
                logger.info(f"✅ Scheduler advanced to epoch {completed_epochs}")
       
    # Load checkpoint states if resuming
    if cfg.ckpt is not None:
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
    log_steps = 0
    running_loss = 0
    running_total_loss = 0  # ✅ NOUVELLE VARIABLE
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
    activate_handler = False
    logger.info(f"Training for {cfg.epochs} epochs...")
    if rank == 0 and activate_handler:
        # ✅ Vérifier si le code s'exécute dans un job SLURM
        is_on_slurm = "SLURM_JOB_ID" in os.environ
        if is_on_slurm:
            logger.info("✅ SLURM environment detected. Setting up timeout signal handler.")
            setup_timeout_signal_handler(
                checkpoint_dir=checkpoint_dir,
                model=model,
                ema=ema,
                opt=opt,
                scaler=scaler,
                lr_scheduler=lr_scheduler,
                logger=logger,
                cfg=cfg,
            )
        else:
            logger.info("⚪ Not running on SLURM. Timeout signal handler will not be set.")

#################################################################################
#                                  Training Loop                                #
#################################################################################
    for epoch in range(start_epoch, cfg.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        logger.info(f"Remaining steps in epoch {epoch}: {remaining_steps_in_epoch}")
        data_iter = iter(loader)

        try:
            while True:
                x, y = next(data_iter)
                x = x.to(device)
                y = y.to(device)             
                # VAE encoding (toujours en fp32)
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    x_clean = vae.encode(x).latent_dist.sample().mul_(0.18215)
                
                # ========== PERTE DE DIFFUSION + CAPTURE D'ACTIVATIONS ==========
                model_kwargs = dict(y=y)                
                # Déterminer le nombre d'échantillons pour le contrastive
                k_samples = getattr(cfg, 'contrastive_num_noisy_versions', 1) if use_contrastive_loss else 1
                if use_amp:
                    with autocast(dtype=autocast_dtype):
                        loss_dict = transport.training_losses(
                            model, x_clean, model_kwargs, k=k_samples
                        )
                        diffusion_loss = loss_dict["loss"]
                        activations = loss_dict["activations"] if "activations" in loss_dict else None
                else:
                    loss_dict = transport.training_losses(
                        model, x_clean, model_kwargs, k=k_samples
                    )
                    diffusion_loss = loss_dict["loss"]
                    activations = loss_dict["activations"] if "activations" in loss_dict else None

                # ========== PERTE CONTRASTIVE (si activée) ==========
                contrastive_loss = 0.0
                dispersive_loss = 0.0
                extra_loss = 0.0 
                jepa_activation = getattr(cfg, 'jepa_activation', False)
                use_extra_loss = getattr(cfg, 'use_extra_loss', False)
                if jepa_activation:
                    with torch.no_grad():
                        t = torch.zeros((x_clean.shape[0],)).to(device) + 0.99
                        ema_output, ema_zs = ema(x_clean, t, **model_kwargs)
                    
                    for k in range(len(activations)):
                        extra_loss += F.mse_loss(activations[k], ema_zs[k])

                if use_contrastive_loss or use_dispersive_loss:
                    features = activations[0]
                    B_times_k, number_of_patches, internal_dim = features.shape
                    batch_size = x_clean.size(0)

                if use_dispersive_loss:
                    if logger and train_steps < 1:
                        logger.info(f"Using dispersive only contrastive loss with {k_samples} samples")
                    for i in range(len(activations)):
                        dispersive_loss += dispersive_info_nce_loss(activations[i], temperature = cfg.dispersive_temperature, norm=cfg.use_norm_in_dispersive, logger = logger, use_l2=cfg.use_l2_in_dispersive,)
                    if logger:
                        logger.info(f"dispersive_loss: {dispersive_loss.item():.6f}")

                elif use_contrastive_loss and k_samples == 1: ##TODO Change the way the k samples are handled probably outa batch and sg()
                    mean_pos_sim = 0.0
                    mean_neg_sim = 0.0
                    with torch.no_grad():   
                        t = torch.zeros((x_clean.shape[0],)).to(device) + 1.0
                        ema_output, ema_zs = ema(x_clean, t, **model_kwargs)
                    for i in range(len(activations)):
                        act_batch_size = activations[i].shape[0]
                        contrastive_loss_one_latent, mean_pos_sim_one_latent, mean_neg_sim_one_latent = paired_info_nce_loss(activations[i].reshape(act_batch_size, -1), ema_zs[i].reshape(act_batch_size, -1), temperature = 0.5)
                        contrastive_loss += contrastive_loss_one_latent
                        mean_pos_sim += mean_pos_sim_one_latent
                        mean_neg_sim += mean_neg_sim_one_latent
                else:
                    if logger:
                        logger.debug("Skipping contrastive loss computation (not enabled or k_samples=1)")
                        contrastive_loss = 0.0

                # ========== COMBINAISON DES PERTES ==========
                contrastive_weight = getattr(cfg, 'contrastive_weight', 0.5)
                dispersive_weight = getattr(cfg, 'dispersive_weight', 0.5)
                total_loss = diffusion_loss + contrastive_weight * contrastive_loss + dispersive_weight * dispersive_loss + extra_loss
                if logger:
                    logger.info(f"diffusion_loss: {diffusion_loss}, contrastive_loss: {contrastive_loss}, dispersive_loss: {dispersive_loss}, extra_loss: {extra_loss}")
                    logger.info(f"total_loss: {total_loss}")
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
                running_total_loss += total_loss.item()  # ✅ ACCUMULER total_loss
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
                    # ✅ MOYENNER AUSSI LA PERTE TOTALE SUR TOUS LES PROCESSUS
                    avg_total_loss = torch.tensor(running_total_loss / log_steps, device=device)
                    dist.all_reduce(avg_total_loss, op=dist.ReduceOp.SUM)
                    avg_total_loss = avg_total_loss.item() / dist.get_world_size()
                    # ✅ MOYENNER LES PERTES CONTRASTIVES ET DISPERSIVES AUSSI
                    avg_contrastive_loss = None
                    avg_dispersive_loss = None
                    avg_extra_loss = None
                    if isinstance(contrastive_loss, torch.Tensor):
                        avg_contrastive_loss = torch.tensor(contrastive_loss.item(), device=device)
                        dist.all_reduce(avg_contrastive_loss, op=dist.ReduceOp.SUM)
                        avg_contrastive_loss = avg_contrastive_loss.item() / dist.get_world_size()
                    
                    if isinstance(dispersive_loss, torch.Tensor):
                        avg_dispersive_loss = torch.tensor(dispersive_loss.item(), device=device)
                        dist.all_reduce(avg_dispersive_loss, op=dist.ReduceOp.SUM)
                        avg_dispersive_loss = avg_dispersive_loss.item() / dist.get_world_size()
                    
                    if isinstance(extra_loss, torch.Tensor):
                        avg_extra_loss = torch.tensor(extra_loss.item(), device=device)
                        dist.all_reduce(avg_extra_loss, op=dist.ReduceOp.SUM)
                        avg_extra_loss = avg_extra_loss.item() / dist.get_world_size()

                    # LOG UNIQUEMENT DEPUIS RANK 0
                    if rank == 0:
                        log_msg = f"(step={train_steps:07d}) Diffusion Loss: {avg_diffusion_loss:.4f}"
                        log_dict = {"train/diffusion_loss": avg_diffusion_loss, 
                                    "train/steps_per_sec": steps_per_sec,
                                    "train/total_loss": avg_total_loss}
                        current_lr = opt.param_groups[0]['lr']
                        log_dict["train/lr"] = current_lr

                        if avg_contrastive_loss is not None and use_contrastive_loss:
                            log_msg += f", Contrastive Loss: {avg_contrastive_loss:.4f}, Total Loss: {avg_total_loss:.4f}"
                            log_dict.update({
                                "train/contrastive_loss": avg_contrastive_loss
                            })
                            if 'mean_pos_sim' in locals():
                                log_dict.update({
                                    "train/mean_pos_sim": mean_pos_sim,
                                    "train/mean_neg_sim": mean_neg_sim 
                                })
                        
                        if avg_dispersive_loss is not None and use_dispersive_loss:
                            log_msg += f", Dispersive Loss: {avg_dispersive_loss:.4f}, Total Loss: {avg_total_loss:.4f}"
                            log_dict.update({
                                "train/dispersive_loss": avg_dispersive_loss
                            })
                        
                        if avg_extra_loss is not None and use_extra_loss:
                            log_msg += f", Extra Loss: {avg_extra_loss:.4f}, Total Loss: {avg_total_loss:.4f}"
                            log_dict.update({
                                "train/extra_loss": avg_extra_loss
                            })
                        for i in range(len(activations)):
                            average_norm_per_patch = torch.norm(activations[i], dim=-1).mean().item()
                            average_norm_per_image = torch.norm(activations[i].reshape(local_batch_size, -1), dim=-1).mean().item()
                            log_dict.update({f"train/activation{i}_average_": average_norm_per_patch})
                            log_msg += f", Activation{i} norm: {average_norm_per_patch}"
                            log_dict.update({f"train/activation{i}_average_norm_per_image": average_norm_per_image})
                            log_msg += f", Activation{i} norm per image: {average_norm_per_image:.4f}"
                        ## Entropy logging
                        if getattr(cfg, 'log_entropy', True) and train_steps % getattr(cfg, 'log_entropy_every', 1000) == 0:
                            for i in range(len(activations)):
                                if logger:
                                    logger.info(f"Calculating entropy for activations of shape {activations[i].shape}")
                                entropy_vectors = activations[i].reshape(-1, activations[i].shape[-1]*activations[i].shape[-2])
                                entropy_mlp = compute_entropy(entropy_vectors)
                                if logger and rank == 0:
                                    logger.info(f"Entropy (activation{i}): {entropy_mlp:.6f} at step {train_steps}")
                                    log_dict.update({f"train/entropy_activation{i}": entropy_mlp})

                        log_msg += f", LR: {current_lr:.2e}"
                        logger.info(log_msg)
                        
                        if cfg.wandb and wandb_initialised:
                            wandb_utils.log(log_dict, step=train_steps)
                 
                    # Reset monitoring variables:
                    running_loss = 0
                    running_total_loss = 0
                    log_steps = 0
                    start_time = time()

                # Save SiT checkpoint:
                if train_steps % cfg.ckpt_every == 0 and train_steps > 0:
                    if rank == 0:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "train_steps": train_steps,
                            "epoch": epoch,  # Optionnel si vous voulez sauvegarder l'epoch
                            "scaler": scaler.state_dict() if scaler is not None else None,
                            "scheduler": lr_scheduler.state_dict() if lr_scheduler else None # ✅ SAUVEGARDER LE SCHEDULER
                        }
                        # Sauvegarder la config séparément si nécessaire
                        config_dict = OmegaConf.to_container(cfg, resolve=True)  # Convertir en dict standard
                        checkpoint["cfg"] = config_dict  # Dict standard compatible avec weights_only=True
                        checkpoint_path = f"{checkpoint_dir}/{epoch}_{train_steps:07d}.pt"
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

                    # ✅ CLEAN PCA VISUALIZATION ON TEST SET
                    if rank == 0 and cfg.get('visualize_pca_rgb', True) and test_iter is not None:
                        test_iter = run_pca_visualization_on_test_set(
                            cfg, ema, vae, transport, test_iter, test_loader, train_steps, wandb_initialised, logger, device
                        )
                   
                    if rank == 0:
                        # Prendre seulement les premières images pour logging (éviter duplications)
                        log_samples = out_samples[:num_sample_images]
                        if cfg.wandb and wandb_initialised:
                            wandb_utils.log_image(log_samples, train_steps)
                        logger.info(f"Generated and logged {num_sample_images} EMA samples.")
        except StopIteration:
            # Fin de l'epoch
            if rank == 0 and logger:
                logger.info(f"Completed epoch {epoch}")
            if rank == 0:
                logger.info(f"Saving end-of-epoch checkpoint for epoch {epoch}...")
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "train_steps": train_steps,
                    "epoch": epoch,  # Optionnel si vous voulez sauvegarder l'epoch
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "scheduler": lr_scheduler.state_dict() if lr_scheduler else None # ✅ SAUVEGARDER LE SCHEDULER
                }
                # Sauvegarder la config séparément si nécessaire
                config_dict = OmegaConf.to_container(cfg, resolve=True)  # Convertir en dict standard
                checkpoint["cfg"] = config_dict  # Dict standard compatible avec weights_only=True
                checkpoint_path = f"{checkpoint_dir}/epoch_finished_{epoch}_step{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()
    # Cleanup
    # if hook_handle:
    #     hook_handle.remove()

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    main()
