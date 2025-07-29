import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
from tqdm import tqdm
import numpy as np
import argparse
import os
from datasets import load_dataset
from models import SiT_models
from diffusers import AutoencoderKL
from utils import capture_intermediate_activations
from train_utils import init_wandb
from omegaconf import OmegaConf
import wandb
from typing import Literal
import hydra
from jaxtyping import Float
import logging
from transport import create_transport

os.environ["HF_DATASETS_CACHE"] = "/capstor/scratch/cscs/vbastien/imagenet_cache"

logger = logging.getLogger(__name__)

class SigLoss(nn.Module):
    """SigLoss.

        This follows `AdaBins <https://arxiv.org/abs/2011.14141>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.

    Adapted from: https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/depth/models/losses/sigloss.py
    """

    def __init__(
        self, valid_mask=True, loss_weight=1.0, max_depth=None, warm_up=False, warm_iter=100, loss_name="sigloss"
    ):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.loss_name = loss_name

        self.eps = 0.001

        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]

        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt):
        """Forward function."""

        loss_depth = self.loss_weight * self.sigloss(depth_pred, depth_gt)
        return loss_depth


class DepthBinProbe(nn.Module):
    """
    Sonde de profondeur qui implémente la logique de classification par bacs (binning).
    Prend les caractéristiques d'un patch [B, N, C] et prédit une carte de profondeur.
    """
    def __init__(self, in_channels, n_bins=256, min_depth=0.1, max_depth=10.0, target_size=256):
        super().__init__()
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.target_size = target_size

        # Tête de classification : prédit un logit pour chaque bac de profondeur
        # On utilise une Conv2d car on va remodeler les features en 2D
        self.conv_depth = nn.Conv2d(in_channels, n_bins, kernel_size=1, padding=0)
        
        # Création des centres des bacs (stratégie UD - Uniform Discretization)
        # On les enregistre comme buffer pour qu'ils soient sur le bon device
        self.register_buffer('bins', torch.linspace(self.min_depth, self.max_depth, self.n_bins))

    def forward(self, features):
        # features a la forme [B, N, C] (Batch, Nb_Patches, Canaux)
        B, N, C = features.shape
        
        # 1. Calculer la taille spatiale de la carte de caractéristiques
        patches_per_side = int(np.sqrt(N)) # Ex: sqrt(256) = 16
        
        # 2. Reshape les features en format image 2D: [B, N, C] -> [B, C, H_feat, W_feat]
        features_2d = features.permute(0, 2, 1).reshape(B, C, patches_per_side, patches_per_side)
        
        # 3. Prédit les logits pour chaque bac
        logits = self.conv_depth(features_2d) # [B, n_bins, H_feat, W_feat]
        
        # 4. Normalise les logits pour obtenir une distribution de probabilité (style AdaBins)
        # C'est la stratégie "linear" de votre code original
        logits = torch.relu(logits) + 1e-6 # Epsilon pour la stabilité
        probs = logits / logits.sum(dim=1, keepdim=True) # [B, n_bins, H_feat, W_feat]
        
        # 5. Calcule la profondeur comme une somme pondérée des centres des bacs
        # C'est l'opération einsum de votre code original
        depth_map = torch.einsum("bchw,c->bhw", probs, self.bins).unsqueeze(1) # [B, 1, H_feat, W_feat]
        
        # 6. Upsample à la taille de l'image cible
        if (patches_per_side, patches_per_side) != (self.target_size, self.target_size):
            depth_map = F.interpolate(
                depth_map,
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False
            )
            
        return depth_map

from torch.utils.data import Dataset
import scipy.io as sio
from PIL import Image

# class NYUDepthV2MatDataset(Dataset):
#     def __init__(self, root_dir, split='train', transform=None, target_transform=None):
#         super().__init__()
#         self.root_dir = root_dir
#         self.split = split
#         self.transform = transform
#         self.target_transform = target_transform

#         main_mat_path = os.path.join(self.root_dir, 'nyu_depth_v2_labeled.mat')
#         splits_mat_path = os.path.join(self.root_dir, 'splits.mat')

#         if not os.path.isdir(self.root_dir):
#             raise NotADirectoryError(f"Le chemin fourni n'est pas un dossier : {self.root_dir}.")
#         if not os.path.exists(main_mat_path):
#             raise FileNotFoundError(f"Fichier introuvable : {main_mat_path}.")
#         if not os.path.exists(splits_mat_path):
#             raise FileNotFoundError(f"Fichier introuvable : {splits_mat_path}.")

#         # ✅ Utiliser h5py pour lire le fichier .mat v7.3
#         print(f"Chargement de {main_mat_path} avec h5py...")
#         with h5py.File(main_mat_path, 'r') as f:
#             # Les données sont stockées comme des datasets HDF5
#             # h5py lit les données avec les dimensions (channels, width, height, N)
#             # Nous les transposons pour correspondre à notre attente (N, height, width, channels)
#             self.images = np.array(f['images']).transpose(3, 2, 1, 0)
#             self.depths = np.array(f['depths']).transpose(2, 1, 0)

#         # Le fichier de splits est petit, on peut garder scipy
#         print(f"Chargement de {splits_mat_path}...")
#         splits_mat = sio.loadmat(splits_mat_path)
#         self.indices = (splits_mat['trainNdxs' if self.split == 'train' else 'testNdxs'].ravel() - 1)
#         print(f"Dataset chargé. Split '{self.split}' avec {len(self.indices)} échantillons.")
#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         real_idx = self.indices[idx]
#         # Les données sont déjà des numpy arrays, pas besoin de les re-convertir
#         image = self.images[real_idx]
#         depth = self.depths[real_idx]
        
#         image_pil = Image.fromarray(image, 'RGB')
        
#         image_tensor = self.transform(image_pil) if self.transform else transforms.ToTensor()(image_pil)
#         depth_tensor = self.target_transform(Image.fromarray(depth)) if self.target_transform else torch.from_numpy(depth.copy()).unsqueeze(0)
        
#         return {"pixel_values": image_tensor, "depth_values": depth_tensor}

@torch.no_grad()
def compute_depth_metrics(pred, gt, max_depth=10.0):
    mask = (gt > 1e-3) & (gt <= max_depth)
    if not mask.any(): return {"rmse": float('inf'), "abs_rel": float('inf')}
    pred = pred[mask]; gt = gt[mask]
    rmse = torch.sqrt(torch.mean((pred - gt) ** 2))
    abs_rel = torch.mean(torch.abs(pred - gt) / gt)
    return {"rmse": rmse.item(), "abs_rel": abs_rel.item()}

# ✅ CORRECTION : Créer une fonction collate qui utilise les bons paramètres
def create_collate_fn(image_size):
    def collate_fn_inner(batch):
        image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # ✅ CORRECTION CRITIQUE : Pas de normalisation pour les depth maps !
        depth_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            # ✅ Conversion manuelle au lieu de ToTensor() qui normalise
        ])
        
        images = []
        depths = []
        
        for idx, item in enumerate(batch):
            img_pil = item["image"].convert("RGB")
            depth_pil = item["depth_map"]
            
            # Transformation de l'image (normale)
            img_tensor = image_transform(img_pil)
            images.append(img_tensor)
            # ✅ DIAGNOSTIC COMPLET (premier item seulement)
            if idx == 0:
                logger.debug(f"\n🔍 DIAGNOSTIC DEPTH MAP:")
                logger.debug(f"   PIL Mode: {depth_pil.mode}")
                logger.debug(f"   PIL Size: {depth_pil.size}")
                
                # Test plusieurs façons de lire
                depth_np_original = np.array(depth_pil, dtype=np.float32)
                logger.debug(f"   Données originales - Range: [{depth_np_original.min():.3f}, {depth_np_original.max():.3f}]")
                logger.debug(f"   Données originales - Shape: {depth_np_original.shape}")
                logger.debug(f"   Données originales - Dtype: {depth_np_original.dtype}")
                logger.debug(f"   Données originales - Unique values (first 10): {np.unique(depth_np_original.flatten())[:10]}")
                
                # Essayer avec différents dtypes
                if depth_np_original.max() == 0:
                    logger.debug(f"   ❌ Données nulles détectées ! Essayons d'autres approches...")
                    
                    # Test avec uint16
                    try:
                        depth_np_uint16 = np.array(depth_pil, dtype=np.uint16)
                        logger.debug(f"   Test uint16 - Range: [{depth_np_uint16.min()}, {depth_np_uint16.max()}]")
                    except Exception as e:
                        logger.debug(f"   Test uint16 - Erreur: {e}")
                    
                    # Test avec uint32
                    try:
                        depth_np_uint32 = np.array(depth_pil, dtype=np.uint32)
                        logger.debug(f"   Test uint32 - Range: [{depth_np_uint32.min()}, {depth_np_uint32.max()}]")
                    except Exception as e:
                        logger.debug(f"   Test uint32 - Erreur: {e}")

                    # Test avec getdata()
                    try:
                        data_list = list(depth_pil.getdata())
                        unique_vals = list(set(data_list))[:10]
                        logger.debug(f"   Test getdata() - Unique values: {unique_vals}")
                        logger.debug(f"   Test getdata() - Min/Max: [{min(data_list)}, {max(data_list)}]")
                    except Exception as e:
                        logger.debug(f"   Test getdata() - Erreur: {e}")



            # ✅ TRAITEMENT ADAPTATIF DES DEPTH MAPS
            # 1. Redimensionner
            depth_resized = depth_pil.resize((image_size, image_size), Image.NEAREST)
            # 2. Convertir en numpy
            depth_np = np.array(depth_resized, dtype=np.float32)
            if idx == 0:
                logger.debug(f"   Après resize - Range: [{depth_np.min():.3f}, {depth_np.max():.3f}]")
            
            # 3. Conversion adaptative basée sur le range détecté
            max_val = depth_np.max()
            min_val = depth_np.min()
            
            if max_val == 0:
                logger.warning(f"❌ ERREUR: Depth map complètement nulle après resize !")
                logger.debug(f"   Essayons de récupérer les données autrement...")
                                # Essayer avec les données brutes de getdata()
                try:
                    data_list = list(depth_pil.resize((image_size, image_size), Image.NEAREST).getdata())
                    depth_from_data = np.array(data_list, dtype=np.float32).reshape(image_size, image_size)
                    if depth_from_data.max() > 0:
                        depth_np = depth_from_data
                        max_val = depth_np.max()
                        if idx == 0:
                            logger.debug(f"   ✅ Récupéré avec getdata() - Range: [{depth_np.min():.3f}, {depth_np.max():.3f}]")
                    else:
                        if idx == 0:
                            logger.warning(f"   ❌ Même avec getdata(), les données sont nulles")
                except Exception as e:
                    if idx == 0:
                        logger.warning(f"   ❌ Erreur avec getdata(): {e}")
            # Conversion basée sur le range final
            if max_val > 10000:
                # Format 16-bit (0-65535)
                depth_meters = depth_np / (2**16 - 1) * 10.0
                if idx == 0:
                    logger.debug(f"   Appliqué: Conversion 16-bit")
            elif max_val > 1000:
                # Format 12-bit ou autre
                depth_meters = depth_np / max_val * 10.0
                if idx == 0:
                    logger.debug(f"   Appliqué: Conversion proportionnelle max={max_val:.1f}")
            elif max_val > 10:
                # Format 8-bit (0-255)
                depth_meters = depth_np / 255.0 * 10.0
                if idx == 0:
                    logger.debug(f"   Appliqué: Conversion 8-bit")
            elif max_val > 1.0:
                # Format déjà en mètres ou proche
                depth_meters = depth_np
                if idx == 0:
                    logger.debug(f"   Appliqué: Pas de conversion")
            elif max_val > 0:
                # Format normalisé (0-1)
                depth_meters = depth_np * 10.0
                if idx == 0:
                    logger.debug(f"   Appliqué: Multiplication par 10")
            else:
                # Données complètement nulles - créer des données factices pour éviter le crash
                depth_meters = np.ones_like(depth_np) * 0.1  # Depth minimale de 10cm
                if idx == 0:
                    logger.debug(f"   Appliqué: Données factices (0.1m partout)")
                       
            # 3. Appliquer la conversion correcte comme dans image2depth
            # Les depth maps sont en 16-bit (0-65535), pas 8-bit (0-255)
            # depth_np /= (2**16 - 1)  # Normaliser [0, 65535] -> [0, 1]
            # depth_np *= 10.0         # Remettre à l'échelle [0, 1] -> [0, 10] mètres

            if idx == 0:
                logger.debug(f"   Final range: [{depth_meters.min():.3f}, {depth_meters.max():.3f}] mètres")
                logger.debug(f"   ===========================================")
            
            # 4. Convertir en tensor PyTorch
            depth_tensor = torch.from_numpy(depth_meters).unsqueeze(0)  # [1, H, W]
            depths.append(depth_tensor)
        
        return {
            "pixel_values": torch.stack(images),
            "depth_values": torch.stack(depths)
        }
    return collate_fn_inner


def main(args):
    device = torch.device(args.device)
    
    print("Chargement du jeu de données NYU Depth V2 depuis Hugging Face...")
    dataset = load_dataset("sayakpaul/nyu_depth_v2", split='train',trust_remote_code=True)
    val_dataset = load_dataset("sayakpaul/nyu_depth_v2", split='validation',trust_remote_code=True)

    def transform(examples):
        image_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        depth_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), antialias=True),
            transforms.ToTensor(),
        ])
        
        images = [image_transform(img.convert("RGB")) for img in examples["image"]]
        depths = [depth_transform(depth.convert("L")) for depth in examples["depth_map"]]
        
        examples["pixel_values"] = torch.stack(images)
        examples["depth_values"] = torch.stack(depths)
        return examples
    # dataset.set_transform(transform)
    # val_dataset.set_transform(transform) 
    # train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    # ✅ CRÉER LES DATALOADERS AVEC LA FONCTION COLLATE
        # ✅ UTILISER LA FONCTION COLLATE CORRIGÉE
    collate_fn = create_collate_fn(args.image_size)
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=collate_fn  # ✅ UTILISER LA FONCTION COLLATE
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_fn  # ✅ UTILISER LA FONCTION COLLATE
    )
    
    # --- Chargement des modèles (VAE + SiT) ---
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device); vae.eval(); vae.requires_grad_(False)
    checkpoint = torch.load(args.ckpt, map_location='cpu')


    if args.ckpt is not None:
        ckpt_path = args.ckpt
        try:
            checkpoint = torch.load(ckpt_path, weights_only=True, map_location='cpu')
            print("✅ Checkpoint loaded with weights_only=True")
        except Exception as e:
            print(f"Erreur lors du chargement du checkpoint avec weights_only=True : {e}")
            checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cpu')
        
        cfg = OmegaConf.create(checkpoint["cfg"])
    print(f"Loading SiT model '{args.model}'...")
    latent_size = args.image_size // 8
    sit_model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=cfg.get('num_classes', 1000),
        use_time=cfg.get('use_time', True),  # Use time embeddings if specified
        encoder_depth=cfg['encoder_depth'],
        use_projectors=cfg['use_projectors'],
        z_dims=cfg['z_dims'],
        learn_sigma=cfg.get('learn_sigma', True)
    )
    wandb_initialised = init_wandb(cfg, 0, logger)
    transport = create_transport(
        cfg['path_type'],
        cfg['prediction'],
        cfg['loss_weight'],
        cfg['train_eps'],
        cfg['sample_eps'],
    )
    sit_model = sit_model.to(device)
    sit_model.load_state_dict(checkpoint["ema"])
    sit_model.eval()
    for param in sit_model.parameters():
        param.requires_grad = False
    print("Model loaded.")


    with torch.no_grad():
        # Prendre le premier batch pour tester
        test_batch = next(iter(train_loader))
        test_images = test_batch["pixel_values"][:1].to(device)  # Prendre juste 1 image
        test_depths = test_batch["depth_values"][:1].to(device)   # Pour débugger aussi
        print(f"✅ Test batch réussi:")
        print(f"  - Shape images: {test_images.shape}")
        print(f"  - Shape depths: {test_depths.shape}")
        print(f"  - Device: {test_images.device}")        
        # Encoder avec le VAE
        latent_dist = vae.encode(test_images).latent_dist
        test_latents = latent_dist.sample() * 0.18215
        
        # Passer dans le SiT
        test_t = torch.full((1,), 0.999, device=device)
        test_y = torch.full((1,), 1000, dtype=torch.long, device=device)
        _, test_activations = sit_model(test_latents, test_t, y=test_y)
        
        # Analyser la taille des activations
        test_features = test_activations[0]
        B, N, C = test_features.shape
        activation_size = int(np.sqrt(N))
        
        print(f"Taille des activations : {activation_size}x{activation_size}")
        print(f"Taille de l'image cible : {args.image_size}x{args.image_size}")
        
        # Calculer le facteur d'upsampling correct
        correct_upsample_factor = args.image_size // activation_size
        print(f"Facteur d'upsampling correct : {correct_upsample_factor}")

    # ✅ AJOUTER UN TEST POUR VÉRIFIER LES RANGES
    with torch.no_grad():
        print("🔍 Vérification des ranges de données...")
        test_batch = next(iter(train_loader))
        test_images = test_batch["pixel_values"][:1].to(device)
        test_depths = test_batch["depth_values"][:1].to(device)
        
        print(f"✅ Images shape: {test_images.shape}")
        print(f"✅ Images range: [{test_images.min():.3f}, {test_images.max():.3f}]")
        print(f"✅ Depths shape: {test_depths.shape}")
        print(f"✅ Depths range: [{test_depths.min():.3f}, {test_depths.max():.3f}]")
        
        # ✅ Les depths devraient être dans [0, 10] mètres environ
        if test_depths.max() > 20 or test_depths.min() < 0:
            print("❌ ATTENTION: Range des depths anormal !")
            print("   Les depths devraient être entre 0 et ~10 mètres")
        # Test du modèle...
        latent_dist = vae.encode(test_images).latent_dist
        test_latents = latent_dist.sample() * 0.18215
        test_t = torch.full((1,), 0.999, device=device)
        test_y = torch.full((1,), 1000, dtype=torch.long, device=device)
        
        # ✅ UTILISER capture_intermediate_activations comme dans l'entraînement
        activations_dict = capture_intermediate_activations(sit_model, test_latents, test_t, test_y, [f'blocks.{i}' for i in range(len(sit_model.blocks))])
        
        if activations_dict:
            first_key = list(activations_dict.keys())[0]
            test_features = activations_dict[first_key]
            B, N, C = test_features.shape
            activation_size = int(np.sqrt(N))
            print(f"✅ Activations capturées: {len(activations_dict)} couches")
            print(f"✅ Première activation shape: {test_features.shape}")
            print(f"✅ Activation spatial size: {activation_size}x{activation_size}")
        else:
            print("❌ ERREUR: Aucune activation capturée !")
            return       



    # --- Configuration des sondes de profondeur pour chaque couche ---
    num_blocks = len(sit_model.blocks)
    layer_names = [f'blocks.{i}' for i in range(num_blocks)]
    feature_dim = sit_model.hidden_size
    
    depth_probes = nn.ModuleList([DepthBinProbe(in_channels=feature_dim, target_size=args.image_size, n_bins=args.n_bins) for _ in layer_names]).to(device)
    optimizers = [torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=1e-4) for probe in depth_probes]
    criterion = SigLoss(max_depth=args.max_depth)
    best_rmse = [float('inf')] * len(layer_names)

    # --- Boucle d'entraînement ---
    for epoch in range(args.epochs):
        for probe in depth_probes: probe.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
        epoch_losses = []
        for batch in pbar:
            images = batch["pixel_values"].to(device)
            depth_gt = batch["depth_values"].to(device)
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * 0.18215
                t = torch.full((images.size(0),), 0.999, device=device)
                y = torch.full((images.size(0),), 1000, dtype=torch.long, device=device)
                activations_dict = capture_intermediate_activations(sit_model, latents, t, y, layer_names)

            batch_losses = []
            for i, layer_name in enumerate(layer_names):
                if layer_name in activations_dict:
                    features = activations_dict[layer_name]
                    optimizers[i].zero_grad()
                    depth_pred = depth_probes[i](features)
                    # ✅ VÉRIFICATION AVANT CALCUL DE LA LOSS
                    if torch.isnan(depth_pred).any() or torch.isinf(depth_pred).any():
                        print(f"❌ NaN/Inf détecté dans les prédictions pour {layer_name}")
                        continue                   
                    loss = criterion(depth_pred, depth_gt)
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"❌ NaN/Inf détecté dans la loss pour {layer_name}")
                        print(f"   Pred range: [{depth_pred.min():.3f}, {depth_pred.max():.3f}]")
                        print(f"   GT range: [{depth_gt.min():.3f}, {depth_gt.max():.3f}]")
                        continue
                    loss.backward()
                    optimizers[i].step()
                    batch_losses.append(loss.item())
            if batch_losses:
                avg_batch_loss = np.mean(batch_losses)
                epoch_losses.append(avg_batch_loss)
                pbar.set_postfix(avg_loss=f"{avg_batch_loss:.4f}")
            else:
                pbar.set_postfix(avg_loss="No valid losses")

        # --- Validation ---
        for probe in depth_probes: 
            probe.eval()

        all_metrics = [[] for _ in layer_names]
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]")
            for batch in pbar_val:
                images = batch["pixel_values"].to(device)
                depth_gt = batch["depth_values"].to(device)
                latents = vae.encode(images).latent_dist.sample() * 0.18215
                t = torch.full((images.size(0),), 0.999, device=device)
                y = torch.full((images.size(0),), 1000, dtype=torch.long, device=device)
                activations_dict = capture_intermediate_activations(sit_model, latents, t, y, layer_names)
                for i, layer_name in enumerate(layer_names):
                    if layer_name in activations_dict:
                        features = activations_dict[layer_name]
                        depth_pred = depth_probes[i](features)
                        # ✅ DEBUGGING : Afficher quelques valeurs pour vérifier
                        if len(all_metrics[i]) == 0:  # Premier batch seulement
                            print(f"\n🔍 Validation debug pour {layer_name}:")
                            print(f"   Pred range: [{depth_pred.min():.3f}, {depth_pred.max():.3f}]")
                            print(f"   GT range: [{depth_gt.min():.3f}, {depth_gt.max():.3f}]")
                        metrics = compute_depth_metrics(depth_pred, depth_gt, max_depth=args.max_depth)
                        all_metrics[i].append(metrics)

        print(f"\n📊 Epoch {epoch+1} Results:")
        if epoch_losses:
            print(f"📈 Training loss moyenne: {np.mean(epoch_losses):.4f}")
        
        for i, layer_name in enumerate(layer_names):
            layer_idx = int(layer_name.split('.')[1])
            
            if all_metrics[i]:
                valid_rmses = [m['rmse'] for m in all_metrics[i] if m['rmse'] != float('inf')]
                if valid_rmses:
                    avg_rmse = np.mean(valid_rmses)
                    median_rmse = np.median(valid_rmses)
                    print(f"Layer {layer_idx:2d}: RMSE = {avg_rmse:.4f} (median: {median_rmse:.4f}, {len(valid_rmses)}/{len(all_metrics[i])} valid)")
                    
                    if avg_rmse < best_rmse[i]:
                        best_rmse[i] = avg_rmse
                        print(f"           ✨ New best!")
                        torch.save(depth_probes[i].state_dict(), f"depth_probe_bins_layer_{layer_idx:02d}_best.pt")
                else:
                    print(f"Layer {layer_idx:2d}: No valid metrics")
            else:
                print(f"Layer {layer_idx:2d}: No metrics computed")

    print(f"\n🏆 Meilleure couche globale : {np.argmin(best_rmse)} avec un RMSE de {np.min(best_rmse):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation des caractéristiques SiT pour la prédiction de profondeur avec binning.")
    parser.add_argument("--model", type=str, default="SiT-B/2")
    parser.add_argument("--ckpt", type=str, default='/capstor/scratch/cscs/vbastien/SiT_anchoring/outputs/SiT-B/2/JEPA_True/Time_Cond_True/2025-07-27/Contrast_False__DivFalse_L2_False/checkpoints/epoch_finished_80_step0405324.pt')
    parser.add_argument("--data-path", type=str, default='/capstor/scratch/cscs/vbastien/SiT_anchoring/data/NYU', help="Chemin vers le dossier contenant nyu_depth_v2_labeled.mat.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_bins", type=int, default=256)
    parser.add_argument("--max_depth", type=float, default=10.0)
    args = parser.parse_args()
    main(args)