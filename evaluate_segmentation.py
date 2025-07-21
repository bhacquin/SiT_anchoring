import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation

import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import os
from datasets import load_dataset
from models import SiT_models
from diffusers import AutoencoderKL
from utils import run_pca_visualization_on_test_set
from train_utils import init_wandb
from transport import create_transport

try:
    from torchmetrics import JaccardIndex
except ImportError:
    print("Avertissement : torchmetrics n'est pas installé. Le mIoU ne sera pas calculé.")
    print("Veuillez l'installer avec : pip install torchmetrics")
    JaccardIndex = None

class PatchToPixelSegmentationHead(nn.Module):
    """
    Tête de segmentation qui respecte la correspondance spatiale des patches.
    Chaque patch est mappé vers sa position correcte dans l'image finale.
    """
    def __init__(self, in_channels, num_classes, patch_pixel_size=16):
        super().__init__()
        self.num_classes = num_classes
        self.patch_pixel_size = patch_pixel_size
        self.pixels_per_patch = patch_pixel_size * patch_pixel_size  # 16×16 = 256
        
        # ✅ Projection : chaque patch (768D) → carré de pixels (256D)
        self.patch_to_pixels = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, self.pixels_per_patch)  # 768 → 256 (16×16 pixels)
        )
        
        # ✅ Classification de chaque pixel individuellement
        self.pixel_classifier = nn.Conv2d(1, num_classes, kernel_size=1)

    def forward(self, features):
        B, N, C = features.shape  # [B, 256, 768]
        
        # ✅ Calculer la grille de patches (√256 = 16×16 patches)
        patches_per_side = int(np.sqrt(N))  # 16
        
        # ✅ Mapper chaque patch vers ses pixels
        pixels = self.patch_to_pixels(features)  # [B, 256, 256]
        
        # ✅ CRUCIAL : Reshape en préservant l'ordre spatial des patches
        # Reshape: [B, 256, 256] → [B, 16, 16, 16, 16]
        #   - Les 2 premiers 16 = position du patch dans la grille
        #   - Les 2 derniers 16 = position du pixel dans le patch
        pixels = pixels.reshape(B, patches_per_side, patches_per_side, 
                               self.patch_pixel_size, self.patch_pixel_size)
        
        # ✅ Réorganiser pour former l'image finale en respectant les positions
        # Permuter les dimensions pour assembler correctement :
        # [B, patch_row, patch_col, pixel_row, pixel_col] → [B, patch_row, pixel_row, patch_col, pixel_col]
        pixels = pixels.permute(0, 1, 3, 2, 4)
        
        # ✅ Reshape final vers l'image complète
        image = pixels.reshape(B, 1, 
                              patches_per_side * self.patch_pixel_size,  # 16 * 16 = 256
                              patches_per_side * self.patch_pixel_size)  # 16 * 16 = 256
        
        # ✅ Classification finale de chaque pixel
        segmentation = self.pixel_classifier(image)  # [B, 21, 256, 256]
        
        return segmentation


class SegmentationHead(nn.Module):
    """
    Une tête de segmentation améliorée avec upsampling progressif et non-linéarités.
    """
    def __init__(self, in_channels, num_classes, target_size=256, activation_size=16):
        super().__init__()
        self.target_size = target_size
        self.activation_size = activation_size
        
        # ✅ Calculer l'upsampling de manière plus intelligente
        self.total_upsample = target_size // activation_size
        
        # ✅ Si l'upsampling est trop grand, faire ça progressivement
        if self.total_upsample >= 8:
            # Upsampling progressif : d'abord ×4, puis interpolation finale
            hidden_channels = max(in_channels // 2, 128)
            
            self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_channels)
            self.relu1 = nn.ReLU(inplace=True)
            
            # Deconv pour upsampler ×4
            self.deconv = nn.ConvTranspose2d(hidden_channels, hidden_channels//2, 
                                           kernel_size=4, stride=4, padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_channels//2)
            self.relu2 = nn.ReLU(inplace=True)
            
            # Projection finale
            self.final_conv = nn.Conv2d(hidden_channels//2, num_classes, kernel_size=1)
            
            # Upsampling final par interpolation
            self.final_upsample = self.total_upsample // 4
            
        else:
            # Pour des facteurs plus petits, approche simple
            hidden_channels = max(in_channels // 2, 64)
            self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_channels)
            self.relu1 = nn.ReLU(inplace=True)
            self.final_conv = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
            self.deconv = None
            self.final_upsample = self.total_upsample

    def forward(self, features):
        B, N, C = features.shape
        
        # Vérifier que N correspond bien à activation_size²
        expected_N = self.activation_size ** 2
        if N != expected_N:
            actual_size = int(np.sqrt(N))
            print(f"⚠️  Taille inattendue : {actual_size}×{actual_size} au lieu de {self.activation_size}×{self.activation_size}")
            self.activation_size = actual_size
            self.total_upsample = self.target_size // actual_size
            self.final_upsample = self.total_upsample if self.deconv is None else self.total_upsample // 4
        
        # Reshape en 2D
        features_2d = features.permute(0, 2, 1).reshape(B, C, self.activation_size, self.activation_size)
        
        # Première convolution
        x = self.conv1(features_2d)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Upsampling progressif si nécessaire
        if self.deconv is not None:
            x = self.deconv(x)
            x = self.bn2(x)
            x = self.relu2(x)
        
        # Projection finale vers les classes
        logits = self.final_conv(x)
        
        # Upsampling final
        if self.final_upsample > 1:
            masks = F.interpolate(
                logits, 
                scale_factor=self.final_upsample, 
                mode='bilinear', 
                align_corners=False
            )
        else:
            masks = logits
            
        return masks


def main(args):
    # Configuration
    device = torch.device(args.device)
    num_classes = 21  # 20 classes + 1 fond pour PASCAL VOC

    # Transformations (inchangées)
    image_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    target_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=Image.NEAREST),
        transforms.Lambda(lambda x: torch.as_tensor(np.array(x), dtype=torch.long))
    ])

    # Vérifier si les données sont complètement extraites
    voc_path = os.path.join(args.data_path, "VOCdevkit", "VOC2012")
    annotations_path = os.path.join(voc_path, "Annotations")
    images_path = os.path.join(voc_path, "JPEGImages")
    # Données considérées comme complètes si les dossiers principaux existent et ne sont pas vides
    data_complete = (
        os.path.exists(annotations_path) and 
        os.path.exists(images_path) and 
        len(os.listdir(annotations_path)) > 1000 and  # Au moins 1000 annotations
        len(os.listdir(images_path)) > 1000  # Au moins 1000 images
    )
    
    if data_complete:
        print("✅ Données PASCAL VOC déjà extraites et complètes.")
        download = False
    else:
        print("📥 Données incomplètes, téléchargement/extraction nécessaire...")
        download = True

    print(f"Chargement du jeu de données PASCAL VOC 2012 depuis {args.data_path}...")
    train_dataset = VOCSegmentation(
        root=args.data_path, 
        year='2012', 
        image_set='train', 
        download=download,
        transform=image_transform, 
        target_transform=target_transform
    )
    val_dataset = VOCSegmentation(
        root=args.data_path, 
        year='2012', 
        image_set='val', 
        download=False,
        transform=image_transform, 
        target_transform=target_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print("Jeu de données chargé.")
    # Chargement du modèle SiT pré-entraîné

    # ✅ Étape 1: Charger le VAE et le geler
    print("Chargement du VAE pré-entraîné...")
    # Utilise le VAE standard de Stable Diffusion 2.1, compatible avec la plupart des modèles
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print("VAE chargé et gelé.")

    if args.ckpt is not None:
        ckpt_path = args.ckpt
        try:
            checkpoint = torch.load(ckpt_path, weights_only=True, map_location='cpu')
            print("✅ Checkpoint loaded with weights_only=True")
        except Exception as e:
            print(f"Erreur lors du chargement du checkpoint avec weights_only=True : {e}")
            checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cpu')
        
        cfg = checkpoint["cfg"]  # Load the config from the checkpoint
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
    wandb_initialised = init_wandb(cfg, 0, None)
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
    # Faire un test rapide pour déterminer la taille des activations
    print("Détection de la taille des activations...")
    with torch.no_grad():
        # Prendre le premier batch pour tester
        test_images, _ = next(iter(train_loader))
        test_images = test_images[:1].to(device)  # Prendre juste 1 image
        
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
    # Création de la tête de segmentation
    feature_dim = sit_model.hidden_size
    # seg_head = SegmentationHead(
    #     in_channels=feature_dim, 
    #     num_classes=num_classes, 
    #     target_size=args.image_size,
    #     activation_size=activation_size  # Utiliser la taille détectée
    # ).to(device)
    seg_head = PatchToPixelSegmentationHead(
        in_channels=feature_dim, 
        num_classes=num_classes, 
        patch_pixel_size=16  # Chaque patch devient 16×16 pixels
    ).to(device)
    # Optimiseur
    optimizer = torch.optim.AdamW(seg_head.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignorer l'étiquette de bordure de PASCAL

    if JaccardIndex:
        metric = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255).to(device)
  

    
    best_miou = 0.0
    for epoch in range(args.epochs):
        seg_head.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
        # ✅ Adapter la boucle pour gérer le format de sortie de Hugging Face
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
                        # ✅ Étape A: Encoder les images en latents avec le VAE
            with torch.no_grad():
                # Le VAE attend des images normalisées entre -1 et 1, ce que fait déjà votre transform.
                latent_dist = vae.encode(images).latent_dist
                latents = latent_dist.sample()
                # Facteur de mise à l'échelle standard pour les latents de ce VAE
                latents = latents * 0.18215
            # Simuler les entrées nécessaires pour le SiT
            # ✅ Utiliser des valeurs fixes pour t et y pour une extraction de caractéristiques cohérente
            t = torch.full((images.size(0),), 0.999, device=device) # t proche de 1 (côté données)
            y = torch.full((images.size(0),), 1000, dtype=torch.long, device=device) # Classe non conditionnelle
            # latents = F.interpolate(images, size=(latent_size, latent_size))
            # ✅ Étape 1: Obtenir les activations du SiT gelé (pas de calcul de gradient ici)
            with torch.no_grad():
                _, activations = sit_model(latents, t, y=y)
                # Sélectionner les caractéristiques de la couche spécifiée
                features = activations[0]
            print(f"Shape des features : {features.shape}")  # Debugging
            optimizer.zero_grad()
            masks_pred = seg_head(features)
            loss = criterion(masks_pred, targets)
            
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        # Boucle de validation
        seg_head.eval()
        if JaccardIndex:
            metric.reset()
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]")
            for i, (images, targets) in enumerate(pbar_val):
                images, targets = images.to(device), targets.to(device)
                latent_dist = vae.encode(images).latent_dist
                latents = latent_dist.sample()
                latents = latents * 0.18215
                # ✅ Utiliser des valeurs fixes pour t et y pour une extraction de caractéristiques cohérente
                t = torch.full((images.size(0),), 0.999, device=device) # t proche de 1 (côté données)
                y = torch.full((images.size(0),), 1000, dtype=torch.long, device=device) # Classe non conditionnelle

                # latents = F.interpolate(images, size=(latent_size, latent_size))

                _, activations = sit_model(latents, t, y=y)

                features = activations[0]
                run_pca_visualization_on_test_set(
                            cfg, sit_model, transport, i, train_loader, i, wandb_initialised, None, device
                        )
                masks_pred = seg_head(features)
                
                if JaccardIndex:
                    metric.update(masks_pred.argmax(dim=1), targets)

        if JaccardIndex:
            miou = metric.compute().item()
            print(f"Epoch {epoch+1} | Validation mIoU: {miou:.4f}")
            if miou > best_miou:
                best_miou = miou
                print(f"✨ Nouveau meilleur mIoU ! Sauvegarde du modèle...")
                torch.save(seg_head.state_dict(), f"segmentation_head_best_layer_{cfg['encoder_depth']}.pt")

    print(f"\nÉvaluation terminée. Meilleur mIoU obtenu : {best_miou:.4f}")

  




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation des caractéristiques SiT sur PASCAL VOC.")
    parser.add_argument("--model", type=str, default="SiT-B/2", help="Architecture du modèle SiT.")
    parser.add_argument("--ckpt", type=str, required=False, default="/capstor/scratch/cscs/vbastien/SiT_anchoring/outputs/SiT-B/2/JEPA_True/Time_Cond_True/2025-07-19/Contrast_False__DivFalse_L2_False/checkpoints/epoch_finished_104_step0525420.pt", help="Chemin vers le checkpoint SiT pré-entraîné (.pt).")
    parser.add_argument("--data-path", type=str, default="./data", help="Chemin pour stocker le jeu de données PASCAL VOC.")
    parser.add_argument("--image-size", type=int, default=256, help="Taille de l'image sur laquelle le SiT a été entraîné.")
    # parser.add_argument("--layer-index", type=int, required=True, help="Index de la couche du Transformer à sonder.")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre d'époques pour entraîner la tête de segmentation.")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille du lot.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taux d'apprentissage pour l'optimiseur.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    main(args)

### Comment utiliser ce script :



