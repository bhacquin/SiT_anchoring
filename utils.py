
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import wandb    
import wandb_utils


@torch.no_grad()
def run_pca_visualization_on_test_set(cfg, ema_model, transport, test_iter, test_loader, train_steps, wandb_initialised, logger, device, same_batch=True):
    """
    Run PCA-RGB visualization on test set, using the provided iterator.
    """
    logger.info("🎨 Running PCA-RGB visualization on test set...")
    if same_batch:
        logger.info("🔄 Using the same batch for PCA visualization")
    try:
        if same_batch:
            test_iter = iter(test_loader)  # Reset iterator to start from the beginning
        try:
            viz_x, viz_labels = next(test_iter)
        except StopIteration:
            # Reset iterator if exhausted
            test_iter = iter(test_loader)
            viz_x, viz_labels = next(test_iter)
            logger.info("🔄 Test iterator reset - cycling through test set again")
        
        viz_x = viz_x.to(device)
        viz_labels = viz_labels.to(device)
        
        # Encode to latents
        viz_latents = vae.encode(viz_x).latent_dist.sample().mul_(0.18215)
        
        # Test different timesteps
        if cfg.get('use_time', True):
            t_to_test = cfg.get('pca_t_to_test', [0.01, 0.25, 0.5, 0.75, 0.99])
        else:
            t_to_test = [0.5]

        if cfg.get('noise_to_test', [0.01, 0.25, 0.5, 0.75, 0.99]) is not None:
            noise_to_add = cfg.get('noise_to_test', [0.01, 0.25, 0.5, 0.75, 0.99])
        else:
            noise_to_add = [1.]
            logger.info("🔇 No noise added to PCA")
        # Log original images for this timestep
        if cfg.wandb and wandb_initialised:
            wandb_utils.log({
                f"test_originals": [wandb.Image(img) for img in viz_x]
            }, step=train_steps)
        for i, noise in enumerate(noise_to_add):
            if noise == 1.:
                xt = viz_latents.clone()
            else:
                viz_noise = torch.full((viz_latents.shape[0],), noise, device=device)
                _ , x0, x1_current = transport.sample(viz_latents)
                _, xt, ut = transport.path_sampler.plan(viz_noise, x0, x1_current)

            for j, t_value in enumerate(t_to_test):
                viz_t = torch.full((viz_latents.shape[0],), t_value, device=device)
                if logger:
                    logger.debug(f"xt.shape: {xt.shape}, viz_t.shape: {viz_t.shape}, viz_labels.shape: {viz_labels.shape}")
                # Get layers to visualize
                layer_names = cfg.get('pca_layers', [f'blocks.{i}' for i in [3, 6, 12, 18, 24]])
                num_blocks = len(ema_model.blocks)
                logger.info(f"🔍 Visualizing PCA for layers: {layer_names} (num_blocks={num_blocks})") 
                valid_layer_names = [name for name in layer_names if int(name.split('.')[1]) < num_blocks]
                
                if not valid_layer_names:
                    logger.warning("⚠️ No valid layer names for PCA visualization")
                    continue
                
                # Capture activations
                activations_dict = capture_intermediate_activations(
                    ema_model, xt, viz_t, viz_labels, 
                    layer_names=valid_layer_names
                )
                
                if activations_dict:
                    
                    # Process each layer
                    for layer_name, layer_activations in activations_dict.items():
                        pca_images_np = visualize_pca_as_rgb(
                            layer_activations
                        )

                        if pca_images_np:
                            wandb_images = [
                                wandb.Image(img, caption=f"Sample {i} - t={t_value:.2f}")
                                for i, img in enumerate(pca_images_np)
                            ]
                            
                            if cfg.wandb and wandb_initialised:
                                wandb_utils.log({
                                    f"pca_visualization_t_{t_value:.2f}_noise_{noise:.2f}/{layer_name}": wandb_images
                                }, step=train_steps)
                                
                    logger.info(f"✅ PCA visualizations logged for timestep {t_value:.2f}")
                else:
                    logger.warning(f"⚠️ No activations captured for timestep {t_value:.2f}")
        
        logger.info("✅ Test set PCA visualization complete")
        return test_iter  # ✅ Return the iterator for next use
        
    except Exception as e:
        logger.error(f"❌ PCA visualization failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return test_iter  # Return iterator even on error


def visualize_pca_as_rgb(layer_activations: torch.Tensor) -> list:
    """
    Transforme les activations d'une couche pour un batch d'images en images RGB via PCA.

    Args:
        layer_activations (torch.Tensor): Les activations pour une seule couche.
                                          Shape attendue: (B, N, D).

    Returns:
        List[np.ndarray]: Une liste d'images RGB (format NumPy), une pour chaque image du batch.
    """
    if not isinstance(layer_activations, torch.Tensor) or len(layer_activations.shape) != 3:
        print(f"❌ Format d'activation invalide pour PCA-RGB. Attendu (B, N, D), reçu {layer_activations.shape}.")
        return []

    batch_size = layer_activations.shape[0]
    rgb_images = []

    for i in range(batch_size):
        try:
            # Shape: (num_patches, dim)
            features = layer_activations[i].detach().cpu().numpy()

            # Appliquer PCA pour obtenir les 3 composantes principales
            pca = PCA(n_components=3)
            transformed = pca.fit_transform(features)

            # Normaliser chaque composante entre 0 et 1 pour former une image RGB
            rgb_image_components = np.zeros_like(transformed)
            for component_idx in range(3):
                comp = transformed[:, component_idx]
                min_val, max_val = comp.min(), comp.max()
                if max_val > min_val:
                    rgb_image_components[:, component_idx] = (comp - min_val) / (max_val - min_val)
                # Si max_val == min_val, la composante est déjà 0, donc on ne fait rien.

            # Redimensionner en grille 2D
            num_patches = features.shape[0]
            grid_size = int(np.sqrt(num_patches))
            if grid_size * grid_size != num_patches:
                print(f"⚠️  Skipping PCA image {i}: {num_patches} patches is not a perfect square.")
                continue

            rgb_image = rgb_image_components.reshape(grid_size, grid_size, 3)
            rgb_images.append(rgb_image)

        except Exception as e:
            print(f"❌ Erreur lors du traitement PCA-RGB pour l'image {i}: {e}")
            continue

    return rgb_images

def capture_intermediate_activations(model, x, t, y, layer_names=None):
    """
    Capture les activations des couches intermédiaires pendant un forward pass
    
    Args:
        model: Le modèle SiT
        x: Input tensor
        t: Timesteps
        y: Labels
        layer_names: Liste des noms de couches à capturer (ex: ['blocks.4', 'blocks.8'])
        
    Returns:
        Dict[str, torch.Tensor] - Activations capturées
    """

    activations = {}
    hooks = []

    # Définir les couches par défaut si non spécifiées
    if layer_names is None:
        # Prendre quelques couches représentatives
        num_blocks = len(model.blocks) if hasattr(model, 'blocks') else 12
        layer_names = [
            f'blocks.{i}' for i in range(num_blocks)
        ]
    def make_hook(name):
        def hook_fn(module, input, output):
            activations[name] = output.clone().detach()
        return hook_fn
    
    # Attacher les hooks
    try:
        for layer_name in layer_names:
            layer = get_layer_by_name(model, layer_name)
            hook = layer.register_forward_hook(make_hook(layer_name))
            hooks.append(hook)
        # Forward pass
        with torch.no_grad():
            _ = model(x, t, y)

        return activations

    except Exception as e:
        print(f"❌ Error in capture_intermediate_activations:")
        raise
    finally:
        # Nettoyer les hooks
        for hook in hooks:
            hook.remove()


def compute_entropy(x: torch.Tensor, num_bins: int = 512) -> float:
    """
    Compute empirical entropy of tensor features.
    
    Args:
        x: (N, d) tensor
        num_bins: Number of bins for histogram
        
    Returns:
        Mean entropy across all dimensions
    """
    x = x.detach().cpu().numpy()
    
    # Check for NaN or infinite values
    if not np.isfinite(x).all():
        print(f"Warning: Non-finite values detected in entropy computation. NaN count: {np.isnan(x).sum()}, Inf count: {np.isinf(x).sum()}")
        # Replace NaN and infinite values with zeros
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    entropies = []
    
    for i in range(x.shape[1]):
        # Get finite values only for this dimension
        col_data = x[:, i]
        finite_mask = np.isfinite(col_data)
        
        if finite_mask.sum() == 0:
            # If no finite values, entropy is 0
            entropies.append(0.0)
            continue
            
        finite_data = col_data[finite_mask]
        
        # Skip if all values are the same (entropy would be 0)
        if len(np.unique(finite_data)) == 1:
            entropies.append(0.0)
            continue
            
        try:
            hist, _ = np.histogram(finite_data, bins=num_bins, density=True)
            hist = hist + 1e-8  # avoid log(0)
            hist = hist / hist.sum()
            ent = -np.sum(hist * np.log(hist))
            entropies.append(ent)
        except Exception as e:
            print(f"Warning: Error computing histogram for dimension {i}: {e}")
            entropies.append(0.0)
        
    return float(np.mean(entropies)) if entropies else 0.0


def get_layer_by_name(model, layer_name):
    # Navigue dans le modèle selon le nom de la couche (ex: "model.dit.blocks.0")
    module = model
    attrs = layer_name.split('.')
    for attr in attrs:
        # Si le module est un wrapper DDP, descend dans .module
        while hasattr(module, 'module') and not hasattr(module, attr):
            module = module.module
        if attr.isdigit():
            module = module[int(attr)]
        else:
            module = getattr(module, attr)
    return module


def get_config_info():
    """Détermine le chemin et nom de la configuration"""
    
    # 1. Vérifier les arguments de ligne de commande pour --config-file
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--config-file='):
            config_file = arg.split('=', 1)[1]
            abs_path = os.path.abspath(config_file)
            if os.path.exists(abs_path):
                config_dir = os.path.dirname(abs_path)
                config_name = os.path.splitext(os.path.basename(abs_path))[0]
                # Retirer cet argument pour éviter les conflits avec Hydra
                sys.argv.pop(i)
                print(f"📋 Using config from --config-file: {abs_path}")
                return config_dir, config_name
            else:
                print(f"❌ Config file not found: {abs_path}")
                sys.exit(1)
    
    # 2. Vérifier les variables d'environnement (pour compatibilité)
    job_config_path = os.environ.get('JOB_CONFIG_PATH')
    if job_config_path:
        # Si le chemin est relatif, le résoudre par rapport au répertoire parent du script
        if not os.path.isabs(job_config_path):
            # Remonter au répertoire parent (sortir de /train/)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            full_path = os.path.join(project_root, job_config_path)
        else:
            full_path = job_config_path
        
        if os.path.exists(full_path):
            config_dir = os.path.dirname(full_path)
            config_name = os.path.splitext(os.path.basename(full_path))[0]
            print(f"📋 Using config from JOB_CONFIG_PATH: {full_path}")
            return config_dir, config_name
        else:
            print(f"⚠️ Config file not found: {full_path}")
            print(f"   JOB_CONFIG_PATH was: {job_config_path}")
    
    # 3. Configuration par défaut
    print("📋 Using default config from current directory")
    return "config", "config"

# Debug pour voir ce qui se passe
print(f"🔍 Current working directory: {os.getcwd()}")
print(f"🔍 Script location: {os.path.abspath(__file__)}")
print(f"🔍 JOB_CONFIG_PATH: {os.environ.get('JOB_CONFIG_PATH', 'NOT_SET')}")