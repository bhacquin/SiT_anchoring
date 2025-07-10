
import os
import sys
import torch
import numpy as np


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