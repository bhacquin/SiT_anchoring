import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union


def dispersive_info_nce_loss(vecs: torch.Tensor, norm: Optional[bool] = True, temperature: float = 0.1, 
                           logger: Optional[object] = None, use_l2: bool = False) -> torch.Tensor:
    """
    Perte dispersive pour encourager la diversité entre toutes les représentations.
    Minimise la similarité moyenne entre toutes les paires de vecteurs.
    
    Args:
        vecs: (B, V, P, D) ou (B*V, P*D) - Vecteurs de caractéristiques
        norm: Si True, normalise les vecteurs (recommandé pour similarité cosinus)
        temperature: Paramètre de température
        use_l2: Si True, utilise la distance L2, sinon similarité cosinus
        logger: Logger optionnel
    
    Returns:
        torch.Tensor: Perte dispersive (scalaire)
    """
    # Gérer les dimensions d'entrée flexibles
    if vecs.ndim == 4:
        B, V, P, D = vecs.shape
        vecs = vecs.reshape(B * V, P * D)  # Aplatir vers (N, feature_dim)
    elif vecs.ndim == 3:
        vecs = vecs.reshape(vecs.shape[0], -1)  # Aplatir vers (N, feature_dim)
    elif vecs.ndim != 2:
        err_msg = f"L'entrée 'vecs' doit être 2D, 3D ou 4D, mais a obtenu {vecs.ndim}D de forme {vecs.shape}."
        if logger: 
            logger.error(err_msg)
        return torch.tensor(0.0, device=vecs.device, dtype=vecs.dtype)
    
    N, feature_dim = vecs.shape
    
    if N < 2:
        if logger:
            logger.warning(f"Perte dispersive : Besoin d'au moins 2 échantillons, obtenu {N}.")
        return torch.tensor(0.0, device=vecs.device, dtype=vecs.dtype)
    
    # Normalisation optionnelle
    if norm:
        vecs = F.normalize(vecs, dim=-1)
    
    if use_l2:
        # Distance L2 : plus c'est grand, plus c'est différent
        # On veut maximiser les distances, donc minimiser leur négatif
        distances = torch.cdist(vecs, vecs, p=2)  # (N, N)
        squared_distances = distances ** 2  # (N, N)
        similarities = -squared_distances
        
    else:
        # Similarité cosinus : plus c'est grand, plus c'est similaire
        # On veut minimiser les similarités
        similarities = torch.matmul(vecs, vecs.T)  # (N, N)
    exp_similarities = torch.exp(similarities.view(-1) / temperature)
    mean_exp_similarities = exp_similarities.mean() 
    loss = torch.log(mean_exp_similarities + 1e-8)  
    return loss


def info_nce_loss(vecs: torch.Tensor, temperature: float = 0.1, logger: Optional[object] = None, 
                  sample_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float, float]:
    """
    Perte contrastive InfoNCE standard au niveau image avec pondération optionnelle par échantillon.
    Chaque vue d'image est traitée comme un vecteur aplati de taille (P*D).
    
    - Positifs : Similarité cosinus entre différentes vues de la même image originale
    - Négatifs : Similarité cosinus entre vues d'images originales différentes

    Args:
        vecs: (B, V, P, D) - B images, V vues, P patches par vue, D dimension.
        temperature: Paramètre de température pour softmax.
        logger: Logger optionnel pour le débogage.
        sample_weights: (B, V) - Poids optionnels pour chaque vue. Si fourni, chaque vue 
                       sera pondérée selon son poids correspondant dans le calcul de la perte.

    Returns:
        loss: Perte InfoNCE standard.
        mean_pos_sim: Similarité positive moyenne (similarité cosinus, avant température).
        mean_neg_sim: Similarité négative moyenne (similarité cosinus, avant température).
    """
    if not isinstance(vecs, torch.Tensor):
        err_msg = f"L'entrée 'vecs' doit être un torch.Tensor, obtenu {type(vecs)}"
        if logger: logger.error(err_msg)
        return torch.tensor(0.0, device='cpu', dtype=torch.float32), 0.0, 0.0

    if vecs.ndim != 4:
        err_msg = f"L'entrée 'vecs' doit être 4D (B, V, P, D), mais a obtenu {vecs.ndim}D de forme {vecs.shape}."
        if logger: logger.error(err_msg)
        return torch.tensor(0.0, device=vecs.device, dtype=vecs.dtype), 0.0, 0.0
        
    B, V, P, D = vecs.shape

    if B == 0:
        if logger: logger.warning("Perte InfoNCE : La taille du lot B est 0. Retour de la perte 0.")
        return torch.tensor(0.0, device=vecs.device, dtype=vecs.dtype), 0.0, 0.0

    if V < 2:
        if logger: logger.warning(f"Perte InfoNCE : V < 2 (V={V}). Aucune paire positive possible. Retour de la perte 0.")
        return torch.tensor(0.0, device=vecs.device, dtype=vecs.dtype), 0.0, 0.0
    
    if P == 0:
        if logger: logger.warning("Perte InfoNCE : Le nombre de patches P est 0. Retour de la perte 0.")
        return torch.tensor(0.0, device=vecs.device, dtype=vecs.dtype), 0.0, 0.0

    # Aplatir chaque vue d'image en un vecteur de taille (P*D)
    # Forme: (B, V, P*D)
    # vecs = F.normalize(vecs, dim=-1)
    vecs_flat = vecs.reshape(B, V, P * D)
    
    # Normaliser les caractéristiques au niveau des vues aplaties
    z = F.normalize(vecs_flat, dim=-1)  # (B, V, P*D)
    # Reshape vers (B*V, P*D) pour faciliter les calculs
    z_flat = z.reshape(B * V, P * D)  # (N, P*D) où N = B*V
    N = z_flat.shape[0]
    
    # Créer les indices d'image pour chaque vue
    img_ids = torch.arange(B, device=vecs.device).repeat_interleave(V)  # [0,0,...,1,1,...,B-1,B-1]
    
    # Préparer les poids par vue si fournis
    view_weights = None
    if sample_weights is not None:
        if sample_weights.shape != (B, V):
            if logger: 
                logger.error(f"sample_weights doit avoir la forme (B={B}, V={V}), mais a obtenu {sample_weights.shape}")
            sample_weights = None
        else:
            # Aplatir les poids pour correspondre à z_flat: (B*V,)
            view_weights = sample_weights.reshape(B * V)
    
    # Calculer la matrice de similarité cosinus entre toutes les paires de vues
    sim_matrix = torch.matmul(z_flat, z_flat.T)  # (N, N)
    
    # Appliquer la température
    sim_matrix_temp = sim_matrix / temperature
    
    # Masques pour définir positifs et négatifs
    same_img = img_ids.unsqueeze(0) == img_ids.unsqueeze(1)  # (N, N)
    # Ne pas inclure la diagonale (similarité avec soi-même)
    not_self = ~torch.eye(N, dtype=torch.bool, device=vecs.device)  # (N, N)
    
    # Positifs: même image, vue différente
    pos_mask = same_img & not_self  # (N, N)
    
    # Négatifs: image différente
    neg_mask = ~same_img  # (N, N)
    
    # Collecter les statistiques sur les similarités brutes (avant température)
    # Optimisation: utiliser seulement la partie triangulaire supérieure pour éviter le double comptage
    upper_tri_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=vecs.device), diagonal=1)
    
    # Statistiques avec optimisation triangulaire supérieure
    pos_sims = sim_matrix[pos_mask]  # Similarités positives uniques
    neg_sims = sim_matrix[neg_mask]  # Similarités négatives uniques
    
    if len(pos_sims) == 0:
        if logger: logger.warning("Perte InfoNCE : Aucune paire positive trouvée. Retour de la perte 0.")
        return torch.tensor(0.0, device=vecs.device, dtype=vecs.dtype), 0.0, 0.0
    
    if len(neg_sims) == 0:
        if logger: logger.warning("Perte InfoNCE : Aucune paire négative trouvée. Retour de la perte 0.")
        return torch.tensor(0.0, device=vecs.device, dtype=vecs.dtype), 0.0, 0.0
    
    # Calculer la perte InfoNCE vectorisée avec pondération optionnelle
    total_loss = torch.tensor(0.0, device=vecs.device, dtype=vecs.dtype)
    total_weight = 0.0
    num_anchors = 0  # Compter le nombre d'ancres avec au moins une paire positive
    
    # Mode standard InfoNCE
    for i in range(N):
        # Trouver les positifs et négatifs pour cette ancre
        pos_indices = torch.where(pos_mask[i])[0]
        neg_indices = torch.where(neg_mask[i])[0]
        
        if len(pos_indices) == 0:
            continue
            
        # Incrémenter le compteur d'ancres valides
        num_anchors += 1
            
        # Obtenir le poids pour cette ancre
        anchor_weight = 1.0
        if view_weights is not None:
            anchor_weight = view_weights[i].item()
        
        # Similarités pour cette ancre (avec température)
        pos_sims_i = sim_matrix_temp[i, pos_indices]  # (num_pos,)
        neg_sims_i = sim_matrix_temp[i, neg_indices]  # (num_neg,)
        
        # InfoNCE: -log(sum(exp(pos)) / (sum(exp(pos)) + sum(exp(neg))))
        # = -log_sum_exp(pos) + log_sum_exp(pos + neg)
        pos_log_sum_exp = torch.logsumexp(pos_sims_i, dim=0)
        all_sims = torch.cat([pos_sims_i, neg_sims_i])
        all_log_sum_exp = torch.logsumexp(all_sims, dim=0)
        
        anchor_loss = -pos_log_sum_exp + all_log_sum_exp
        
        # Appliquer le poids de l'échantillon à cette perte d'ancre
        weighted_loss = anchor_loss * anchor_weight
        total_loss += weighted_loss
        total_weight += anchor_weight
    
    if total_weight == 0:
        if logger: logger.warning("Perte InfoNCE : Poids total nul. Retour de la perte 0.")
        return torch.tensor(0.0, device=vecs.device, dtype=vecs.dtype), 0.0, 0.0
    
    # Moyenne pondérée
    final_loss = total_loss / total_weight
    
    # Statistiques
    mean_pos_sim = pos_sims.mean().item()
    mean_neg_sim = neg_sims.mean().item()

    return final_loss, mean_pos_sim, mean_neg_sim

