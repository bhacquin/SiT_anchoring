import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union


def paired_info_nce_loss(
    x_vecs: torch.Tensor, 
    y_vecs: torch.Tensor, 
    temperature: float = 0.1
) -> Tuple[torch.Tensor, float, float]:
    """
    Calcule la perte InfoNCE pour deux lots de vecteurs appariés.

    Pour chaque paire (X[i], Y[i]), elle les traite comme des paires positives. Toutes les autres
    combinaisons (X[i], Y[j] où i!=j), (X[i], X[j] où i!=j), et 
    (Y[i], Y[j] où i!=j) sont traitées comme des paires négatives.

    Args:
        x_vecs (torch.Tensor): Le premier lot de vecteurs, de forme (N, D).
        y_vecs (torch.Tensor): Le deuxième lot de vecteurs, de forme (N, D).
        temperature (float): Le paramètre de température pour mettre à l'échelle les logits.

    Returns:
        Tuple[torch.Tensor, float, float]: Un tuple contenant :
            - La perte InfoNCE calculée (tenseur scalaire).
            - La similarité moyenne des paires positives.
            - La similarité moyenne des paires négatives.
    """
    # --- 1. Validation et Préparation ---
    device = x_vecs.device
    N, D = x_vecs.shape
    
    # Normaliser les vecteurs pour utiliser la similarité cosinus
    x_vecs = F.normalize(x_vecs, dim=-1)
    y_vecs = F.normalize(y_vecs, dim=-1)
    
    # Concaténer les deux lots pour un calcul de similarité efficace
    # La forme devient (2*N, D)
    all_vecs = torch.cat([x_vecs, y_vecs], dim=0)
    
    # --- 2. Calcul de la Matrice de Similarité ---
    # Calcule la similarité cosinus entre chaque vecteur et tous les autres
    # sim_matrix a une forme de (2*N, 2*N)
    sim_matrix = torch.matmul(all_vecs, all_vecs.T)
    
    # Mettre à l'échelle par la température
    sim_matrix = sim_matrix / temperature
    
    # --- 3. Création des Masques pour les Paires Positives et Négatives ---
    # Les paires positives sont (X[i], Y[i]) et (Y[i], X[i])
    # Dans la matrice concaténée, cela correspond aux indices (i, i+N) et (i+N, i)
    pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
    pos_mask[torch.arange(N), torch.arange(N) + N] = True
    pos_mask[torch.arange(N) + N, torch.arange(N)] = True
    
    # Les paires négatives sont toutes les autres paires, sauf la similarité d'un vecteur avec lui-même
    # Créer un masque qui est vrai partout
    neg_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
    # Mettre la diagonale à False (un vecteur n'est pas négatif par rapport à lui-même)
    neg_mask.fill_diagonal_(False)
    # Retirer les paires positives du masque des négatifs
    neg_mask = neg_mask & ~pos_mask
    # --- 4. Calcul de la Perte ---
    # Pour chaque ancre, nous voulons maximiser la similarité avec son positif
    # par rapport à tous ses négatifs.
    
    # Sélectionner les similarités positives (une par ligne/ancre)
    # La forme est (2*N, 1)
    pos_sims = sim_matrix[pos_mask].view(2 * N, -1)
    
    # Sélectionner les similarités négatives
    # La forme est (2*N, 2*N - 2)
    neg_sims = sim_matrix[neg_mask].view(2 * N, -1)
    
    # Concaténer les positifs et les négatifs pour le calcul du softmax
    # La forme devient (2*N, 2*N - 1)
    logits = torch.cat([pos_sims, neg_sims], dim=1)
    
    # Les étiquettes pour la perte d'entropie croisée sont toujours 0, car
    # le positif est toujours à la première position (index 0) dans nos logits.
    labels = torch.zeros(2 * N, dtype=torch.long, device=device)
    
    # Calculer la perte d'entropie croisée, qui est équivalente à la perte InfoNCE
    loss = F.cross_entropy(logits, labels)
    
    # --- 5. Calcul des Statistiques pour le Suivi ---
    # Utiliser la matrice de similarité *avant* la mise à l'échelle par la température
    raw_sim_matrix = torch.matmul(all_vecs, all_vecs.T)
    mean_pos_sim = raw_sim_matrix[pos_mask].mean().item()
    mean_neg_sim = raw_sim_matrix[neg_mask].mean().item()
    
    return loss, mean_pos_sim, mean_neg_sim


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
    if vecs.ndim == 4:
        B, V, P, D = vecs.shape
        vecs = vecs.reshape(B * V, P * D)
    elif vecs.ndim == 3:
        B, N, C = vecs.shape
        vecs = vecs.reshape(B, N * C)  # Needs to be per image 
    else:
        raise NotImplementedError(f"Unsupported shape: {vecs.shape}")
    
    N, feature_dim = vecs.shape
    
    if N < 2:
        if logger:
            logger.warning(f"Besoin d'au moins 2 patches, obtenu {N}.")
        return torch.tensor(0.0, device=vecs.device, dtype=vecs.dtype)
    
    if use_l2:
        # z = vecs
        #     # def disp_loss(self, z): # Dispersive Loss implementation (InfoNCE-L2 variant)
        # z = z.reshape((z.shape[0],-1)) # flatten
        # # diff = torch.nn.functional.pdist(z).pow(2)/z.shape[1] # pairwise distance
        # diff = torch.nn.functional.cdist(z,z).pow(2)/z.shape[1]
        # # diff = torch.concat((diff, diff, torch.zeros(z.shape[0]).to(vecs.device)))  # match JAX implementation of full BxB matrix
        # print(f"🔍 diff result: {diff.mean().item():.6f}")
        # return torch.log(torch.exp(-diff).mean()) # calculate loss
        distances_condensed = torch.nn.functional.pdist(vecs, p=2).pow(2) / vecs.shape[1]  # Pairwise distance
        N = vecs.shape[0]
        distances_matrix = torch.zeros(N, N, device=vecs.device)
        triu_indices = torch.triu_indices(N, N, offset=1, device=vecs.device)
        distances_matrix[triu_indices[0], triu_indices[1]] = distances_condensed
        distances_matrix = distances_matrix + distances_matrix.T  # Symétrique
        # print(f"🔍 distances_matrix result: {distances_matrix.mean().item():.6f}")
        logsum = torch.logsumexp(-distances_matrix / temperature, dim=0)
        log_mean_exp = logsum - torch.log(torch.tensor(len(distances_matrix), dtype=torch.float, device=vecs.device))
        return log_mean_exp.mean()
        # Maintenant distances_matrix est (N, N) avec diagonale = 0
    else:
        vecs = F.normalize(vecs, dim=-1)
        similarities = -torch.matmul(vecs, vecs.T)  # (N, N)
        similarities.fill_diagonal_(-1)
    mean_exp = torch.exp(-similarities / temperature).mean()
    log_mean_exp = torch.log(mean_exp)
    # logsum = torch.logsumexp(-similarities / temperature, dim=0)
    print(f"🔍 log_mean_exp result: {log_mean_exp.item():.6f}")
    return log_mean_exp


