import torch.nn as nn
import torch
import torch.nn.functional as F

def info_nce_loss(vecs: torch.Tensor, temperature: float = 0.1, logger: Optional[object] = None, 
                  sample_weights: Optional[torch.Tensor] = None, use_divergent_only: bool = False, use_log_mean: bool = True) -> Tuple[torch.Tensor, float, float]:
    """
    Perte contrastive InfoNCE simplifiée au niveau image avec pondération optionnelle par échantillon.
    Chaque vue d'image est traitée comme un vecteur aplati de taille (P*D).
    
    - Positifs : Similarité cosinus entre différentes vues de la même image originale
    - Négatifs : Similarité cosinus entre vues d'images originales différentes

    Args:
        vecs: (B, V, P, D) - B images, V vues, P patches par vue, D dimension.
        temperature: Paramètre de température pour softmax.
        logger: Logger optionnel pour le débogage.
        sample_weights: (B, V) - Poids optionnels pour chaque vue. Si fourni, chaque vue 
                       sera pondérée selon son poids correspondant dans le calcul de la perte.
        use_divergent_only: Si True, utilise seulement la partie répulsive (négative) de la perte.

    Returns:
        loss: Perte InfoNCE.
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
    # z = vecs_flat   
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
    pos_sims = sim_matrix[pos_mask & upper_tri_mask]  # Similarités positives uniques
    neg_sims = sim_matrix[neg_mask & upper_tri_mask]  # Similarités négatives uniques
    
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
    
    if use_divergent_only:
        # Mode divergent STABILISÉ : seulement la partie répulsive (négative)
        for i in range(N):
            neg_indices = torch.where(neg_mask[i])[0]
            
            if len(neg_indices) == 0:
                continue
                
            num_anchors += 1
                
            # Obtenir le poids pour cette ancre
            anchor_weight = 1.0
            if view_weights is not None:
                anchor_weight = view_weights[i].item()
            
            # Similarités négatives pour cette ancre (avec température)
            neg_sims_i = sim_matrix_temp[i, neg_indices]  # (num_neg,)
            
            # --- CORRECTION ---
            # Ancien code instable :
            # neg_log_sum_exp = torch.logsumexp(neg_sims_i, dim=0)
            # anchor_loss = neg_log_sum_exp

            # Nouveau code stable :
            # On veut minimiser la similarité, donc on minimise la moyenne des exponentielles.
            # exp(sim/τ) est toujours positif. En minimisant sa moyenne, on force sim à être petit.
            # Cette formulation est plus stable que logsumexp car elle n'implique pas un log externe.
            if use_log_mean:
                # Utiliser log_mean pour une meilleure stabilité numérique
                anchor_loss = torch.log(torch.mean(torch.exp(neg_sims_i)))
            else:
                anchor_loss = torch.exp(neg_sims_i).mean()

            # Appliquer le poids de l'échantillon à cette perte d'ancre
            weighted_loss = anchor_loss * anchor_weight
            total_loss += weighted_loss
            total_weight += anchor_weight
    else:
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
    
    # Statistiques - ajuster selon le mode
    if use_divergent_only:
        mean_pos_sim = 0.0  # Pas utilisé en mode divergent
        mean_neg_sim = neg_sims.mean().item() if len(neg_sims) > 0 else 0.0
    else:
        mean_pos_sim = pos_sims.mean().item()
        mean_neg_sim = neg_sims.mean().item()

    return final_loss, mean_pos_sim, mean_neg_sim


class SimpleInfoNCE(nn.Module):
    """
    Simple InfoNCE Loss for Diffusion Models
    
    Takes a tensor with multiple noisy versions of different images and automatically
    separates them into positive and negative pairs based on the batch structure.
    
    Expected input: tensor of shape (batch_size * k, feature_dim) where k is the number
    of noisy versions per image, organized as:
    [img1_v1, img1_v2, ..., img1_vk, img2_v1, img2_v2, ..., img2_vk, ...]
    """
    
    def __init__(self, temperature=0.5):
        super(SimpleInfoNCE, self).__init__()
        self.temperature = temperature
        
    def forward(self, features, batch_size, k):
        """
        Compute InfoNCE loss for diffusion training
        
        Args:
            features (torch.Tensor): Features from noisy versions 
                                   Shape: (batch_size * k, feature_dim)
            batch_size (int): Original batch size (number of different images)
            k (int): Number of noisy versions per image
        
        Returns:
            torch.Tensor: InfoNCE loss
        """
        # Handle potential multi-dimensional features by flattening to 2D
        if features.dim() > 2:
            # Flatten to (batch_size * k, -1) 
            features = features.contiguous().view(features.shape[0], -1)
        
        # Verify input dimensions
        expected_samples = batch_size * k
        if features.shape[0] != expected_samples:
            raise ValueError(f"Expected {expected_samples} samples, got {features.shape[0]}")
        
        if k < 2:
            raise ValueError(f"Need at least 2 versions per image for contrastive learning, got k={k}")
        
        device = features.device
        
        # Ensure features require gradients
        if not features.requires_grad:
            features = features.requires_grad_(True)
        
        # L2 normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create positive mask: versions of the same image are positive with each other
        positive_mask = torch.zeros(expected_samples, expected_samples, device=device, dtype=torch.bool)
        
        for i in range(batch_size):
            # All indices for image i
            start_idx = i * k
            end_idx = (i + 1) * k
            
            # Set all-to-all positives within versions of same image (excluding self)
            for idx1 in range(start_idx, end_idx):
                for idx2 in range(start_idx, end_idx):
                    if idx1 != idx2:  # Exclude auto-similarity
                        positive_mask[idx1, idx2] = True
        
        # Instead of masking diagonal to -9e15, we'll handle it differently to preserve gradients
        # Create mask for diagonal elements
        self_mask = torch.eye(expected_samples, device=device, dtype=torch.bool)
        
        # Compute InfoNCE loss using logsumexp over all positives
        total_loss = 0.0
        num_queries = 0
        
        for query_idx in range(expected_samples):
            pos_mask = positive_mask[query_idx]  # All positives for this query
            
            if pos_mask.any():  # If there are positives for this query
                # Get similarities for this query (excluding self)
                query_similarities = similarity_matrix[query_idx]
                
                # Positive similarities (excluding self)
                pos_similarities = query_similarities[pos_mask]
                
                # All similarities (excluding self) for denominator
                neg_mask = ~pos_mask & ~self_mask[query_idx]  # Not positive and not self
                neg_similarities = query_similarities[neg_mask]
                
                # Combine positive and negative similarities
                all_similarities = torch.cat([pos_similarities, neg_similarities])
                
                # InfoNCE: log(exp(pos).sum()) - log(exp(all).sum())
                log_sum_exp_pos = torch.logsumexp(pos_similarities, dim=0)
                log_sum_exp_all = torch.logsumexp(all_similarities, dim=0)
                
                loss_i = -(log_sum_exp_pos - log_sum_exp_all)
                total_loss = total_loss + loss_i
                num_queries += 1
        
        # Return mean loss, ensuring it requires gradients
        if num_queries > 0:
            final_loss = total_loss / num_queries
        else:
            final_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return final_loss


# Keep the old class for backward compatibility but mark as deprecated
class DiffusionInfoNCE(nn.Module):
    """
    DEPRECATED: Use SimpleInfoNCE instead
    """
    
    def __init__(self, temperature=0.07, reduction='mean', num_noisy_versions=1, include_clean_as_positive=True):
        super(DiffusionInfoNCE, self).__init__()
        print("WARNING: DiffusionInfoNCE is deprecated. Use SimpleInfoNCE instead.")
        self.temperature = temperature
        self.simple_infonce = SimpleInfoNCE(temperature=temperature)
        
    def forward(self, clean_features, noisy_features):
        """
        Backward compatibility wrapper
        """
        # For backward compatibility, just use the first set of features
        batch_size = clean_features.shape[0]
        
        # Combine clean and noisy features
        all_features = torch.cat([clean_features, noisy_features], dim=0)
        k = all_features.shape[0] // batch_size
        
        return self.simple_infonce(all_features, batch_size, k)


# Utility function for easy usage
def simple_info_nce_loss(features, batch_size, k, temperature=0.5):
    """
    Convenient function to compute InfoNCE loss
    
    Args:
        features (torch.Tensor): Features tensor (batch_size * k, feature_dim)
        batch_size (int): Original batch size
        k (int): Number of versions per image
        temperature (float): Temperature parameter
    
    Returns:
        torch.Tensor: InfoNCE loss
    """
    loss_fn = SimpleInfoNCE(temperature=temperature)
    return loss_fn(features, batch_size, k)


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing SimpleInfoNCE ===")
    
    # Test configuration
    batch_size = 4
    k = 3  # 3 noisy versions per image
    feature_dim = 128
    
    # Create test features: (batch_size * k, feature_dim)
    # Organization: [img1_v1, img1_v2, img1_v3, img2_v1, img2_v2, img2_v3, ...]
    features = torch.randn(batch_size * k, feature_dim)
    
    # Test SimpleInfoNCE
    loss_fn = SimpleInfoNCE(temperature=0.5)
    loss = loss_fn(features, batch_size, k)
    
    print(f"Input shape: {features.shape}")
    print(f"Batch size: {batch_size}, k: {k}")
    print(f"InfoNCE Loss: {loss.item():.4f}")
    
    # Test with gradient computation
    features.requires_grad_(True)
    loss = loss_fn(features, batch_size, k)
    loss.backward()
    
    print(f"Gradient computed successfully: {features.grad is not None}")
    print(f"Gradient norm: {features.grad.norm().item():.4f}")
    
    print("\n=== Test completed ===")
    
    # Test multi-dimensional features (spatial features)
    print("\n=== Testing with spatial features ===")
    spatial_features = torch.randn(batch_size * k, 256, 8, 8)  # Spatial features
    loss_spatial = loss_fn(spatial_features, batch_size, k)
    print(f"Spatial features shape: {spatial_features.shape}")
    print(f"InfoNCE Loss (spatial): {loss_spatial.item():.4f}")
