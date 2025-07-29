# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F  
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

# ✅ ADD CUSTOM FLEXIBLE PATCH EMBED
class FlexiblePatchEmbed(nn.Module):
    """
    Flexible 2D Image to Patch Embedding that can handle variable input sizes.
    Based on timm's PatchEmbed but without strict size checking.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, bias=True):
        super().__init__()
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.flatten = flatten
        
        # Calculate num_patches based on img_size (this is just for reference)
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.img_size = img_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # ✅ NO SIZE CHECKING - accept any size that's divisible by patch_size
        B, C, H, W = x.shape
        
        # Optional: Check if divisible by patch_size
        if H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0:
            raise ValueError(f"Input size ({H}, {W}) is not divisible by patch_size {self.patch_size}")
        
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        encoder_depth=[],
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        projection_dim=2048,
        use_projectors=[],
        z_dims=[],
        use_time=True, 
        is_teacher=False,
        use_cls_token=False,
    ):
        super().__init__()
        self.is_teacher = is_teacher
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.use_time = use_time
        self.z_dims = z_dims
        self.encoder_depth = encoder_depth
        self.use_projectors = use_projectors
        self.use_cls_token = use_cls_token

        self.capturing = False
        self.captured_activations = {}
        self.capture_layers = []

        # ✅ Validate that lists have the same length
        if len(encoder_depth) != len(use_projectors):
            raise ValueError(f"encoder_depth ({len(encoder_depth)}) and use_projectors ({len(use_projectors)}) must have the same length")
        # ✅ Create a mapping from encoder depth to projector index
        self.depth_to_projector = {depth: idx for idx, depth in enumerate(self.encoder_depth)}
        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.x_embedder = FlexiblePatchEmbed(
            img_size=input_size, 
            patch_size=patch_size, 
            in_chans=in_channels, 
            embed_dim=hidden_size, 
            bias=True
        )
        if self.use_time:
            self.t_embedder = TimestepEmbedder(hidden_size)
        else:
            self.t_embedder = None
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        num_patches = self.x_embedder.num_patches

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        # Will use fixed sin-cos embedding:
        pos_embed_size = num_patches + (1 if self.use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_size, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        # ✅ Create projectors only where use_projectors[i] is True
        self.projectors = nn.ModuleList()
        for i, should_use_projector in enumerate(use_projectors):
            if should_use_projector:
                self.projectors.append(build_mlp(hidden_size, projection_dim, z_dims[i]))
            else:
                self.projectors.append(None)  # Placeholder for consistency
        
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 
                                            int(self.x_embedder.num_patches ** 0.5), 
                                            cls_token=self.use_cls_token)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize cls_token
        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        if self.use_time:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) 
        num_patches_input = x.shape[1]
        num_patches_stored = self.pos_embed.shape[1]

        if num_patches_input != num_patches_stored:
            # print(f"🔍 DEBUG:")
            # print(f"   num_patches_input: {num_patches_input}")
            # print(f"   num_patches_stored: {num_patches_stored}")
            # print(f"   use_cls_token: {self.use_cls_token}")
            # print(f"   pos_embed shape: {self.pos_embed.shape}")
            
            # Calculate grid sizes BEFORE using them
            if self.use_cls_token:
                adjusted_patches_stored = num_patches_stored - 1  # Adjust for CLS token
            else:
                adjusted_patches_stored = num_patches_stored
            
            stored_grid_size = int(math.sqrt(adjusted_patches_stored))
            input_grid_size = int(math.sqrt(num_patches_input))
            
            # print(f"   adjusted_patches_stored: {adjusted_patches_stored}")
            # print(f"   stored_grid_size: {stored_grid_size}")
            # print(f"   input_grid_size: {input_grid_size}")
            # print(f"   stored_grid_size^2: {stored_grid_size * stored_grid_size}")
            # print(f"   input_grid_size^2: {input_grid_size * input_grid_size}")
            
            # ✅ Check if grid sizes are valid
            if stored_grid_size * stored_grid_size != adjusted_patches_stored:
                raise ValueError(f"stored_grid_size {stored_grid_size} doesn't match adjusted_patches_stored {adjusted_patches_stored}")
            if input_grid_size * input_grid_size != num_patches_input:
                raise ValueError(f"input_grid_size {input_grid_size} doesn't match num_patches_input {num_patches_input}")

        # ✅ FIXED: Handle CLS token logic properly
        if self.use_cls_token:
            # pos_embed includes CLS token, so separate them
            pos_embed_cls = self.pos_embed[:, 0:1, :]  # CLS token embedding
            pos_embed_patches = self.pos_embed[:, 1:, :]  # Patch embeddings only
            num_patches_stored_for_interpolation = pos_embed_patches.shape[1]
        else:
            # pos_embed only contains patch embeddings
            pos_embed_patches = self.pos_embed
            num_patches_stored_for_interpolation = num_patches_stored

        # ✅ FIXED: Interpolation logic
        if num_patches_input != num_patches_stored_for_interpolation:
            stored_grid_size = int(math.sqrt(num_patches_stored_for_interpolation))
            input_grid_size = int(math.sqrt(num_patches_input))

            # Reshape to 2D grid for interpolation
            pos_embed_grid = pos_embed_patches.permute(0, 2, 1).reshape(
                1, self.hidden_size, stored_grid_size, stored_grid_size
            )
            
            # Interpolate
            pos_embed_interpolated = F.interpolate(
                pos_embed_grid,
                size=(input_grid_size, input_grid_size),
                mode='bicubic',
                align_corners=False
            )

            # Reshape back to sequence
            pos_embed_final = pos_embed_interpolated.reshape(
                1, self.hidden_size, num_patches_input
            ).permute(0, 2, 1)
        else:
            # Sizes match, no interpolation needed
            pos_embed_final = pos_embed_patches

        # ✅ FIXED: Add CLS token BEFORE adding positional embeddings
        if self.use_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)  # CLS token at position 0

        # ✅ FIXED: Add positional embeddings only to patch tokens
        if self.use_cls_token:
            # x[:, 0] is CLS (no pos embed), x[:, 1:] are patches (get pos embed)
            x = torch.cat([
                x[:, 0:1],  # CLS token unchanged
                x[:, 1:] + pos_embed_final  # Add pos embed only to patches
            ], dim=1)
        else:
            # No CLS token, add pos embed to all tokens
            x = x + pos_embed_final

        if self.use_time:
            t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)

        if self.use_time:                               # (N, D)
            c = t + y
        else:
            c = y                              # (N, D)
        zs = []  # List to store projected representations at each encoder depth
        if self.capturing:
            self.captured_activations = {}

        for i, block in enumerate(self.blocks):
            x = block(x, c) # (N, T, D)
            layer_name = f'blocks.{i}'
            is_target_layer = self.capturing and layer_name in self.capture_layers
            
            # ✅ Capturer l'activation brute si c'est une couche cible
            if is_target_layer:
                self.captured_activations[f'{layer_name}.raw'] = x.clone()
            # ✅ Check if we should collect activations at this depth
            current_depth = i
            if current_depth in self.depth_to_projector:
                if self.is_teacher:
                    zs.append(x.clone())
                else:
                    projector_idx = self.depth_to_projector[current_depth]      
                    # ✅ Check if we should use a projector for this specific depth
                    if self.use_projectors[projector_idx]:
                        # Use projector MLP
                        projector = self.projectors[projector_idx]
                        if projector is not None:  # Extra safety check
                            # Get dimensions
                            N, T, D = x.shape
                            # Project and reshape
                            z = projector(x.reshape(-1, D)).reshape(N, T, -1)
                            zs.append(z)
                            if is_target_layer:
                                self.captured_activations[f'{layer_name}.projected'] = z.clone()
                        else:
                            # Fallback to raw activations if projector is None
                            zs.append(x.clone())
                    else:
                        # Return raw activations without projection
                        zs.append(x.clone())
        cls_output = None
        if self.use_cls_token:
            cls_output = x[:, 0]
            x = x[:, 1:]

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x, zs, cls_output

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out, zs = self.forward(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1), zs


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}
