import os
import wandb
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from copy import deepcopy
from torchvision.transforms import CenterCrop

def create_scheduler(optimizer, cfg, total_steps):
    """
    Crée un scheduler de learning rate avec une phase de warm-up optionnelle.
    """
    use_warmup = getattr(cfg, 'use_warmup', False)
    use_scheduler = getattr(cfg, 'use_scheduler', False)

    if not use_warmup and not use_scheduler:
        return None, None  # Pas de scheduler

    schedulers = []
    milestones = []

    # 1. Phase de Warm-up
    if use_warmup:
        warmup_steps = getattr(cfg, 'warmup_steps', 1000)
        warmup_init_lr = getattr(cfg, 'warmup_init_lr', 1e-6)
        
        # Calcule le facteur de départ pour le warm-up
        start_factor = warmup_init_lr / cfg.learning_rate
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=start_factor, total_iters=warmup_steps
        )
        schedulers.append(warmup_scheduler)
        milestones.append(warmup_steps)
        
        # Le reste de l'entraînement se fera sur les pas restants
        main_scheduler_steps = total_steps - warmup_steps
    else:
        main_scheduler_steps = total_steps

    # 2. Phase principale du Scheduler
    if use_scheduler:
        scheduler_type = getattr(cfg, 'scheduler_type', 'cosine')
        
        if scheduler_type == 'cosine':
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=main_scheduler_steps
            )
        elif scheduler_type == 'step':
            step_size = getattr(cfg, 'scheduler_step_size', 30) # en epochs
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size * (total_steps // cfg.epochs)
            )
        else: # 'linear'
            main_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=main_scheduler_steps
            )
        
        schedulers.append(main_scheduler)

    if not schedulers:
        return None, None

    # Enchaîner les schedulers (warm-up puis principal)
    if len(schedulers) > 1:
        sequential_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=schedulers, milestones=milestones
        )
        return sequential_scheduler, "step" # Se met à jour à chaque pas
    else:
        # Si seulement warm-up ou seulement scheduler principal
        return schedulers[0], "step"



def get_layer_output_dim(model, layer_name, input_shape=(2, 3, 32, 32)):
    """
    Détermine automatiquement la dimension de sortie d'une couche donnée
    
    Args:
        model: Le modèle SiT
        layer_name: Nom de la couche (ex: 'blocks.8')
        input_shape: Shape d'entrée pour le test (batch_size, channels, height, width)
    
    Returns:
        int: Dimension de sortie de la couche
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Créer un input dummy
    dummy_input = torch.randn(input_shape, device=device)
    dummy_t = torch.randn(input_shape[0], device=device)
    dummy_y = torch.randint(0, 1000, (input_shape[0],), device=device)  # Labels de classe aléatoires
    
    # Hook temporaire pour capturer la sortie
    captured_output = {}
    
    def capture_hook(module, input, output):
        captured_output['output'] = output
    
    # Trouver et enregistrer le hook
    hook_handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            hook_handle = module.register_forward_hook(capture_hook)
            break
    
    if hook_handle is None:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    try:
        # Forward pass pour capturer la sortie
        with torch.no_grad():
            _ = model(dummy_input, dummy_t, dummy_y)
        
        if 'output' not in captured_output:
            raise RuntimeError(f"No output captured for layer {layer_name}")
        
        output = captured_output['output']
        
        # Déterminer la dimension
        if output.dim() == 2:  # (batch_size, features)
            dim = output.shape[1]
        elif output.dim() == 4:  # (batch_size, channels, height, width)
            dim = output.shape[1] * output.shape[2] * output.shape[3]
        elif output.dim() == 3:  # (batch_size, seq_len, features)
            dim = output.shape[2]
        else:
            # Flatten tout sauf la dimension batch
            dim = output.numel() // output.shape[0]
        
        print(f"✅ Layer {layer_name} output shape: {output.shape}, feature dim: {dim}")
        return dim
        
    finally:
        # Supprimer le hook
        hook_handle.remove()
        model.train()

class ActivationCapture:
    """
    Capture simple d'activations avec gradients préservés pour loss contrastive
    """
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def get_activation_hook(self, name):
        def hook(module, input, output):
            # Ne pas faire .detach() pour garder les gradients !
            self.activations[name] = output.clone()
        return hook
    
    def register_hook(self, model, layer_name):
        """
        Enregistre un hook sur une couche spécifique
        
        Args:
            model: Le modèle SiT
            layer_name: Nom de la couche (ex: 'blocks.8')
        """
        for name, module in model.named_modules():
            if name == layer_name:
                print(f"[DEBUG] Hooking layer: {name}, type: {type(module)}")
                hook = module.register_forward_hook(self.get_activation_hook(name))
                self.hooks.append(hook)
                print(f"✅ Hook registered on {name} with gradients preserved")
                return
        raise ValueError(f"Layer {layer_name} not found in model")
    
    def clear_activations(self):
        """Vide les activations mais garde les hooks"""
        self.activations = {}
    
    def remove_hooks(self):
        """Supprime tous les hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)

def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument("--sampling-method", type=str, default="dopri5", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")

def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument("--diffusion-form", type=str, default="sigma", \
                        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],\
                        help="form of diffusion coefficient in the SDE")
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument("--last-step", type=none_or_str, default="Mean", choices=[None, "Mean", "Tweedie", "Euler"],\
                        help="form of last step taken in the SDE")
    group.add_argument("--last-step-size", type=float, default=0.04, \
                        help="size of the last step taken")

def init_wandb(cfg, rank, logger = None):
    """
    Initialisation de wandb (seulement sur le processus principal)
    """
    use_wandb = getattr(cfg, 'wandb', False)
    if rank == 0 and use_wandb:
        try:
            if hasattr(cfg, "wandb_api_key") and cfg.wandb_api_key and cfg.wandb_api_key != "<TON_API_KEY_WANDB>":
                os.environ["WANDB_API_KEY"] = cfg.wandb_api_key
            
            project_name = getattr(cfg, 'wandb_project', getattr(cfg, 'project_name', 'dit-diffusion-anchoring'))
            run_name = getattr(cfg, 'wandb_run_name', getattr(cfg, 'run_name', 'ddp-native-test'))
            
            wandb.init(project=project_name, name=run_name, config=dict(cfg))
            if logger:
                logger.info(f"✅ Wandb initialisé - Projet: {project_name}, Run: {run_name}")
            print(f"✅ Wandb initialisé - Projet: {project_name}, Run: {run_name}")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Erreur lors de l'initialisation de Wandb: {e}")
            print(f"⚠️ Erreur lors de l'initialisation de Wandb: {e}")
            print("🔄 Continuer sans Wandb")

def get_layer_by_name(model, layer_name):
    """
    Récupère une couche du modèle par son nom.
    Utilise la notation avec points (ex: 'blocks.8' ou 'final_layer')
    """
    parts = layer_name.split('.')
    current = model
    
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        elif part.isdigit() and hasattr(current, '__getitem__'):
            # Si c'est un nombre et que l'objet supporte l'indexation
            current = current[int(part)]
        else:
            raise AttributeError(f"Layer '{layer_name}' not found in model. Part '{part}' not found in {type(current)}")
    
    return current