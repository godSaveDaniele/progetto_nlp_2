from contextlib import contextmanager
from typing import Any

from torch import Tensor, nn
from transformers import PreTrainedModel

@contextmanager
def collect_activations(
    model: PreTrainedModel, hookpoint: str,
):
    
    activations = {}
    handle = None

    def create_hook(hookpoint: str):
        def hook_fn(module: nn.Module, input: Any, output: Tensor) -> Tensor | None:
            # If output is a tuple (like in some transformer layers), take first element
            
            if isinstance(output, tuple):
                activations[hookpoint] = output[0]
            else:
                activations[hookpoint] = output

        return hook_fn

    for name, module in model.named_modules():
        if name == hookpoint:
            
            # registra un hook di tipo "forward" per ogni modulo del modello il cui nome compare nella lista "hookpoints" passata in input
            # di fatto ogni volta che il modello esegue una fase di forward, l'output dei moduli interessati viene intercettato e memorizzato in activations[hookpoint]
            handle = module.register_forward_hook(create_hook(name)) 

    try:
        yield activations
    finally:
        if handle is not None:
            handle.remove()