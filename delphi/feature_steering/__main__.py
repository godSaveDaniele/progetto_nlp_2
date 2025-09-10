import torch
from transformers import (
    AutoModel,
    PreTrainedModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)
from typing import Callable
from typing import Any
from pathlib import Path
from sparsify import SparseCoder

from functools import partial

from torch import Tensor

from delphi.feature_steering.collect_activations import collect_activations

def run():
    #PARAMETRI 
    dtype = torch.float16 # alternativa "auto"
    load_in_8bit = True

    name_hf_model = "meta-llama/Llama-3.2-1B-Instruct"
    token_hf = "your_hf_token_here"

    path_local_sae = "/path/to/local/sae"
    name_hookpoint = "layer.name"


    #CARICAMENTO DEL MODELLO DA HUGGING FACE
    # Loads a pre-trained model using Hugging Face's AutoModel
    model = AutoModel.from_pretrained(
        name_hf_model,
        device_map={"": "cuda"}, # Places the model on GPU
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=load_in_8bit)
            if load_in_8bit else None ), 
        torch_dtype=dtype,  # Sets the data type for model weights
        token=token_hf,
    )

    #CARICAMENTO MODELLO SAE
    # Loads sparse autoencoders (SAEs) at a specific hookpoint in the model
    hook_fn = load_hooks_sparse_coders(
        model = model,
        path_local_sae = path_local_sae,
        hookpoint = name_hookpoint
    )
    # Initializes the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name_hf_model, token=token_hf)

    process_prompt(model=model, hookpoint=name_hookpoint, hookpoint_fn=hook_fn, tokenizer=tokenizer)

    del model, tokenizer, hook_fn


def process_prompt(
    model: PreTrainedModel,
    hookpoint: str,
    hookpoint_fn: Callable[...,Any],
    tokenizer: PreTrainedTokenizer,
):
    latent_id = -1 # id del latente da boostare
    boost = -1 # quanto boostare il latente
    prompt = "" # prompt da analizzare

    # Tokenizzazione del prompt
    input = tokenizer(
        prompt, 
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length",
    )

    input = {k: v.to(model.device) for k, v in input.items()}

    with torch.no_grad():
        with collect_activations(model, hookpoint) as activations:
            output = model.generate(**input)

            for hookpoint, latents in activations.items():
                sae_latents = hookpoint_fn(latents)





def sae_dense_latents(x: Tensor, sae: SparseCoder) -> Tensor:
    """Run `sae` on `x`, yielding the dense activations."""
    x_in = x.reshape(-1, x.shape[-1]) # si lavora sempre in batch
    encoded = sae.encode(x_in)
    buf = torch.zeros(
        x_in.shape[0], sae.num_latents, dtype=x_in.dtype, device=x_in.device
    )

    # inserisce le attivazioni sparse nel buffer denso (le posizioni non attivate restano a zero)
    buf = buf.scatter_(-1, encoded.top_indices, encoded.top_acts.to(buf.dtype))
    
    return buf.reshape(*x.shape[:-1], -1)


def load_hooks_sparse_coders ( 
    model: PreTrainedModel,
    path_local_sae: str,
    hookpoint: str,
) -> Callable[[Tensor], Tensor]:

    device = model.device or "cpu"

    name_path = Path(path_local_sae)
    if not name_path.exists():
        raise FileNotFoundError(f"SAE model file not found at: {name_path}")

    # Caricamento degli autoencoder sparsi (da path locale oppure scaricandoli da Hugging Face)
 

    """Carica il modello SAE da disco."""
    sparse_model= SparseCoder.load_from_disk(name_path / hookpoint, device=device)

    
    print(f"Resolving path for hookpoint: {hookpoint}")
    
    return partial(
        sae_dense_latents, sae=sparse_model
    )





if __name__ == "__main__":
    run()