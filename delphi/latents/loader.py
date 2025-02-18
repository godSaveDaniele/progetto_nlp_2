import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import numpy as np
import torch
from jaxtyping import Float
from safetensors.numpy import load_file
from torch import Tensor
from transformers import AutoTokenizer

from delphi.utils import (
    load_tokenized_data,
)

from ..config import LatentConfig
from .latents import Latent, LatentRecord


class ActivationData(NamedTuple):
    """
    Represents the activation data for a latent.
    """

    locations: Float[Tensor, "n_examples 2"]
    """Tensor of latent locations."""

    activations: Float[Tensor, "n_examples"]
    """Tensor of latent activations."""


class LatentData(NamedTuple):
    """
    Represents the output of a TensorBuffer.
    """

    latent: Latent
    """The latent associated with this output."""

    module: str
    """The module associated with this output."""

    activation_data: ActivationData
    """The activation data for this latent."""


@dataclass
class TensorBuffer:
    """
    Lazy loading buffer for cached splits.
    """

    path: str
    """Path to the tensor file."""

    module_path: str
    """Path of the module."""

    latents: Optional[Float[Tensor, "num_latents"]] = None
    """Tensor of latent indices."""

    min_examples: int = 120
    """Minimum number of examples required. Defaults to 120."""

    _tokens: Optional[Float[Tensor, "batch seq"]] = None
    """Tensor of tokens."""

    def __iter__(self):
        """
        Iterate over the buffer, yielding BufferOutput objects.

        Yields:
            Union[BufferOutput, None]: BufferOutput if enough examples,
                None otherwise.
        """
        latents, split_locations, split_activations = self.load_data_per_latent()

        for i in range(len(latents)):
            latent_locations = split_locations[i]
            latent_activations = split_activations[i]
            if len(latent_locations) < self.min_examples:
                yield None
            else:
                yield LatentData(
                    Latent(self.module_path, int(latents[i].item())),
                    self.module_path,
                    ActivationData(latent_locations, latent_activations),
                )

    @property
    def tokens(self) -> Float[Tensor, "batch seq"] | None:
        if self._tokens is None:
            self._tokens = self.load_tokens()
        return self._tokens

    def load_data_per_latent(self):
        locations, activations, _ = self.load()
        indices = torch.argsort(locations[:, 2], stable=True)
        activations = activations[indices]
        locations = locations[indices]
        unique_latents, counts = torch.unique_consecutive(
            locations[:, 2], return_counts=True
        )
        latents = unique_latents
        split_locations = torch.split(locations, counts.tolist())
        split_activations = torch.split(activations, counts.tolist())

        return latents, split_locations, split_activations

    def load(
        self,
    ) -> tuple[
        Float[Tensor, "locations 2"],
        Float[Tensor, "activations"],
        Float[Tensor, "batch seq"] | None,
    ]:
        split_data = load_file(self.path)
        first_latent = int(self.path.split("/")[-1].split("_")[0])
        activations = torch.tensor(split_data["activations"])
        locations = torch.tensor(split_data["locations"].astype(np.int64))
        if "tokens" in split_data:
            tokens = torch.tensor(split_data["tokens"].astype(np.int64))
        else:
            tokens = None

        locations[:, 2] = locations[:, 2] + first_latent

        if self.latents is not None:
            wanted_locations = torch.isin(locations[:, 2], self.latents)
            locations = locations[wanted_locations]
            activations = activations[wanted_locations]

        return locations, activations, tokens

    def load_tokens(self) -> Float[Tensor, "batch seq"] | None:
        _, _, tokens = self.load()
        return tokens


class LatentDataset:
    """
    Dataset which constructs TensorBuffers for each module and latent.
    """

    def __init__(
        self,
        raw_dir: str,
        cfg: LatentConfig,
        tokenizer: Optional[Callable] = None,
        modules: Optional[list[str]] = None,
        latents: Optional[dict[str, torch.Tensor]] = None,
        constructor: Optional[Callable] = None,
        sampler: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize a LatentDataset.

        Args:
            raw_dir: Directory containing raw latent data.
            cfg: Configuration for latent processing.
            modules: list of module names to include.
            latents: Dictionary of latents per module.
        """
        self.cfg = cfg
        self.buffers: list[TensorBuffer] = []
        self.all_data: dict[str, dict[int, ActivationData] | None] = {}
        self.tokens = None

        if modules is None:
            self.modules = os.listdir(raw_dir)
        else:
            self.modules = modules
        if latents is None:
            self._build(raw_dir)
        else:
            self._build_selected(raw_dir, latents)
        # TODO: this assumes that all modules have the same config
        cache_config_dir = f"{raw_dir}/{self.modules[0]}/config.json"
        with open(cache_config_dir, "r") as f:
            cache_config = json.load(f)
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(cache_config["model_name"])
        else:
            self.tokenizer = tokenizer
        self.cache_config = cache_config
        self.constructor = constructor
        self.sampler = sampler
        self.transform = transform

        # TODO: is it possible to do this without loading all data?
        if self.constructor is not None:
            if self.constructor.keywords["constructor_type"] == "neighbours":
                self.all_data = self._load_all_data(raw_dir, self.modules)

        self.load_tokens()

    def load_tokens(self):
        """
        Load tokenized data for the dataset.
        Caches the tokenized data if not already loaded.

        Returns:
            torch.Tensor: The tokenized dataset.
        """
        if not hasattr(self, "tokens"):
            self.tokens = load_tokenized_data(
                self.cache_config["ctx_len"],
                self.tokenizer,  # type: ignore
                self.cache_config["dataset_repo"],
                self.cache_config["dataset_split"],
                self.cache_config["dataset_name"],
                column_name=self.cache_config.get(
                    "dataset_column",
                    self.cache_config.get("dataset_row", "raw_content"),
                ),
            )
        return self.tokens

    def _edges(self, raw_dir: str, module: str) -> list[tuple[int, int]]:
        module_dir = Path(raw_dir) / module
        safetensor_files = [f for f in module_dir.glob("*.safetensors")]
        edges = []
        for file in safetensor_files:
            start, end = file.stem.split("_")
            edges.append((int(start), int(end)))
        edges.sort(key=lambda x: x[0])
        return edges

    def _build(self, raw_dir: str):
        """
        Build dataset buffers which load all cached latents.

        Args:
            raw_dir (str): Directory containing raw latent data.
            modules (Optional[list[str]]): list of module names to include.
        """

        for module in self.modules:
            edges = self._edges(raw_dir, module)
            for start, end in edges:
                path = f"{raw_dir}/{module}/{start}_{end}.safetensors"
                tensor_buffer = TensorBuffer(
                    path, module, min_examples=self.cfg.min_examples
                )
                if self.tokens is None:
                    self.tokens = tensor_buffer.tokens
                self.buffers.append(tensor_buffer)
                self.all_data[module] = None
            self.all_data[module] = None

    def _build_selected(
        self,
        raw_dir: str,
        latents: dict[str, torch.Tensor],
    ):
        """
        Build a dataset buffer which loads only selected latents.

        Args:
            raw_dir (str): Directory containing raw latent data.
            latents (dict[str, Union[int, torch.Tensor]]): Dictionary of latents
                per module.
        """

        for module in self.modules:
            edges = self._edges(raw_dir, module)
            selected_latents = latents[module]
            boundaries = [edges[0][0]] + [edge[1] + 1 for edge in edges]

            bucketized = torch.bucketize(
                selected_latents, torch.tensor(boundaries), right=True
            )
            unique_buckets = torch.unique(bucketized)

            for bucket in unique_buckets:
                mask = bucketized == bucket
                _selected_latents = selected_latents[mask]

                start, end = boundaries[bucket.item() - 1], boundaries[bucket.item()]
                # Adjust end by one as the path avoids overlap
                path = f"{raw_dir}/{module}/{start}_{end-1}.safetensors"
                tensor_buffer = TensorBuffer(
                    path, module, _selected_latents, min_examples=self.cfg.min_examples
                )
                if self.tokens is None:
                    self.tokens = tensor_buffer.tokens
                self.buffers.append(tensor_buffer)
            self.all_data[module] = None

    def __len__(self):
        """Return the number of buffers in the dataset."""
        return len(self.buffers)

    def _load_all_data(self, raw_dir: str, modules: list[str]):
        """For each module, load all locations and activations"""
        all_data = {}
        for buffer in self.buffers:
            module = buffer.module_path
            if module not in all_data:
                all_data[module] = {}
            temp_latents = buffer.latents
            # we remove the filter on latents
            buffer.latents = None
            latents, locations, activations = buffer.load_data_per_latent()
            # we restore the filter on latents
            buffer.latents = temp_latents
            for latent, location, activation in zip(latents, locations, activations):
                all_data[module][latent.item()] = ActivationData(location, activation)
        return all_data

    def __iter__(self):
        """
        Synchronous iterator that wraps the asynchronous iterator.
        Creates a new event loop to drive the async generator.
        """
        # Create a new event loop.
        loop = asyncio.new_event_loop()
        async_gen = self.__aiter__()
        try:
            while True:
                # Retrieve the next item from the asynchronous iterator
                record = loop.run_until_complete(anext(async_gen))
                yield record
        except StopAsyncIteration:
            return
        finally:
            loop.close()

    async def __aiter__(self):
        """Asynchronously iterate over the dataset."""
        for buffer in self.buffers:
            async for record in self._aprocess_buffer(buffer):
                yield record

    async def _aprocess_buffer(self, buffer: TensorBuffer):
        """
        Asynchronously process a buffer.

        Args:
            buffer (TensorBuffer): Buffer to process.

        Yields:
            Optional[LatentRecord]: Processed latent record or None.
        """
        for data in buffer:
            if data is not None:
                record = await self._aprocess_latent(data)
                if record is not None:
                    yield record
            await asyncio.sleep(0)  # Allow other coroutines to run

    async def _aprocess_latent(self, latent_data: LatentData) -> LatentRecord:
        """
        Asynchronously process a single latent.

        Args:
            buffer_output (BufferOutput): Latent data to process.

        Returns:
            Optional[LatentRecord]: Processed latent record or None.
        """
        record = LatentRecord(latent_data.latent)
        if self.transform is not None:
            self.transform(record)
        if self.constructor is not None:
            self.constructor(
                record=record,
                activation_data=latent_data.activation_data,
                all_data=self.all_data[latent_data.module],
                tokens=self.tokens,
            )
        if self.sampler is not None:
            self.sampler(record)
        return record
