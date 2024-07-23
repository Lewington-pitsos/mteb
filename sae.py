from numpy import memmap as memmap
import shelve
import torch
import numpy as np
from typing import Any
from transformers import AutoTokenizer
from sae_lens import HookedSAETransformer, SAE
from tqdm import tqdm
import os
from mteb.model_meta import ModelMeta


class SAEncoder():
    def __init__(self, transformer_name, sae_model, sae_id, max_sequence_length, device, batch_size=32, use_cache=False, cache_base=None) -> None:
        self.transformer = HookedSAETransformer.from_pretrained(transformer_name, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_sequence_length = max_sequence_length

        self.sae, _, _ = SAE.from_pretrained(
            release = sae_model, # see other options in sae_lens/pretrained_saes.yaml
            sae_id = sae_id, 
            device = device
        )

        self.batch_size = batch_size

        self.device = device
        self.means = []
        self.stds = []

        self.use_cache = use_cache
        if self.use_cache:
            self.cache = Cache(base_file=cache_base)
            self.cache.clear()

        self.mteb_model_meta = ModelMeta(
            name = 'SAENcoder',
            revision = 'maskfeatures',
            languages=None,
            release_date=None
        )

                
    def encode(
        self, sentences: list[str], **kwargs: Any
    ) -> torch.Tensor | np.ndarray:    
        if len(sentences) > self.batch_size:
            batches = [sentences[i:i + self.batch_size] for i in range(0, len(sentences), self.batch_size)]
        else:
            batches = [sentences]

        all_fts = []
        with torch.no_grad():
            for batch in tqdm(batches, desc='Encoding sentences'):
                features = None
                if self.use_cache:
                    activations, got_all = self.cache.get(batch)

                    if got_all:
                        features = activations

                if features is None:
                    output = self.tokenizer(batch, padding='longest', truncation=True, max_length=self.max_sequence_length,  return_tensors='pt')

                    input_ids = output['input_ids']
                    attention_mask = output['attention_mask']

                    _, all_hidden_states = self.transformer.run_with_cache(
                        input_ids, 
                        attention_mask=attention_mask, 
                        prepend_bos=True, 
                        stop_at_layer=self.sae.cfg.hook_layer + 1
                    )

                    hidden_states = all_hidden_states[self.sae.cfg.hook_name]

                    features = self.sae.encode(hidden_states) * attention_mask.unsqueeze(-1)

                    seq_lens = torch.sum(attention_mask, dim=1)
                    features = torch.sum(features, dim=1) / seq_lens.unsqueeze(-1)
                    if self.use_cache:
                        self.cache.add(batch, features)

                self.means.append(torch.mean(features).item())
                self.stds.append(torch.std(features).item())
                all_fts.append(features)

        final_output = torch.tensor(torch.cat(all_fts, dim=0))

        assert final_output.shape[0] == len(sentences)

        return final_output
    

class Cache():
    def __init__(self, base_file=None) -> None:
        self.cache_file = 'cache.db' if base_file is None else base_file + 'cache.db'
        self.memmap_file = 'activations.dat' if base_file is None else base_file + 'activations.dat'
        self.memmap_dtype = np.float32
        self.memmap = None
        self.memmap_id = 0
        self.activations_shape = None  # Will be set when the first activations are added

    def _initialize_memmap(self, activations_shape):
        if self.memmap is None:
            initial_size = 1000  # Start with space for 1000 activations
            self.memmap_shape = (initial_size,) + activations_shape
            self.memmap = np.memmap(self.memmap_file, dtype=self.memmap_dtype, mode='w+', shape=self.memmap_shape)
            self.activations_shape = activations_shape

    def _resize_memmap(self, new_size):
        temp_filename = self.memmap_file + '.tmp'
        temp_memmap = np.memmap(temp_filename, dtype=self.memmap_dtype, mode='w+', shape=(new_size,) + self.activations_shape)
        temp_memmap[:self.memmap_id] = self.memmap[:self.memmap_id]
        self.memmap._mmap.close()
        os.remove(self.memmap_file)
        os.rename(temp_filename, self.memmap_file)
        self.memmap = temp_memmap

    def add(self, sentences, activations):
        if self.activations_shape is None:
            self._initialize_memmap(activations.shape[1:])

        num_activations = len(activations)
        required_size = self.memmap_id + num_activations

        if required_size > self.memmap.shape[0]:
            self._resize_memmap(required_size * 2)  # Double the size to reduce the frequency of resizing

        self.memmap[self.memmap_id:self.memmap_id + num_activations] = activations
        self.memmap.flush()

        with shelve.open(self.cache_file) as db:
            for i, text in enumerate(sentences):
                db[text] = self.memmap_id + i

        self.memmap_id += num_activations

    def get(self, sentences):
        with shelve.open(self.cache_file) as db:
            activations = []
            got_all = True
            for sent in sentences:
                if sent in db:
                    activations.append(self.memmap[db[sent]])
                else:
                    return [], False

        return torch.tensor(np.array(activations)), got_all

    def clear(self):
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        if os.path.exists(self.memmap_file):
            os.remove(self.memmap_file)
        self.memmap = None
        self.memmap_id = 0
        self.activations_shape = None