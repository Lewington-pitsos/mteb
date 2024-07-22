import torch
import numpy as np
from typing import Any
from transformers import AutoTokenizer
from sae_lens import HookedSAETransformer, SAE
from tqdm import tqdm
import os

class SAEncoder():
    def __init__(self, transformer_name, sae_model, sae_id, max_sequence_length, device, batch_size=32, use_cache=False) -> None:
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
            self.cache = 'cache'

            if os.path.exists(self.cache):
                for file in os.listdir(self.cache):
                    os.remove(os.path.join(self.cache, file))
                os.rmdir(self.cache)
                
    def encode(
        self, sentences: list[str], **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        
        if self.use_cache:
            current_cache = self.cache + '/' + f'{sentences[0][:20].replace(" ", "_")}_{sentences[-1][-20:].replace(" ", "_")}.pt'
            if os.path.exists(current_cache):
                return torch.load(current_cache)


        if len(sentences) > self.batch_size:
            batches = [sentences[i:i + self.batch_size] for i in range(0, len(sentences), self.batch_size)]
        else:
            batches = [sentences]

        all_fts = []
        with torch.no_grad():
            for batch in tqdm(batches):
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

                self.means.append(torch.mean(features).item())
                self.stds.append(torch.std(features).item())
                all_fts.append(features)

        final_output = torch.tensor(torch.cat(all_fts, dim=0))

        assert final_output.shape[0] == len(sentences)

        if self.use_cache:
            if not os.path.exists(self.cache):
                os.makedirs(self.cache)
            torch.save(final_output, current_cache)

        return final_output