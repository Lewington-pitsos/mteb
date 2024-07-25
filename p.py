import time
import torch
from sae_lens import HookedSAETransformer, SAE

MODEL_ID = 'gpt2'
MODEL_NAME = 'gpt2-small-res-jb'
SAE_ID = 'blocks.10.hook_resid_pre'

sae, _, _ = SAE.from_pretrained(
    release = MODEL_NAME, # see other options in sae_lens/pretrained_saes.yaml
    sae_id = SAE_ID, 
    device='cuda'
)


bs = [1, 4, 8, 16, 32, 64, 128]
times = []
for i in bs:
    print(i)
    t = torch.rand(i, 1024, 768, device='cuda')



    start = time.time()
    for i in range(10):
        y = sae.forward(t)
        # zero the gradient
        
    end = time.time()
    times.append(end - start)

print(bs)
print(times)


