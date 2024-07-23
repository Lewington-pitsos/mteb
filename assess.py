from datasets import load_dataset
from sae import SAEncoder
import mteb
import json

model = SAEncoder('gpt2', 'gpt2-small-res-jb', "blocks.10.hook_resid_pre", 1024, 'cpu', batch_size=32, use_cache=True)

tasks = mteb.get_tasks(tasks=[
    # "Banking77Classification"
    "EmotionClassification"
])
evaluation = mteb.MTEB(tasks=tasks)
evaluation.run(model)

with open('stats2.json', 'w') as f:
    json.dump({'means': model.means, 'stds': model.stds}, f)