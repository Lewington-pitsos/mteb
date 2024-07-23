from datasets import load_dataset
from sae import SAEncoder
import mteb
import json
from sentence_transformers import SentenceTransformer

model = SAEncoder('gpt2', 'gpt2-small-res-jb', "blocks.10.hook_resid_pre", 1024, 'cpu', batch_size=32, use_cache=True)

# model_name = "sentence-transformers/all-MiniLM-L6-v2"
# model = SentenceTransformer(model_name)

tasks = mteb.get_tasks(tasks=[
    # "Banking77Classification"
    "EmotionClassification"
])
evaluation = mteb.MTEB(tasks=tasks)
evaluation.run(model)
