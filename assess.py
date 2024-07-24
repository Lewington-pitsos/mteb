import os
from datasets import load_dataset
from sae import SAEncoder
import mteb
import json
from sentence_transformers import SentenceTransformer
import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

with open('.credentials.json') as f:
    creds = json.load(f)
    
os.environ['HF_TOKEN'] = creds['HF_TOKEN']


TASK_LIST = [
    "AmazonCounterfactualClassification",
    # "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",

    # # "ArxivClusteringP2P",
    # # "ArxivClusteringS2S",
    # "BiorxivClusteringP2P",
    # "BiorxivClusteringS2S",
    # "MedrxivClusteringP2P",
    # "MedrxivClusteringS2S",
    # # "RedditClustering",
    # # "RedditClusteringP2P",
    # # "StackExchangeClustering",
    # "StackExchangeClusteringP2P",
    # "TwentyNewsgroupsClustering",


]
tasks = mteb.get_tasks(languages=["eng"], tasks=TASK_LIST)
evaluation = mteb.MTEB(tasks=tasks)

# model = SAEncoder('gpt2', 'gpt2-small-res-jb', "blocks.10.hook_resid_pre", 1024, 'cpu', batch_size=32, use_cache=True)

# MODEL_ID = 'gpt2'
# MODEL_NAME = 'gpt2-small-res-jb'
# SAE_ID = 'blocks.10.hook_resid_pre'
# BATCH_SIZE=32
REVISION='no-error'

MODEL_ID = 'google/gemma-2b-it'
MODEL_NAME = 'gemma-2b-it-res-jb'
SAE_ID = "blocks.12.hook_resid_post"
BATCH_SIZE=8


model = SAEncoder(
    MODEL_ID, 
    MODEL_NAME, 
    SAE_ID, 
    max_sequence_length=1024, 
    batch_size=BATCH_SIZE, 
    use_cache=False,
    device=device,
    revision=REVISION
)

evaluation.run(model)
