import torch
from sae import SAEncoder
import pytest

@pytest.fixture(scope='session')
def saencoder():
    model = SAEncoder('gpt2', 'gpt2-small-res-jb', "blocks.10.hook_resid_pre", 1024, 'cpu')
    return model



def test_saencoder(saencoder):
    sentences = ["Hello, my dog is cute", "Hello, my cat is cute"]

    embeddings = saencoder.encode(sentences)

    assert embeddings.shape == (2, 24576), embeddings.shape

def test_batch_is_deterministic(saencoder):
    dog_sentence = "Hello, my dog is cute"
    cat_sentence = "Hello, my cat is really cute"

    sentences = [dog_sentence, cat_sentence]
    embeddings = saencoder.encode(sentences)
    emb1 = embeddings[0]

    sentences = [dog_sentence, cat_sentence]
    embeddings = saencoder.encode(sentences)

    emb2 = embeddings[0]

    assert emb1.shape == emb2.shape, emb1.shape
    assert torch.allclose(emb1, emb2), f'{torch.mean(emb1).item()} vs {torch.mean(emb2).item()}'

def test_means_not_effected_by_padding(saencoder):
    dog_sentence = "Hello, my dog is cute"

    sentences = [dog_sentence, "Hello, my cat is really cute"]
    embeddings = saencoder.encode(sentences)
    emb1 = embeddings[0]

    sentences = [dog_sentence, "Hello, my cat is cute, long long long long long long sentence"]
    embeddings = saencoder.encode(sentences)
    emb2 = embeddings[0]

    assert emb1.shape == emb2.shape, emb1.shape
    assert torch.allclose(emb1, emb2, 1e-3), f'{torch.mean(emb1).item()} vs {torch.mean(emb2).item()}'
