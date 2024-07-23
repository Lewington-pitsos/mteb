import time
import os
import torch
from sae import SAEncoder, Cache
import pytest

@pytest.fixture(scope='session')
def saencoder():
    model = SAEncoder('gpt2', 'gpt2-small-res-jb', "blocks.10.hook_resid_pre", 1024, 'cpu')
    return model

@pytest.fixture()
def testing_cache():
    cache = Cache('test_cache.db')
    yield cache

    cache.clear()

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

def test_model_caching():
    model = SAEncoder('gpt2', 'gpt2-small-res-jb', "blocks.10.hook_resid_pre", 1024, 'cpu', use_cache=True, cache_base='test_cache_')
    n=256
    sentences = [f"Hello, my dog is cute, this is a long sentence a really really long sentence indeed ypu got no idea buddy how god damn long of a sentence this is my gum ohhhhh baby this sentence you got no clue {i}" for i in range(n)]

    start = time.time()
    embeddings1 = model.encode(sentences)
    end = time.time()

    elapsed = end - start

    start = time.time()
    embeddings2 = model.encode(sentences)
    end = time.time()

    elapsed2 = end - start
    print(elapsed2)

    assert elapsed2 < elapsed, f"Second time elapsed: {elapsed2}, first time elapsed: {elapsed}"

    assert torch.allclose(embeddings1, embeddings2)


    model.cache.clear()

def test_cache(testing_cache):
    sentences = ["Hello, my dog is cute", "Hello, my cat is cute"]
    activations, got_all = testing_cache.get(sentences)
    assert not got_all, got_all

    testing_cache.add([sentences[0]], torch.tensor([[1, 2, 3]]))

    activations, got_all = testing_cache.get(sentences)
    assert not got_all, got_all


    testing_cache.add([sentences[1]], torch.tensor([[4, 5, 6]]))
    activations, got_all = testing_cache.get(sentences)
    assert got_all, got_all

    assert isinstance(activations, torch.Tensor)

    assert torch.allclose(activations[0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(activations[1], torch.tensor([4.0, 5.0, 6.0]))

def test_fast_writing(testing_cache):
    n=1000
    sentences = [f"Hello, my dog is cute {i} llll" for i in range(n)]

    tensors = (torch.rand(n, 25000) > 1.5) * 1

    start = time.time()
    testing_cache.add(sentences, tensors)
    activations, got_all = testing_cache.get(sentences)
    end = time.time()

    elapsed = end - start
    print(elapsed)
    assert elapsed < 0.5, f"Time elapsed: {elapsed}"
