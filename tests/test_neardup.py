import pytest
import sys
import pickle
import os
sys.path.append("..")
from jaccardupy import MinHash, LSH

content = [
    "It is carrying instruments to analyse the unexplored region's geology",
    "The landing is being seen as a major milestone in space exploration.",
    "There have been numerous missions to the Moon in recent years, but the vast majority have been to orbit.",
    "It is carrying instruments to analyse unexplored region's geology",
    "The landing is being seen as a major milestone in space exploration.",
    "There have been numerous missions at the Moon in recent years, but the vast majority have been to orbit.",
    "It is carrying instruments to analyse the unexplored region's geology",
    "The landing is being seen as major milestone in space exploration.",
    "There have been numerous missions to the Moon in recent years, but in the vast majority have been to orbit."
]

labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]

update_content = ["The landing is being seen as major milestones in space exploration."]
update_label = 10

minhash_uni = None
lsh = LSH()


def test_minhash_uni():
    global minhash_uni
    minhash_uni = MinHash(content, char_n_gram=9, permutations=100, hash_bits=64, method='universal', seed=3)
    assert minhash_uni.signatures.shape == (9, 100)
    assert minhash_uni.permutations == 100
    assert minhash_uni.char_n_gram == 9


def test_minhash_k_smallest():
    minhash = MinHash(content, char_n_gram=2, permutations=50, hash_bits=64, method='k_smallest_values', seed=3)
    assert minhash.signatures.shape == (9, 50)
    assert minhash.permutations == 50
    assert minhash.char_n_gram == 2


def test_lsh_initialization():
    assert lsh.use_jaccard is False
    lsh.update(minhash_uni, labels)
    assert lsh.signatures is None
    assert sorted(lsh.contains()) == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_lsh_query():
    assert sorted(lsh.query(1, sensitivity=8)) == [4, 7]
    with pytest.raises(ValueError):
        lsh.query(100, sensitivity=8)
    with pytest.raises(ValueError):
        lsh.query(1, min_jaccard=0.8)
    with pytest.raises(ValueError):
        lsh.query(1,  sensitivity=60)


def test_lsh_update():
    update_hash = MinHash(update_content, char_n_gram=9, permutations=100, hash_bits=64, method='universal', seed=3)
    lsh.update(update_hash, [update_label])
    assert lsh.signatures is None
    assert sorted(lsh.contains()) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert sorted(lsh.query(10, sensitivity=8)) == [2, 5, 8]
    with pytest.raises(ValueError):
        lsh.update(update_hash, [10])
    with pytest.raises(ValueError):
        update_hash.permutations = 12
        lsh.update(update_hash, [11])
    update_hash.permutations = 100


def test_lsh_remove():
    lsh.remove(7)
    with pytest.raises(ValueError):
        lsh.remove(7)
    assert sorted(lsh.contains()) == [1, 2, 3, 4, 5, 6, 8, 9, 10]
    with pytest.raises(ValueError):
        lsh.query(7)


def test_lsh_pickle():
    global lsh
    lsh = LSH(minhash_uni, labels)
    pickle_file = open('test_pickle.p', 'wb')
    pickle.dump(lsh, pickle_file)
    pickle_file.close()
    pickle_file = open('test_pickle.p', 'rb')
    pickle_lsh = pickle.load(pickle_file)
    pickle_file.close()
    os.remove('test_pickle.p')
    assert pickle_lsh._buckets == lsh._buckets
    assert pickle_lsh._i_bucket == lsh._i_bucket


def test_label():
    assert len(labels) == 9


minhash = MinHash(content, char_n_gram=9, permutations=100, hash_bits=64, method='universal', seed=3)
lsh_init_jaccard = LSH(minhash, labels=labels, use_jaccard=True)


def test_lsh_jaccard_initialization():
    assert lsh_init_jaccard.use_jaccard is True
    assert lsh_init_jaccard.permutations == 100
    assert lsh_init_jaccard.no_of_bands == 50
    signatures = lsh_init_jaccard.signatures
    assert signatures is not None
    assert sorted(signatures.keys()) == labels
    with pytest.raises(ValueError):
        LSH(labels=labels, use_jaccard=True)
    with pytest.raises(ValueError):
        LSH(minhash, use_jaccard=True)


def test_lsh_jaccard_query():
    assert sorted(lsh_init_jaccard.query(1, sensitivity=8)) == [4, 7]
    assert sorted(lsh_init_jaccard.query(1, min_jaccard=0.4)) == [4, 7]
    assert sorted(lsh_init_jaccard.query(1, sensitivity=2,  min_jaccard=0.4)) == [4, 7]
    assert sorted(lsh_init_jaccard.query(1, sensitivity=30, min_jaccard=0.4)) == [7]
    with pytest.raises(ValueError):
        lsh_init_jaccard.query(100, min_jaccard=0.4)


def test_lsh_jaccard_update():
    update_hash = MinHash(update_content, char_n_gram=9, permutations=100, hash_bits=64, method='universal', seed=3)
    lsh_init_jaccard.update(update_hash, [update_label])
    assert lsh_init_jaccard.use_jaccard is True
    assert sorted(lsh_init_jaccard.contains()) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert sorted(lsh_init_jaccard.query(10, sensitivity=8)) == [2, 5, 8]
    assert sorted(lsh_init_jaccard.signatures.keys()) == labels + [update_label]


def test_lsh_jaccard_remove():
    lsh_init_jaccard.remove(10)
    assert lsh_init_jaccard.use_jaccard is True
    assert sorted(lsh_init_jaccard.contains()) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    with pytest.raises(ValueError):
        lsh_init_jaccard.query(10, min_jaccard=0.4)
    assert sorted(lsh_init_jaccard.signatures.keys()) == labels
