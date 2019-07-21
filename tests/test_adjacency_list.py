import pytest
from snapy import MinHash, LSH

content = [
    'Jupiter is primarily composed of hydrogen with a quarter of its mass being helium',
    'Jupiter moving out of the inner Solar System would have allowed the formation of inner planets.',
    'A helium atom has about four times as much mass as a hydrogen atom, so the composition changes '
    'when described as the proportion of mass contributed by different atoms.',
    'Jupiter is primarily composed of hydrogen and a quarter of its mass being helium',
    'A helium atom has about four times as much mass as a hydrogen atom and the composition changes '
    'when described as a proportion of mass contributed by different atoms.',
    'Theoretical models indicate that if Jupiter had much more mass than it does at present, it would shrink.',
    'This process causes Jupiter to shrink by about 2 cm each year.',
    'Jupiter is mostly composed of hydrogen with a quarter of its mass being helium',
    'The Great Red Spot is large enough to accommodate Earth within its boundaries.'
]
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
seed = 3


def test_lsh_adjacency_list():
    minhash = MinHash(content, char_n_gram=9, permutations=100, hash_bits=64, method='universal', seed=3)
    lsh = LSH(minhash, labels, no_of_bands=50, use_jaccard=True)

    adj_list = lsh.adjacency_list(sensitivity=10)
    assert type(adj_list) is dict
    assert adj_list == {
        1: [8, 4],
        2: [],
        3: [5],
        4: [1, 8],
        5: [3],
        6: [],
        7: [],
        8: [1, 4],
        9: []
    }

    adj_list = lsh.adjacency_list(jaccard=0.5)
    assert type(adj_list) is dict
    assert adj_list == {
        1: [8, 4],
        2: [],
        3: [5],
        4: [1],
        5: [3],
        6: [],
        7: [],
        8: [1],
        9: []
    }

    adj_list = lsh.adjacency_list(jaccard=0.5, keep_jaccard=True)
    assert type(adj_list) is dict
    assert adj_list == {
        1: [(0.5267175572519084, 8), (0.6129032258064516, 4)],
        2: [],
        3: [(0.639344262295082, 5)],
        4: [(0.6129032258064516, 1)],
        5: [(0.639344262295082, 3)],
        6: [],
        7: [],
        8: [(0.5267175572519084, 1)],
        9: []
    }

    adj_list = lsh.adjacency_list(jaccard=0.5, keep_jaccard=False, average_jaccard=True)
    assert type(adj_list) is dict
    assert adj_list == {
        1: (2, 0.56981039152918),
        2: (0, 0),
        3: (1, 0.639344262295082),
        4: (1, 0.6129032258064516),
        5: (1, 0.639344262295082),
        6: (0, 0),
        7: (0, 0),
        8: (1, 0.5267175572519084),
        9: (0, 0)
    }
