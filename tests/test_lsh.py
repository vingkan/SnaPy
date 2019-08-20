import pytest
from snapy import MinHash, LSH
import numpy as np
from collections import defaultdict

seed = 3
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
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

minhash = MinHash(content)


def test_empty_lsh():
    lsh = LSH()
    assert lsh.no_of_bands is None
    assert lsh._buckets == defaultdict(list)
    assert lsh._i_bucket == defaultdict(list)


def 

