# SnaPy
[![Build Status](https://travis-ci.com/justinnbt/SnaPy.svg?branch=master)](https://travis-ci.com/justinnbt/SnaPy)
[![PyPI version](https://badge.fury.io/py/snapy.svg)](https://badge.fury.io/py/snapy)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
<br>
Python library for detecting near duplicate texts in a corpus at scale using Locality Sensitive Hashing.<br>
As described in Mining Massive Datasets http://infolab.stanford.edu/~ullman/mmds/ch3.pdf.

### Installation
Install SnaPy using: `pip install snapy`<br>
Install mmh3 library needed for Minhash using: `pip install mmh3`

### Usage Example
``` python
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

# Create MinHash object.
minhash = MinHash(content, char_n_gram=9, permutations=100, hash_bits=64, method='universal', seed=3)

# Create LSH model.
lsh = LSH(minhash, labels, no_of_bands=50)

# Query to find near duplicates for text 1.
print(lsh.query(1, sensitivity=8))
>>> [8, 4]

```
### Contributing
Contributions are very welcome, message us or just submit a pull request!

### Authors
Justin Boylan-Toomey

### License
MIT License
