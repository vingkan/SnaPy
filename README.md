# SnaPy
[![Build Status](https://travis-ci.com/justinnbt/SnaPy.svg?branch=master)](https://travis-ci.com/justinnbt/SnaPy)
[![PyPI version](https://badge.fury.io/py/snapy.svg)](https://badge.fury.io/py/snapy)
[![Downloads](https://pepy.tech/badge/snapy)](https://pepy.tech/project/snapy)
[![Python Version](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg)](https://pypi.org/project/snapy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
<br>
Python library for detecting near duplicate texts in a corpus at scale using Locality Sensitive Hashing.<br>
As described in Mining Massive Datasets http://infolab.stanford.edu/~ullman/mmds/ch3.pdf.

## Installation
Install SnaPy using: `pip install snapy`<br>
Install mmh3 library needed for Minhash using: `pip install mmh3`

## Quickstart Example
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
minhash = MinHash(content, n_gram=9, permutations=100, hash_bits=64, method='multi_hash', seed=3)

# Create LSH model.
lsh = LSH(minhash, labels, no_of_bands=50)

# Query to find near duplicates for text 1.
print(lsh.query(1, min_jaccard=0.5))
>>> [8, 4]

# Generate minhash signature and add new texts to LSH model.
new_text = [
    'Jupiter is primarily composed of hydrogen with a quarter of its mass being helium',
    'Jupiter moving out of the inner Solar System would have allowed the formation of inner planets.',
]
new_labels = ['doc1', 'doc2']

minhash = MinHash(new_text, n_gram=9, permutations=100, hash_bits=64, method='multi_hash', seed=3)
lsh.update(new_text, new_labels)

# Check contents of documents.
print(lsh.contains())

```
## MinHash
Generates MinHash object that contains matrix of Minhash Signatures for each text.
### Parameters
<b>text: {list or ndarray}</b><br>
Iterable containing text strings.<br><br>
<b>char_n_gram: int, optional, default: 9</b><br>
Number of characters to be used in each shingle.<br><br>
<b>permutations: int, optional, default: 100</b><br>
Size of hash values in each document signature.<br><br>
<b>vhash_bits: int, optional, default: 64</b><br>
Hash value size, must be 32, 64 or 128 bit.<br><br>
<b>method: str, optional, default: 'universal'</b><br>
Method to be used for minhash function, must be universal or k_smallest_values.<br><br>
<b>seed: int, optional, default: None</b><br>
Seeds from which to generate random hash function.<br><br>
### Properties
<b>char_n_gram: int</b><br>
Number of characters used for creating shingles.<br><br>
<b>permutations: int</b><br>
Number of permutations used to create signatures.<br><br>
<b>hash_bits: int</b><br>
Hash value size used to create signatures.<br><br>
<b>method: str</b><br>
Method used in minhash function<br><br>
<b>hash_seeds: ndarray</b><br>
Seeds used for each hash permutation in minhash function.<br><br>
<b>signatures: ndarray</b><br>
Matrix of text signatures generated by minhash function.<br>
n = text row, m = selected permutations.<br><br>

## LSH
LSH model of text similarity.
### Parameters
<b>minhash, optional, default: None</b><br>
Object returned by MinHash class.<br><br>
<b>labels: {list or ndarray}, optional, default: None</b><br>
Iterable containing labels for text in minhash object.<br><br>
<b>no_of_bands: int, optional, default: None</b><br>
Number of bands to break minhash signature into.<br><br>
<b>use_jaccard: bool, optional, default: False</b><br>
Should MinHash signatures be retained for later estimation of Jaccard similarity.<br><br>
### Methods
```.update(minhash, new_labels)```<br>
Updates model with new MinHash signature matrix and labels.<br><br>
```.query(label, sensitivity=1, min_jaccard=None)```<br>
Takes a label and returns the labels of any near duplicate/similar texts.<br><br>
```.def remove(label):```<br>
Remove file label and minhash signature from model.<br><br>
```.contains()```<br>
Returns list of labels contained in the model.<br><br>
```.adjacency_list(sensitivity=1, jaccard=None, keep_jaccard=False, average_jaccard=False)```<br>
Returns an adjacency list that can be used to create a text similarity graph.<br><br>
### Properties
<b>no_of_bands: int</b><br>
Number of bands used in LSH model.<br><br>
<b>permutations: int</b><br>
Number of permutations used to create minhash signatures used in LSH model.<br><br>

## Contributing
Contributions are very welcome, message us or just submit a pull request!

## Authors
Justin Boylan-Toomey

## License
MIT License
