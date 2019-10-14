import time
import numpy as np
from snapy import MinHash


n_samples = 10000
n_permutations = 100
n_gram = 9
hash_size = 64
seed = 12

vocab_full = {}
with open("data/vocab.nytimes.txt") as file_vocab:
    line = file_vocab.readline()
    i = 0
    while line:
        vocab_full[i] = line
        i += 1
        line = file_vocab.readline()

docs = []
vocab_corpus = set()
nonzero_entries = 0
with open("data/docword.nytimes.txt") as file_docword:
    line = file_docword.readline()
    n_read = 0
    while line and n_read < n_samples:
        ids = [int(d) for d in line.split()]
        ids = list(filter(lambda d: d in vocab_full, ids))
        words = [vocab_full[i] for i in ids]
        doc = " ".join(words)
        if len(doc) >= n_gram:
            n_read += 1
            nonzero_entries += len(ids)
            vocab_corpus.update(ids)
            docs.append(doc)
        line = file_docword.readline()


print("D = {:,d} documents".format(len(docs)))
print("W = {:,d} possible words".format(len(vocab_full)))
print("V = {:,d} actual words".format(len(vocab_corpus)))
print("Z = {:,d} nonzero entries".format(nonzero_entries))
print("P = {:,d} permutations".format(n_permutations))
print("")

# SMH
s = time.time()
smh_minhash = MinHash(
    docs,
    hash_bits=hash_size,
    permutations=n_permutations,
    n_gram=n_gram,
    seed=seed,
    method="sparse_multi_hash",
)
d = time.time()
print("{:.3f} secs".format(d - s))
assert smh_minhash.seed == seed
assert smh_minhash.method == "sparse_multi_hash"
assert type(smh_minhash.signatures) is np.ndarray
assert smh_minhash.signatures.shape == (len(docs), n_permutations)

# RMH
s = time.time()
minhash = MinHash(
    docs,
    hash_bits=hash_size,
    permutations=n_permutations,
    n_gram=n_gram,
    seed=seed,
    method="multi_hash",
)
d = time.time()
print("{:.3f} secs".format(d - s))
assert minhash.seed == seed
assert minhash.method == "multi_hash"
assert type(minhash.signatures) is np.ndarray
assert minhash.signatures.shape == (len(docs), n_permutations)

assert smh_minhash.signatures[0][0] == minhash.signatures[0][0]
assert smh_minhash.signatures[-1][-1] == minhash.signatures[-1][-1]
