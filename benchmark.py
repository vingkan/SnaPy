import gc
import timeit
import numpy as np
from snapy import MinHash


def make_test_data(n_samples):
    '''
    Beforing running benchmarks, download and extract the New York Times News
    Articles dataset from the Bag of Words collection.

    Link: https://archive.ics.uci.edu/ml/datasets/Bag+of+Words
    Source: David Newman, University of California, Irvine
    Citation: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
        [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
        School of Information and Computer Science.
    '''
    # Read vocabulary
    vocab_full = {}
    with open('data/vocab.nytimes.txt') as file_vocab:
        line = file_vocab.readline()
        i = 0
        while line:
            vocab_full[i] = line
            i += 1
            line = file_vocab.readline()
    # Read documents
    docs = []
    vocab_corpus = set()
    nonzero_entries = 0
    with open('data/docword.nytimes.txt') as file_docword:
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
    return docs


def run_minhash(docs, kwargs):
    minhash = MinHash(docs, **kwargs)


def time_minhash(n_samples, n_runs, minhash_kwargs):
    secs = timeit.timeit(
        'run_minhash(docs, minhash_kwargs)',
        setup='gc.enable();',
        number=n_runs,
        globals={
            'gc': gc,
            'docs': make_test_data(n_samples),
            'minhash_kwargs': minhash_kwargs,
            'run_minhash': run_minhash
        }
    )
    mean_secs = secs / n_runs
    return mean_secs


n_runs = 1
dataset_sizes = [10, 100, 1000]
permutation_values = [10, 100, 1000, 10_000]

n_gram = 9
hash_size = 64
seed = 12

print("Dataset: New York Times News Articles")
print("\tTrials = {}".format(n_runs))
print("\tN-Grams Size = {}".format(n_gram))
print("\tHash Size = {}".format(hash_size))
print()

for n_permutations in permutation_values:
    print("Multi Hash, Permutations = {:,d}".format(n_permutations))
    for n_samples in dataset_sizes:
        mean_secs = time_minhash(
            n_samples=n_samples,
            n_runs=n_runs,
            minhash_kwargs={
                'hash_bits': hash_size,
                'permutations': n_permutations,
                'n_gram': n_gram,
                'seed': seed,
                'method': 'multi_hash'
            }
        )
        print("\tMinHash(N = {:,d}) = {:.3f} secs".format(n_samples, mean_secs))
    print()

n_permutations = 9
print("K-Smallest Values, Permutations = {:,d}".format(n_permutations))
for n_samples in dataset_sizes:
    mean_secs = time_minhash(
        n_samples=n_samples,
        n_runs=n_runs,
        minhash_kwargs={
            'hash_bits': hash_size,
            'permutations': n_permutations,
            'n_gram': n_gram,
            'seed': seed,
            'method': 'k_smallest_values'
        }
    )
    print("\tMinHash(N = {:,d}) = {:.3f} secs".format(n_samples, mean_secs))
print()
