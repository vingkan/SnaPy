# Class for generating a minhash matrix from a text corpus.
# Author: Justin Boylan-Toomey

import numpy as np
import mmh3
import heapq


class MinHash:
    def __init__(
            self,
            text,
            n_gram=9,
            n_gram_type='char',
            permutations=100,
            hash_bits=64,
            method='multi_hash',
            seed=None
    ):
        """ Generates a minhash signature matrix for texts in a corpus.

        Args:
            text (list, np.array): Iterable containing text content of each document.
            n_gram (int): Number of characters to be used in each shingle.
            n_gram_type (str): Type of n gram to use for shingles, must be char or term.
            permutations (int): Number of hash values in each document signature.
            hash_bits (int): Hash value size, must be 32, 64 or 128 bit.
            method (str): Method to be used for minhash function, must be multi_hash
                or k_smallest_values.
            seed (int): Seeds from which to generate random hash function.
        """
        self.n_gram = n_gram
        if n_gram_type not in ['char', 'term']:
            raise ValueError(
                'Only "char" and "term" n_gram types are supported.'
            )
        self.n_gram_type = n_gram_type
        self.permutations = permutations
        if hash_bits not in [32, 64, 128]:
            raise ValueError(
                'Only 32, 64 and 128 bit hashes are supported.'
            )
        self.hash_bits = hash_bits
        if method not in [
            'multi_hash',
            'k_smallest_values'
        ]:
            raise ValueError(
                'Only "multi_hash" and "k_smallest_value" hash methods are supported.'
            )
        self.method = method
        if seed:
            np.random.seed(seed)
        if method == 'multi_hash':
            self.hash_seeds = np.random.randint(
                low=1, high=100_000_000, size=permutations
            )
        else:
            self.hash_seeds = np.random.randint(
                low=1, high=100_000_000
            )
        # Run methods.
        self._shingles = self._k_shingles(text)
        self.signatures = self._min_hash()

    def _k_shingles(self, texts):
        """ Break string into k-shingles consisting of k characters and return generator object.

        Args:
            texts (list, array): list of texts contents.

        Yields:
            List: Shingles for each input text.
        """
        trim_overflow = (self.n_gram - 1) * -1
        if type(texts) == str:
            texts = [texts]
        for text in texts:
            if self.n_gram_type == 'char':
                shingles = [
                               text[char:char + self.n_gram]
                               for char in range(len(text))
                           ][:trim_overflow]
            else:
                terms = text.split()
                shingles = [
                               ' '.join(terms[term:term + self.n_gram])
                               for term in range(len(terms))
                           ][:trim_overflow]
            if not shingles:
                raise ValueError(
                    'Shingle "n_gram" size must not exceed minimum text length.'
                )
            yield shingles

    def _multi_hash(self, document):
        """ Generates a texts minhash signature using multi hash method.

        Args:
            document (list): List of document shingles.

        Returns:
            list: Minhash signature.
        """
        signature = []
        for seed in np.nditer(self.hash_seeds):
            self._min_value = None
            for shingle in document:
                if self.hash_bits == 64:
                    hash_value = mmh3.hash64(
                        shingle, int(seed)
                    )[0]
                elif self.hash_bits == 32:
                    hash_value = mmh3.hash(
                        shingle, int(seed)
                    )
                else:
                    hash_value = mmh3.hash128(
                        shingle, int(seed)
                    )
                if not self._min_value:
                    self._min_value = hash_value
                elif self._min_value > hash_value:
                    self._min_value = hash_value
            signature.append(self._min_value)
        return signature

    def _k_smallest_hash(self, document):
        """ Generates a texts minhash signature using k smallest neighbours method.

        Args:
            document (list): List of document shingles.

        Returns:
            list: Minhash signature.
        """
        signature = []
        heapq.heapify(signature)
        if len(document) <= self.permutations:
            raise ValueError(
                'N permutations must not be >= n shingles for k_smallest_values method'
            )
        for shingle in document:
            if self.hash_bits == 64:
                hashed_shingle = mmh3.hash64(
                    shingle, self.hash_seeds
                )[0]
            elif self.hash_bits == 32:
                hashed_shingle = mmh3.hash(
                    shingle, self.hash_seeds
                )
            else:
                hashed_shingle = mmh3.hash128(
                    shingle, self.hash_seeds
                )
            heapq.heappush(signature, hashed_shingle)
        return heapq.nsmallest(self.permutations, signature)

    def _min_hash(self):
        """ Calculates document signature by calling the selected hashing method.

        Returns:
             np.array: Minhash signature matrix.
        """
        signatures = []
        for document in self._shingles:
            if self.method is 'multi_hash':
                signature = self._multi_hash(document)
                signatures.append(signature)
            elif self.method is 'k_smallest_values':
                signature = self._k_smallest_hash(document)
                signatures.append(signature)
        return np.array(signatures)
