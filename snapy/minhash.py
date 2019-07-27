# Class for generating a minhash matrix from a text corpus.
# Author: Justin Boylan-Toomey

import numpy as np
import mmh3
import heapq


class MinHash:
    def __init__(self, text, char_n_gram=9, permutations=100, hash_bits=64, method='multi_hash', seed=None):
        """ Generates a minhash signature matrix for texts in a corpus.

        Args:
            text (list, np.array): Iterable containing text content of each document.
            char_n_gram (int): Number of characters to be used in each shingle.
            permutations (int): Number of hash values in each document signature.
            hash_bits (int): Hash value size, must be 32, 64 or 128 bit.
            method (str): Method to be used for minhash function, must be universal
                or k_smallest_values.
            seed (int): Seeds from which to generate random hash function.
        """
        self.char_n_gram = char_n_gram
        self.permutations = permutations
        if hash_bits not in [32, 64, 128]:
            raise ValueError('Only 32, 64 and 128 bit hashes are supported.')
        self.hash_bits = hash_bits
        if method not in ['universal', 'multi_hash', 'k_smallest_values', 'smallest_values']:
            raise ValueError('Only universal and k smallest value hash methods are supported.')
        self.method = method
        if seed:
            np.random.seed(seed)
        if method in ['universal', 'multi_hash']:
            self.hash_seeds = np.random.randint(low=1, high=100_000_000, size=permutations)
        else:
            self.hash_seeds = np.random.randint(low=1, high=100_000_000)
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
        end = (self.char_n_gram - 1) * -1
        if type(texts) == str:
            texts = [texts]
        for text in texts:
            shingles = [text[char:char + self.char_n_gram] for char in range(len(text))][:end]
            if not shingles:
                raise ValueError('Character n grams exceed text length.')
            yield shingles

    def _universal_hash(self, document):
        """ Generates a texts minhash signature using universal method.

        Args:
            document (list): List of document shingles.

        Returns:
            list: Minhash signature.
        """
        signature = []
        for seed in np.nditer(self.hash_seeds):
            self._min_value = None
            if self.hash_bits == 64:
                for shingle in document:
                    hash_value = mmh3.hash64(shingle, int(seed))[0]
                    if not self._min_value:
                        self._min_value = hash_value
                    elif self._min_value > hash_value:
                        self._min_value = hash_value
                signature.append(self._min_value)
            elif self.hash_bits == 32:
                for shingle in document:
                    hash_value = mmh3.hash(shingle, int(seed))
                    if not self._min_value:
                        self._min_value = hash_value
                    elif self._min_value > hash_value:
                        self._min_value = hash_value
                signature.append(self._min_value)
            elif self.hash_bits == 128:
                for shingle in document:
                    hash_value = mmh3.hash128(shingle, int(seed))
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
            raise ValueError('N permutations must not be >= n shingles for k smallest value method')
        if self.hash_bits == 64:
            for shingle in document:
                heapq.heappush(signature, mmh3.hash64(shingle, self.hash_seeds)[0])
            return heapq.nsmallest(self.permutations, signature)
        if self.hash_bits == 32:
            for shingle in document:
                heapq.heappush(signature, mmh3.hash(shingle, self.hash_seeds))
            return heapq.nsmallest(self.permutations, signature)
        if self.hash_bits == 128:
            for shingle in document:
                heapq.heappush(signature, mmh3.hash128(shingle, self.hash_seeds))
            return heapq.nsmallest(self.permutations, signature)

    def _min_hash(self):
        """ Calculates document signature by calling the selected hashing method.

        Returns:
             np.array: Minhash signature matrix.
        """
        signatures = []
        for document in self._shingles:
            if self.method is 'universal':
                signature = self._universal_hash(document)
                signatures.append(signature)
            elif self.method is 'k_smallest_values':
                signature = self._k_smallest_hash(document)
                signatures.append(signature)
        return np.array(signatures)
