import numpy as np
from scipy import sparse

class Batcher(object):
    pass

class WindowedBatcher(object):

    def __init__(self, review_sequence, review_encoding,
                 target_sequence, target_encoding, batch_size=100, sequence_length=50, lengths=None,
                 use_lengths=True):

        self.review_sequence = review_sequence
        self.review_encoding = review_encoding
        self.target_sequence = target_sequence
        self.target_encoding = target_encoding

        self.batch_index = 0
        self.batches = []
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.length = len(self.review_sequence)
        self.lengths = lengths

        self.batch_index = 0
        self.X = self.review_sequence.seq
        self.y = self.target_sequence.seq

        N, D = self.X.shape
        assert N > self.batch_size * self.sequence_length, "File has to be at least %u characters" % (self.batch_size * self.sequence_length)

        self.X = self.X[:N - N % (self.batch_size * self.sequence_length)]
        self.lengths = self.lengths[:N - N % (self.batch_size * self.sequence_length)]
        self.y = self.y[:N - N % (self.batch_size * self.sequence_length)]

        self.N, self.D = self.X.shape
        _, self.Dy = self.y.shape
        self.X = self.X.reshape((self.N / self.sequence_length, self.sequence_length, self.D))
        self.y = self.y.reshape((self.N / self.sequence_length, self.sequence_length, self.Dy))

        self.lengths  = self.lengths.reshape((self.N / self.sequence_length, self.sequence_length))
        self.use_lengths = use_lengths

        self.N, self.S, self.D = self.X.shape

        self.num_sequences = self.N / self.sequence_length
        self.num_batches = self.N / self.batch_size
        self.batch_cache = {}

    def next_batch(self):
        idx = (self.batch_index * self.batch_size)
        if self.batch_index >= self.num_batches:
            self.batch_index = 0
            idx = 0

        if self.batch_index in self.batch_cache:
            batch = self.batch_cache[self.batch_index]
            self.batch_index += 1
            return batch

        X = self.X[idx:idx + self.batch_size]
        y = self.y[idx:idx + self.batch_size]
        lengths = self.lengths[idx:idx+self.batch_size]
        Xbatch = []
        ybatch = []
        for i in xrange(X.shape[0]):
            Xbatch.append([self.review_encoding.convert_representation(t)
                        for t in X[i]])
            if self.use_lengths:
                ybatch.append([self.target_encoding.convert_representation(t) * lengths[i, j]
                            for j, t in enumerate(y[i])])
            else:
                ybatch.append([self.target_encoding.convert_representation(t)
                            for j, t in enumerate(y[i])])

        Xbatch = np.swapaxes(Xbatch, 0, 1)
        ybatch = np.swapaxes(ybatch, 0, 1)
        self.batch_index += 1
        return np.array(Xbatch), np.array(ybatch)
