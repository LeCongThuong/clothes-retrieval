import itertools
import numpy as np
from torch.utils.data import Sampler


def grouper(iterable, n):
    it = itertools.cycle(iter(iterable))
    for _ in range((len(iterable) - 1) // n + 1):
        yield list(itertools.islice(it, n))

# full label coverage per 'epoch'


class PKSampler(Sampler):

    def __init__(self, data_source, classes, labels_to_samples, mapping_files_to_global_id, p=64, k=16):
        super().__init__(data_source)
        self.p = p
        self.k = k
        self.data_source = data_source
        self.classes = classes
        self.labels_to_samples = labels_to_samples
        self.mapping_files_to_global_id = mapping_files_to_global_id

    def __iter__(self):
        rand_labels = np.random.permutation(
            np.arange(len(self.labels_to_samples.keys())))

        for labels_indices in grouper(rand_labels, self.p):
            for label_index in labels_indices:
                label = list(self.labels_to_samples.keys())[label_index]
                samples = self.labels_to_samples[label]
                replace = True if len(samples) < self.k else False
                for s in np.random.choice(samples, self.k, replace=replace):
                    index = self.mapping_files_to_global_id[s]
                    yield index

    def __len__(self):
        num_labels = len(set(self.classes))
        samples = ((num_labels - 1) // self.p + 1) * self.p * self.k
        return samples
