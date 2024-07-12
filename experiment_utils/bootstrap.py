import random
import numpy as np


def bootstrap_mean_error(accuracies: list,samplesize:int = 100,bootstrapsample:int = 1000):
    sample_mean = []
    for k in range(bootstrapsample):
        Sample = random.choices(accuracies,k =samplesize)
        Ehat = np.mean(Sample)
        sample_mean.append(Ehat)
    mean = np.mean(np.array(sample_mean))
    std = np.std(np.array(sample_mean))

    return mean, std