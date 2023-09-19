from surprise.model_selection import ShuffleSplit
from itertools import chain
import numpy as np
from utils import get_rng

class CustomShufflesplit(ShuffleSplit):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def custom_splits(self, data):
        """Generator function to iterate over trainsets and testsets.
        Args:
            data(:obj:`Dataset<surprise.dataset.Dataset>`): The data containing
                ratings that will be divided into trainsets and testsets.
        Yields:
            tuple of (trainset, testset)
        """

        test_size, train_size = self.validate_train_test_sizes(
            self.test_size, self.train_size, len(data.raw_ratings)
        )
        rng = get_rng(self.random_state)

        for _ in range(self.n_splits):

            if self.shuffle:
                permutation = rng.permutation(len(data.raw_ratings))
                # permutation_trust = rng.permutation(len(data.raw_trusts))
            else:
                permutation = np.arange(len(data.raw_ratings))
                # permutation_trust = np.arange(len(data.raw_trusts))

            raw_trainset = [data.raw_ratings[i] for i in permutation[:test_size]]
            raw_testset = [
                data.raw_ratings[i]
                for i in permutation[test_size : (test_size + train_size)]
            ]
            # raw_trusts = [data.raw_trusts[i] for i in permutation_trust[:]]
            raw_trusts = data.raw_trusts

            trainset = data.custom_construct_trainset(raw_trainset, raw_trusts)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset

def custom_train_test_split(
    data, test_size=0.2, train_size=None, random_state=None, shuffle=True
):
    """Split a dataset into trainset and testset.
    See an example in the :ref:`User Guide <train_test_split_example>`.
    Note: this function cannot be used as a cross-validation iterator.
    Args:
        data(:obj:`Dataset <surprise.dataset.Dataset>`): The dataset to split
            into trainset and testset.
        test_size(float or int ``None``): If float, it represents the
            proportion of ratings to include in the testset. If int,
            represents the absolute number of ratings in the testset. If
            ``None``, the value is set to the complement of the trainset size.
            Default is ``.2``.
        train_size(float or int or ``None``): If float, it represents the
            proportion of ratings to include in the trainset. If int,
            represents the absolute number of ratings in the trainset. If
            ``None``, the value is set to the complement of the testset size.
            Default is ``None``.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for determining the folds. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same splits over multiple calls to ``split()``.
            If RandomState instance, this same instance is used as RNG. If
            ``None``, the current RNG from numpy is used. ``random_state`` is
            only used if ``shuffle`` is ``True``.  Default is ``None``.
        shuffle(bool): Whether to shuffle the ratings in the ``data``
            parameter. Shuffling is not done in-place. Default is ``True``.
    """
    ss = CustomShufflesplit(
        n_splits=1,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    return next(ss.custom_splits(data))


# from surprise.model_selection import KFold

class CustomKFold():
    """A basic cross-validation iterator.

    Each fold is used once as a testset while the k - 1 remaining folds are
    used for training.

    See an example in the :ref:`User Guide <use_cross_validation_iterators>`.

    Args:
        n_splits(int): The number of folds.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for determining the folds. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same splits over multiple calls to ``split()``.
            If RandomState instance, this same instance is used as RNG. If
            ``None``, the current RNG from numpy is used. ``random_state`` is
            only used if ``shuffle`` is ``True``.  Default is ``None``.
        shuffle(bool): Whether to shuffle the ratings in the ``data`` parameter
            of the ``split()`` method. Shuffling is not done in-place. Default
            is ``True``.
    """

    def __init__(self, n_splits=5, random_state=None, shuffle=True):

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, data):
        """Generator function to iterate over trainsets and testsets.

        Args:
            data(:obj:`Dataset<surprise.dataset.Dataset>`): The data containing
                ratings that will be divided into trainsets and testsets.

        Yields:
            tuple of (trainset, testset)
        """

        if self.n_splits > len(data.raw_ratings) or self.n_splits < 2:
            raise ValueError(
                "Incorrect value for n_splits={}. "
                "Must be >=2 and less than the number "
                "of ratings".format(len(data.raw_ratings))
            )

        # We use indices to avoid shuffling the original data.raw_ratings list.
        indices = np.arange(len(data.raw_ratings))

        if self.shuffle:
            get_rng(self.random_state).shuffle(indices)

        start, stop = 0, 0
        for fold_i in range(self.n_splits):
            start = stop
            stop += len(indices) // self.n_splits
            if fold_i < len(indices) % self.n_splits:
                stop += 1

            raw_trainset = [
                data.raw_ratings[i] for i in chain(indices[:start], indices[stop:])
            ]
            raw_testset = [data.raw_ratings[i] for i in indices[start:stop]]
            raw_trusts = data.raw_trusts

            trainset = data.custom_construct_trainset(raw_trainset, raw_trusts)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset

    def get_n_folds(self):

        return self.n_splits