import numpy as np
import pytest

from data.utils import calculate_class_weights, create_train_val_split


def test_class_weights_ignore_invalid_labels_and_stabilize_absent_classes() -> None:
    masks = [np.array([[0, 0, 1, -1, 5]])]
    weights = calculate_class_weights(masks, num_classes=3)
    assert weights.tolist() == pytest.approx([2 / 3, 4 / 3, 0])


def test_split_does_not_mutate_numpy_global_rng() -> None:
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    create_train_val_split(list(range(10)), seed=7)
    assert np.random.random() == expected
