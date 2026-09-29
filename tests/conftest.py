import cv2
import numpy as np
import pytest

from pv_anomaly.config import Config


def _write(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), array)


@pytest.fixture
def write():
    """Write an image, creating parent directories."""
    return _write


@pytest.fixture
def dataset(tmp_path):
    """A <root>/{train,test}/{defect,label} tree with `_mask` suffixed JPEGs."""
    rng = np.random.default_rng(0)
    for split, n in [("train", 6), ("test", 3)]:
        for i in range(n):
            mask = np.zeros((128, 128), np.uint8)
            mask[20:40, 20:40] = 255
            _write(
                tmp_path / split / "defect" / f"{i:04d}.jpg",
                rng.integers(0, 255, (128, 128), dtype=np.uint8),
            )
            _write(tmp_path / split / "label" / f"{i:04d}_mask.jpg", mask)
    return tmp_path


@pytest.fixture
def cfg(dataset):
    return Config(root=dataset, tile=128, stride=128, num_classes=2)
