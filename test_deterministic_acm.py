import unittest
import numpy as np
from skimage.segmentation import checkerboard_level_set

from deterministic_acm import chan_vese
from generate_detection_analysis import extract_detection_features


class DeterministicACMTests(unittest.TestCase):
    def test_split_iterations_equal_continuous_run(self):
        image = np.random.default_rng(12).random((24, 24))
        initial = checkerboard_level_set(image.shape, 6)
        partial = chan_vese(image, 35, initial)
        split = chan_vese(image, 10, partial, start_iteration=35)
        np.testing.assert_array_equal(split, chan_vese(image, 45, initial))

    def test_features_do_not_depend_on_previous_detections(self):
        image = np.random.default_rng(42).random((32, 32))
        box = np.array([5, 5, 24, 24])
        first = extract_detection_features(image, box, .8, None)
        extract_detection_features(1-image, box, .3, None)
        repeated = extract_detection_features(image, box, .8, None)
        self.assertEqual(first, repeated)


if __name__ == "__main__":
    unittest.main()
