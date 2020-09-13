import unittest
from tf_version.get_npz import pic_to_npz
from tf_version.preprocess import import_npz

class file_tests(unittest.TestCase):
    def test_converts(self):
        pic_to_npz('./raw_imgs', 'test_npz', './test_data')
        data = import_npz('./test_data/test_npz.npz')
        self.assertEqual(data.size, (30, 3, 32, 32))
