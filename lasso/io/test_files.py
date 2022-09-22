
import unittest
from lasso.io.files import collect_files


class Test(unittest.TestCase):

    def test_collect_files(self):
        files = collect_files('test/io_test', '*.txt')
        self.assertEqual(len(files), 1)

        files = collect_files('test/io_test/', '*.txt', recursive=True)
        self.assertEqual(len(files), 2)

        files1, files2 = collect_files(
            'test/io_test/', ['*.txt', '*.yay'], recursive=True)
        self.assertEqual(len(files1), 2)
        self.assertEqual(len(files2), 1)

        files1, files2 = collect_files(
            ['test/io_test/', 'test/io_test/subfolder'],
            ['*.txt', '*.yay'])
        self.assertEqual(len(files1), 2)
        self.assertEqual(len(files2), 1)
