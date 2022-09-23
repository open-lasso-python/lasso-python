
import os
import sys
import numpy as np
from unittest import TestCase
import filecmp

from lasso.io.BinaryBuffer import BinaryBuffer


class BinaryBufferTest(TestCase):

    def setUp(self):

        # read file
        self.bb = BinaryBuffer("test/simple_d3plot/d3plot")

    def test_init(self):

        # test some stuff for fun
        self.assertEqual(self.bb.mv_[40:42].tobytes(), b'\xaf\\')
        self.assertEqual(len(self.bb), len(self.bb.mv_))
        self.assertEqual(len(self.bb), 192512)

    def test_memoryview(self):

        self.assertEqual(self.bb.mv_, self.bb.memoryview)
        with self.assertRaises(AssertionError):
            self.bb.memoryview = None
        self.memoryview = memoryview(bytearray(b''))

    def test_reading(self):

        # numbers
        self.assertEqual(self.bb.read_number(44, np.int32), 1)
        self.assertEqual(self.bb.read_number(56, np.float32), 960.)

        self.assertEqual(self.bb.read_text(0, 40), ' '*40)
        self.assertEqual(self.bb.read_text(52, 4), 'R712')

        self.assertListEqual(self.bb.read_ndarray(
            60, 12, 1, np.int32).tolist(), [4, 4915, 6])

    def test_save(self):

        self.bb.save('test/tmp')
        eq = filecmp.cmp('test/simple_d3plot/d3plot', 'test/tmp')
        os.remove("test/tmp")
        self.assertEqual(eq, True)

    def test_writing(self):

        bb = BinaryBuffer('test/simple_d3plot/d3plot')
        bb.write_number(44, 13, np.int32)
        self.assertEqual(bb.read_number(44, np.int32), 13)

        array = np.array([1, 2, 3, 4], np.int32)
        bb.write_ndarray(array, 44, 1)
        self.assertListEqual(bb.read_ndarray(
            44, 16, 1, array.dtype).tolist(), array.tolist())

    def test_size(self):

        bb = BinaryBuffer('test/simple_d3plot/d3plot')
        self.assertEqual(bb.size, 192512)
        self.assertEqual(bb.size, len(bb))

        bb.size = 192511
        self.assertEqual(bb.size, 192511)

        bb.size = 192512
        self.assertEqual(bb.size, 192512)
        self.assertEqual(bb.mv_[-1:len(bb)].tobytes(), b'0')
