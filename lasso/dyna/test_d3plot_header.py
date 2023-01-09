from unittest import TestCase

import numpy as np
from lasso.dyna.d3plot_header import (
    D3plotFiletype,
    D3plotHeader,
    d3plot_filetype_from_integer,
    get_digit,
)
from lasso.io.binary_buffer import BinaryBuffer


class D3plotHeaderTest(TestCase):
    def test_loading(self):

        filepaths = [
            "test/simple_d3plot/d3plot",
            "test/d3plot_node_temperature/d3plot",
            "test/d3plot_beamip/d3plot",
            "test/d3plot_solid_int/d3plot",
        ]

        for filepath in filepaths:
            D3plotHeader().load_file(filepath)

        # TODO more

    def test_get_digit(self) -> None:

        number = 1234567890

        # the numbers are sorted from the lowest importance
        # upwards
        # 0 -> 0
        # 1 -> 9
        # ...
        number_str = str(number)[::-1]

        for index in range(len(number_str)):
            digit = get_digit(number, index)
            self.assertEqual(
                digit,
                int(number_str[index]),
                f"index {index} digit {digit} digit_str {number_str[index]}",
            )

        self.assertEqual(get_digit(number, 10), 0)

    def test_d3plot_filetype_from_integer(self) -> None:

        self.assertEqual(
            d3plot_filetype_from_integer(1),
            D3plotFiletype.D3PLOT,
        )
        self.assertEqual(
            d3plot_filetype_from_integer(5),
            D3plotFiletype.D3PART,
        )
        self.assertEqual(
            d3plot_filetype_from_integer(11),
            D3plotFiletype.D3EIGV,
        )

        # INFOR is forbidden
        with self.assertRaises(ValueError):
            d3plot_filetype_from_integer(4)

        with self.assertRaises(ValueError):
            d3plot_filetype_from_integer(0)

    def test_determine_file_settings(self) -> None:

        # the routine checks the "filetype" flag
        # if it makes any sense under any circumstances
        # we assume the corresponding file settings

        # 44 -> int32
        # 88 -> int64
        for position in (44, 88):
            for filetype in (D3plotFiletype.D3PLOT, D3plotFiletype.D3PART, D3plotFiletype.D3EIGV):

                bb = BinaryBuffer()
                bb.memoryview = memoryview(bytearray(256))
                bb.write_number(position, filetype.value, np.int32)

                word_size, itype, ftype = D3plotHeader._determine_file_settings(bb)

                if position == 44:
                    self.assertEqual(word_size, 4)
                    self.assertEqual(itype, np.int32)
                    self.assertEqual(ftype, np.float32)
                else:
                    self.assertEqual(word_size, 8)
                    self.assertEqual(itype, np.int64)
                    self.assertEqual(ftype, np.float64)

        # error
        bb = BinaryBuffer()
        bb.memoryview = memoryview(bytearray(256))

        with self.assertRaises(RuntimeError):
            D3plotHeader._determine_file_settings(bb)
