import unittest
import pandas as pd
import numpy as np

from training import windowing


class TestCreateWindows(unittest.TestCase):


    def test_correct_windowing_one_simulation_no_overlap(self):
        # Test that windows are of correct size and overlap
        df = pd.DataFrame({
            'simulation_id': [42, 42, 42, 42, 42, 42],
            'input1': [1, 2, 3, 4, 5, 6],
            'input2': [8, 9, 10, 11, 12, 13],
            'output1': [15, 16, 17, 18, 19, 20]
        })
        window_size = 3
        overlap_size = 0
        input_columns = ['simulation_id', 'input1', 'input2']
        output_columns = ['output1']

        windows = windowing.create_windows(df, window_size, overlap_size, input_columns, output_columns)

        self.assertEqual(len(windows), 2)
        self.assert_windows_have_expected_structure(df, input_columns, output_columns, overlap_size, window_size, windows)

    def test_correct_windowing_one_simulation_no_overlap_window_size_1(self):
        # Test that windows are of correct size and overlap
        df = pd.DataFrame({
            'simulation_id': [42, 42, 42, 42, 42, 42],
            'input1': [1, 2, 3, 4, 5, 6],
            'input2': [8, 9, 10, 11, 12, 13],
            'output1': [15, 16, 17, 18, 19, 20]
        })
        window_size = 1
        overlap_size = 0
        input_columns = ['simulation_id', 'input1', 'input2']
        output_columns = ['output1']

        windows = windowing.create_windows(df, window_size, overlap_size, input_columns, output_columns)

        self.assertEqual(len(windows), 6)
        self.assert_windows_have_expected_structure(df, input_columns, output_columns, overlap_size, window_size, windows)


    def test_correct_windowing_one_simulation_no_overlap_small_windows(self):
        # Test that windows are of correct size and overlap
        df = pd.DataFrame({
            'simulation_id': [42, 42, 42, 42, 42, 42],
            'input1': [1, 2, 3, 4, 5, 6],
            'input2': [8, 9, 10, 11, 12, 13],
            'output1': [15, 16, 17, 18, 19, 20]
        })
        window_size = 1
        overlap_size = 0
        input_columns = ['simulation_id', 'input1', 'input2']
        output_columns = ['output1']

        windows = windowing.create_windows(df, window_size, overlap_size, input_columns, output_columns)

        self.assertEqual(len(windows), 6)

        self.assert_windows_have_expected_structure(df, input_columns, output_columns, overlap_size, window_size, windows)

    def assert_windows_have_expected_structure(self, df, input_columns, output_columns, overlap_size, window_size, windows):
        for i, (input_data, output_data) in enumerate(windows):
            self.assertEqual(len(input_data), window_size)
            self.assertEqual(len(output_data), window_size)

            start_index = i * (window_size - overlap_size)
            for row_index in range(window_size):
                expected_row_inputs = df.loc[start_index + row_index, input_columns].values
                self.assertTrue(np.array_equal(np.array(input_data[row_index]), expected_row_inputs))

                expected_row_outputs = df.loc[start_index + row_index, output_columns].values
                self.assertTrue(np.array_equal(np.array(output_data[row_index]), expected_row_outputs))

    def test_correct_windowing_one_simulation_overlap(self):
        # Test that windows are of correct size and overlap
        df = pd.DataFrame({
            'simulation_id': [42, 42, 42, 42, 42, 42],
            'input1': [1, 2, 3, 4, 5, 6],
            'input2': [8, 9, 10, 11, 12, 13],
            'output1': [15, 16, 17, 18, 19, 20]
        })
        window_size = 3
        overlap_size = 2
        input_columns = ['simulation_id', 'input1', 'input2']
        output_columns = ['output1']

        windows = windowing.create_windows(df, window_size, overlap_size, input_columns, output_columns)

        self.assertEqual(len(windows), 4)
        self.assert_windows_have_expected_structure(df, input_columns, output_columns, overlap_size, window_size, windows)

    def test_correct_windowing_one_simulation_overlap_larger_window(self):
        # Test that windows are of correct size and overlap
        df = pd.DataFrame({
            'simulation_id': [42, 42, 42, 42, 42, 42],
            'input1': [1, 2, 3, 4, 5, 6],
            'input2': [8, 9, 10, 11, 12, 13],
            'output1': [15, 16, 17, 18, 19, 20]
        })
        window_size = 5
        overlap_size = 2
        input_columns = ['simulation_id', 'input1', 'input2']
        output_columns = ['output1']

        windows = windowing.create_windows(df, window_size, overlap_size, input_columns, output_columns)

        self.assertEqual(len(windows), 2)
        for window_idx, (input_data, output_data) in enumerate(windows):
            self.assertEqual(len(input_data), window_size)
            self.assertEqual(len(output_data), window_size)

            start_index = window_idx * (window_size - overlap_size)

            if window_idx == 0:
                for row_index in range(5):
                    expected_row_inputs = df.loc[start_index + row_index, input_columns].values
                    self.assertTrue(np.array_equal(np.array(input_data[row_index]), expected_row_inputs))

                    expected_row_outputs = df.loc[start_index + row_index, output_columns].values
                    self.assertTrue(np.array_equal(np.array(output_data[row_index]), expected_row_outputs))
            if window_idx == 1:
                for row_index in range(3):
                    expected_row_inputs = df.loc[start_index + row_index, input_columns].values
                    self.assertTrue(np.array_equal(np.array(input_data[row_index]), expected_row_inputs))

                    expected_row_outputs = df.loc[start_index + row_index, output_columns].values
                    self.assertTrue(np.array_equal(np.array(output_data[row_index]), expected_row_outputs))

                for row_index in range(3, window_size):
                    self.assertTrue(np.all(np.array(input_data[row_index]) == 0))
                    self.assertTrue(np.all(np.array(output_data[row_index]) == 0))

    def test_short_sequence_padding(self):
        df = pd.DataFrame({
            'simulation_id': [42, 42, 42, 42, 42, 42],
            'input1': [1, 2, 3, 4, 5, 6],
            'input2': [8, 9, 10, 11, 12, 13],
            'output1': [15, 16, 17, 18, 19, 20]
        })
        window_size = 100
        overlap_size = 0
        input_columns = ['simulation_id', 'input1', 'input2']
        output_columns = ['output1']

        windows = windowing.create_windows(df, window_size, overlap_size, input_columns, output_columns)
        self.assertEqual(len(windows), 1)

        for i, (input_data, output_data) in enumerate(windows):
            self.assertEqual(len(input_data), window_size)
            self.assertEqual(len(output_data), window_size)

            start_index = i * (window_size - overlap_size)
            for row_index in range(6):
                expected_row_inputs = df.loc[start_index + row_index, input_columns].values
                self.assertTrue(np.array_equal(np.array(input_data[row_index]), expected_row_inputs))

                expected_row_outputs = df.loc[start_index + row_index, output_columns].values
                self.assertTrue(np.array_equal(np.array(output_data[row_index]), expected_row_outputs))

            for row_index in range(6, window_size):
                self.assertTrue(np.all(np.array(input_data[row_index]) == 0))
                self.assertTrue(np.all(np.array(output_data[row_index]) == 0))



    def test_last_window_padding(self):
        # Test padding behavior for the last window in a sequence
        df = pd.DataFrame({
            'simulation_id': [42, 42, 42, 42, 42, 42],
            'input1': [1, 2, 3, 4, 5, 6],
            'input2': [8, 9, 10, 11, 12, 13],
            'output1': [15, 16, 17, 18, 19, 20]
        })
        window_size = 5
        overlap_size = 0
        input_columns = ['simulation_id', 'input1', 'input2']
        output_columns = ['output1']

        windows = windowing.create_windows(df, window_size, overlap_size, input_columns, output_columns)

        for i, (input_data, output_data) in enumerate(windows):
            start_index = i * (window_size - overlap_size)

            if i == 0:
                for row_index in range(5):
                    expected_row_inputs = df.loc[start_index + row_index, input_columns].values
                    self.assertTrue(np.array_equal(np.array(input_data[row_index]), expected_row_inputs))

                    expected_row_outputs = df.loc[start_index + row_index, output_columns].values
                    self.assertTrue(np.array_equal(np.array(output_data[row_index]), expected_row_outputs))
            if i == 1:
                for row_index in range(1):
                    expected_row_inputs = df.loc[start_index + row_index, input_columns].values
                    self.assertTrue(np.array_equal(np.array(input_data[row_index]), expected_row_inputs))

                    expected_row_outputs = df.loc[start_index + row_index, output_columns].values
                    self.assertTrue(np.array_equal(np.array(output_data[row_index]), expected_row_outputs))

                for row_index in range(1, window_size):
                    self.assertTrue(np.all(np.array(input_data[row_index]) == 0))
                    self.assertTrue(np.all(np.array(output_data[row_index]) == 0))


    def test_correct_grouping(self):
        # Test that windows are of correct size and overlap
        df = pd.DataFrame({
            'simulation_id': [42, 42, 42, 42, 42, 42, 69, 69],
            'input1': [1, 2, 3, 4, 5, 6, 101, 102],
            'input2': [8, 9, 10, 11, 12, 13, 103, 104],
            'output1': [15, 16, 17, 18, 19, 20, 105, 106]
        })
        window_size = 5
        overlap_size = 2
        input_columns = ['simulation_id', 'input1', 'input2']
        output_columns = ['output1']

        windows = windowing.create_windows(df, window_size, overlap_size, input_columns, output_columns)

        self.assertEqual(len(windows), 3)

        for window_idx, (input_data, output_data) in enumerate(windows):
            self.assertEqual(len(input_data), window_size)
            self.assertEqual(len(output_data), window_size)

            start_index = window_idx * (window_size - overlap_size)

            if window_idx == 0:
                for row_index in range(5):
                    expected_row_inputs = df.loc[start_index + row_index, input_columns].values
                    self.assertTrue(np.array_equal(np.array(input_data[row_index]), expected_row_inputs))

                    expected_row_outputs = df.loc[start_index + row_index, output_columns].values
                    self.assertTrue(np.array_equal(np.array(output_data[row_index]), expected_row_outputs))
            if window_idx == 1:
                for row_index in range(3):
                    expected_row_inputs = df.loc[start_index + row_index, input_columns].values
                    self.assertTrue(np.array_equal(np.array(input_data[row_index]), expected_row_inputs))

                    expected_row_outputs = df.loc[start_index + row_index, output_columns].values
                    self.assertTrue(np.array_equal(np.array(output_data[row_index]), expected_row_outputs))

                for row_index in range(3, window_size):
                    self.assertTrue(np.all(np.array(input_data[row_index]) == 0))
                    self.assertTrue(np.all(np.array(output_data[row_index]) == 0))
            if window_idx == 2:
                for row_index in range(2):
                    expected_row_inputs = df.loc[start_index + row_index, input_columns].values
                    self.assertTrue(np.array_equal(np.array(input_data[row_index]), expected_row_inputs))

                    expected_row_outputs = df.loc[start_index + row_index, output_columns].values
                    self.assertTrue(np.array_equal(np.array(output_data[row_index]), expected_row_outputs))

                for row_index in range(2, window_size):
                    self.assertTrue(np.all(np.array(input_data[row_index]) == 0))
                    self.assertTrue(np.all(np.array(output_data[row_index]) == 0))

if __name__ == '__main__':
    unittest.main()