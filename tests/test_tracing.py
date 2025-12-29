import unittest
import numpy as np
from numpy.testing import assert_array_equal

from nanojax import TraceTape, FuncTracer


class TestTraceTape(unittest.TestCase):
    def test_tracer_wraps_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        tracer = FuncTracer(arr)
        assert_array_equal(tracer.array, arr)

    def test_add_traces_operation(self):
        with TraceTape() as trace:
            a = FuncTracer(np.array([1.0, 2.0]))
            b = FuncTracer(np.array([3.0, 4.0]))
            c = a + b
            assert isinstance(c, FuncTracer)

            self.assertEqual(len(trace), 1)
            self.assertEqual(trace[0].func, np.add)
            self.assertIs(trace[0].output, c)
            assert_array_equal(c.array, np.array([4.0, 6.0]))

    def test_no_trace_without_tape(self):
        a = FuncTracer(np.array([1.0, 2.0]))
        b = FuncTracer(np.array([3.0, 4.0]))
        c = a + b
        assert_array_equal(c, np.array([4.0, 6.0]))


if __name__ == "__main__":
    unittest.main()
