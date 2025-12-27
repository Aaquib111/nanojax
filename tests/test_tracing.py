import unittest
import numpy as np

from nanojax import TraceTape, FuncTracer, get_current_trace


class TestTraceTape(unittest.TestCase):
    def test_trace_stack_empty_by_default(self):
        self.assertIsNone(get_current_trace())

    def test_trace_tape_creates_trace(self):
        with TraceTape() as trace:
            self.assertIsNotNone(get_current_trace())
            self.assertIs(get_current_trace(), trace)

    def test_trace_tape_cleans_up(self):
        with TraceTape():
            pass
        self.assertIsNone(get_current_trace())

    def test_nested_trace_tapes(self):
        with TraceTape() as outer:
            self.assertIs(get_current_trace(), outer)
            with TraceTape() as inner:
                self.assertIs(get_current_trace(), inner)
                self.assertIsNot(get_current_trace(), outer)
            self.assertIs(get_current_trace(), outer)
        self.assertIsNone(get_current_trace())


class TestFuncTracer(unittest.TestCase):
    def test_tracer_wraps_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        tracer = FuncTracer(arr)
        np.testing.assert_array_equal(tracer.array, arr)

    def test_add_traces_operation(self):
        with TraceTape() as trace:
            a = FuncTracer(np.array([1.0, 2.0]))
            b = FuncTracer(np.array([3.0, 4.0]))
            c = a + b
            assert isinstance(c, FuncTracer)

            self.assertEqual(len(trace), 1)
            self.assertEqual(trace[0].func, np.add)
            self.assertIs(trace[0].output, c)
            np.testing.assert_array_equal(c.array, np.array([4.0, 6.0]))

    def test_no_trace_without_tape(self):
        a = FuncTracer(np.array([1.0, 2.0]))
        b = FuncTracer(np.array([3.0, 4.0]))
        c = a + b
        np.testing.assert_array_equal(c, np.array([4.0, 6.0]))


if __name__ == "__main__":
    unittest.main()
