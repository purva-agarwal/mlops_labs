import sys
import os
import unittest
import math

# include project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import calculator


class TestCalculator(unittest.TestCase):

    def test_fun1_addition(self):
        self.assertEqual(calculator.fun1(2, 3), 5)
        self.assertEqual(calculator.fun1(-1, 1), 0)

    def test_fun2_subtraction(self):
        self.assertEqual(calculator.fun2(5, 3), 2)
        self.assertEqual(calculator.fun2(-1, -1), 0)

    def test_fun3_multiplication(self):
        self.assertEqual(calculator.fun3(2, 3), 6)
        self.assertEqual(calculator.fun3(5, 0), 0)

    def test_fun4_three_sum(self):
        self.assertEqual(calculator.fun4(2, 3, 5), 10)

    def test_fun5_division(self):
        self.assertEqual(calculator.fun5(10, 2), 5)
        with self.assertRaises(ZeroDivisionError):
            calculator.fun5(1, 0)

    def test_fun6_power(self):
        self.assertEqual(calculator.fun6(2, 3), 8)
        self.assertEqual(calculator.fun6(4, 0.5), 2)

    def test_fun7_average(self):
        self.assertEqual(calculator.fun7(1, 2, 3), 2)
        with self.assertRaises(ValueError):
            calculator.fun7()

    def test_factorial(self):
        self.assertEqual(calculator.factorial(5), 120)
        with self.assertRaises(ValueError):
            calculator.factorial(-1)

    def test_percentage(self):
        self.assertEqual(calculator.percentage(50, 200), 25)
        with self.assertRaises(ZeroDivisionError):
            calculator.percentage(1, 0)

    def test_quadratic_roots(self):
        roots = calculator.quadratic_roots(1, -3, 2)  # x² -3x +2 = 0 => (1,2)
        self.assertCountEqual(roots, [1.0, 2.0])
        self.assertEqual(calculator.quadratic_roots(1, 2, 1), [-1.0])  # one root
        self.assertEqual(calculator.quadratic_roots(1, 0, 1), [])      # no real roots
        with self.assertRaises(ValueError):
            calculator.quadratic_roots(0, 1, 2)

    def test_simple_interest(self):
        self.assertEqual(calculator.simple_interest(1000, 5, 2), 100)
        with self.assertRaises(ValueError):
            calculator.simple_interest("a", 5, 2)

    def test_bmi(self):
        self.assertAlmostEqual(calculator.bmi(70, 1.75), 22.8571, places=4)
        with self.assertRaises(ValueError):
            calculator.bmi(70, 0)


if __name__ == "__main__":
    unittest.main()
