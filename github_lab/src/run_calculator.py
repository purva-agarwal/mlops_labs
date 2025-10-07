import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from src import calculator

print("Addition:", calculator.fun1(2, 3))
print("Subtraction:", calculator.fun2(10, 4))
print("Multiplication:", calculator.fun3(3, 3))
print("Division:", calculator.fun5(20, 5))
print("Power:", calculator.fun6(2, 5))
print("Average:", calculator.fun7(10, 20, 30))
print("Factorial:", calculator.factorial(5))
print("Percentage:", calculator.percentage(50, 200))
print("Quadratic Roots:", calculator.quadratic_roots(1, -3, 2))
print("Simple Interest:", calculator.simple_interest(1000, 5, 2))
print("BMI:", calculator.bmi(70, 1.75))