import pytest
import math
from src import calculator


def test_fun1():
    assert calculator.fun1(2, 3) == 5
    assert calculator.fun1(-1, 1) == 0

def test_fun2():
    assert calculator.fun2(5, 3) == 2
    assert calculator.fun2(-1, -1) == 0

def test_fun3():
    assert calculator.fun3(2, 3) == 6
    assert calculator.fun3(5, 0) == 0

def test_fun4():
    assert calculator.fun4(2, 3, 5) == 10

def test_fun5():
    assert calculator.fun5(10, 2) == 5
    with pytest.raises(ZeroDivisionError):
        calculator.fun5(1, 0)

def test_fun6():
    assert calculator.fun6(2, 3) == 8
    assert calculator.fun6(4, 0.5) == 2

def test_fun7():
    assert calculator.fun7(1, 2, 3) == 2
    with pytest.raises(ValueError):
        calculator.fun7()

def test_factorial():
    assert calculator.factorial(5) == 120
    with pytest.raises(ValueError):
        calculator.factorial(-1)

def test_percentage():
    assert calculator.percentage(50, 200) == 25
    with pytest.raises(ZeroDivisionError):
        calculator.percentage(1, 0)

def test_quadratic_roots():
    roots = calculator.quadratic_roots(1, -3, 2)
    assert set(roots) == {1.0, 2.0}
    assert calculator.quadratic_roots(1, 2, 1) == [-1.0]
    assert calculator.quadratic_roots(1, 0, 1) == []
    with pytest.raises(ValueError):
        calculator.quadratic_roots(0, 1, 2)

def test_simple_interest():
    assert calculator.simple_interest(1000, 5, 2) == 100
    with pytest.raises(ValueError):
        calculator.simple_interest("a", 5, 2)

def test_bmi():
    assert round(calculator.bmi(70, 1.75), 4) == 22.8571
    with pytest.raises(ValueError):
        calculator.bmi(70, 0)
