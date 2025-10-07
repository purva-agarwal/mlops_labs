import math

def fun1(x, y):
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    return x + y

def fun2(x, y):
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    return x - y

def fun3(x, y):
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    return x * y

def fun4(x, y, z):
    return x + y + z

def fun5(x, y):
    """Divides x by y."""
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    if y == 0:
        raise ZeroDivisionError("Cannot divide by zero.")
    return x / y

def fun6(x, y):
    """Returns x raised to the power y."""
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    return x ** y

def fun7(*args):
    """Returns the average of any number of inputs."""
    if not args:
        raise ValueError("At least one number required.")
    if not all(isinstance(i, (int, float)) for i in args):
        raise ValueError("All inputs must be numbers.")
    return sum(args) / len(args)

def factorial(n):
    """Calculates the factorial of a non-negative integer."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer.")
    return math.factorial(n)

def percentage(part, whole):
    """Calculates the percentage of part in whole."""
    if whole == 0:
        raise ZeroDivisionError("Whole cannot be zero.")
    return (part / whole) * 100

def quadratic_roots(a, b, c):
    """Returns the real roots of a quadratic equation ax² + bx + c = 0."""
    if a == 0:
        raise ValueError("a cannot be zero in a quadratic equation.")
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return [] 
    elif discriminant == 0:
        return [-b / (2*a)]
    else:
        root1 = (-b + math.sqrt(discriminant)) / (2*a)
        root2 = (-b - math.sqrt(discriminant)) / (2*a)
        return [root1, root2]

def simple_interest(p, r, t):
    """Calculates Simple Interest = (P × R × T) / 100"""
    if any(not isinstance(i, (int, float)) for i in [p, r, t]):
        raise ValueError("All inputs must be numeric.")
    return (p * r * t) / 100

def bmi(weight, height):
    """Calculates Body Mass Index (BMI)."""
    if height <= 0:
        raise ValueError("Height must be positive.")
    return weight / (height ** 2)