from sympy import symbols, Eq, solve

# Define symbols
x, y = symbols('x y')

# Single-variable equation: x^2 - 2 = 0
eq1 = Eq(x**2 - 2, 0)
sol1 = solve(eq1, x)
print("Solution of single-variable equation:", sol1)

# System of two linear equations:
# 2x + y = 3
# x - y = 1
eq2 = Eq(2*x + y, 3)
eq3 = Eq(x - y, 1)
sol2 = solve((eq2, eq3), (x, y))
print("Solution of system of equations:", sol2)

if __name__ == "__main__":
    print("Running sympydemo as a script")