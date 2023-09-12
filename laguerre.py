import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_laguerre as L
import qade

n = 4  # Laguerre equation parameter
x = np.linspace(0, 1, 20)  # Sample points at which the equation is to be evaluated
y = qade.function(n_in=1, n_out=1)  # Define the function to be solved for

# Define the equation and boundary conditions
eq = qade.equation(x * y[2] + (1 - x) * y[1] + n * y[0], x)
bcs = qade.equation(y[0] - [1, L(n, 1)], [0, 1])

# Solve the equation using as the basis of functions the monomials [1, x, x^2, x^3]
y_sol = qade.solve([eq, bcs], qade.basis("monomial", 4), scales=(n - 2), num_reads=500)

# Show the results
print(f"loss = {y_sol.loss:.3}, weights = {y_sol.weights}")
plt.plot(x, y_sol(x), linewidth=5)
plt.plot(x, L(n, x), color="black", linestyle="dashed")
plt.show()
