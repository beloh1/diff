import matplotlib.pyplot as plt
import numpy as np
import qade

t = np.linspace(0, 1, 20)  # Sample points at which the equation is to be evaluated
x, y = qade.function(n_in=1, n_out=2)  # Functions to solve for

# Define the equations and initial conditions.
#  The equations are multiplied by a factor of 1/10, in order for the initial conditions
#  to have more relative importance in the loss, so that they are satisfied.
eq1 = qade.equation((2 * x[1] + x[0] + 3 * y[0]) / 10, t)
eq2 = qade.equation((2 * y[1] + 3 * x[0] + y[0]) / 10, t)
ic1 = qade.equation(x[0] - 5, 0)
ic2 = qade.equation(y[0] - 3, 0)

# Solve the equation using as the basis of functions the monomials [1, t, t^2]
xy_sol = qade.solve([eq1, eq2, ic1, ic2], qade.basis("monomial", 3), scales=4)
print(f"loss = {xy_sol.loss}, weights = {np.around(xy_sol.weights, 2)}")


def xy_true(t):  # Analytical solution, for comparison purposes
    t = np.atleast_1d(t)
    return np.column_stack(
        [np.exp(t) + 4 * np.exp(-2 * t), -np.exp(t) + 4 * np.exp(-2 * t)]
    )


plt.plot(t, xy_sol(t)[:, 0], label="$x(t)$", linewidth=5)
plt.plot(t, xy_sol(t)[:, 1], label="$y(t)$", linewidth=5)
plt.plot(t, xy_true(t)[:, 0], linestyle="dashed", color="black")
plt.plot(t, xy_true(t)[:, 1], linestyle="dashed", color="black")
plt.legend()
plt.xlabel("$t$")
plt.show()
