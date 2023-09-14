import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import qade

n = 9  

# Define the initial conditions
f0 = [1, 0, 0]  # f(0)=1, f'(0)=0, f''(0)=0

# Create an array of x values
x = np.linspace(0, 1, 20)  # Sample points at which the equation is to be evaluated

# Solve the differential equation


f = qade.function(n_in=1, n_out=1)  # Define the function to be solved for

# Define the equation and boundary conditions
eq = qade.equation( f[2] + f[1] + f[0] - np.sin(x), x)
#bcs = qade.equation(y[0] - [1, 2], [0, 1])

bc1 = qade.equation(f[0] - 1, 0)  # f(0) = 1
bc2 = qade.equation(f[1], 0)      # f'(0) = 0
bc3 = qade.equation(f[2], 0)      # f''(0) = 0

# Combine the boundary conditions into a list
bcs = [bc1, bc2, bc3]

# Solve the equation using monomials as the basis functions
f_sol = qade.solve([eq] + bcs, qade.basis("monomial", n), scales=(n - 2), num_reads=500)


# Solve the equation using as the basis of functions the monomials [1, x, x^2, x^3, ...]
#f_sol = qade.solve([eq, bcs], qade.basis("monomial", n), scales=(n - 2), num_reads=500)

print(f"loss = {f_sol.loss:.3}, weights = {f_sol.weights}")
print(f" weights = {f_sol.weights}")

#value of the approximate solution at point t
def dot_product(t):
    power_vector = [t ** i for i in range(n)]
    return np.dot(power_vector, f_sol.weights.flatten())

#array of approximate solutions at x
approx = [dot_product(i) for i in x]

def func(x):
    return np.sin(x)

def model(f, x):
    dfdx = [f[1], f[2], func(x) - f[0] - f[1] - f[2]]
    return dfdx

solution = odeint(model, f0, x)

y = solution[:, 0]

#euclidean norm of the difference vector, or in other words: error
el_diff = y - approx 
euclid_norm = np.linalg.norm(el_diff)
print(f"Monomial basis of dimension {n} gives us error: ", euclid_norm)

#y_trig =qade.solve([eq, bcs], qade.basis("trig", 4), scales=(n - 2), num_reads=500)
#print(f"loss = {y_trig.loss:.3}, weights = {y_trig.weights}")

#y_four =qade.solve([eq, bcs], qade.basis("fourier", 4), scales=(n - 2), num_reads=500)
#print(f"loss = {y_four.loss:.3}, weights = {y_four.weights}")



#plt.plot(x, y_sol(x), linewidth=5)
#plt.plot(x, L(n, x), color="black", linestyle="dashed")
#plt.show()

#print(y_sol)
#print(y_trig)
#print(y_four)