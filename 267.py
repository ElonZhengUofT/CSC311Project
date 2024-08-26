import numpy as np
import matplotlib.pyplot as plt

# Define epsilon
epsilon = 0.1

# Define nullclines
x_nullcline = np.linspace(-2, 2, 400)
y_nullcline = -x_nullcline - 3 * epsilon * x_nullcline**2

# Define vector field
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)
xdot = Y
ydot = -X - 3 * epsilon * X**2

# Plot nullclines
plt.plot(x_nullcline, np.zeros_like(x_nullcline), 'b', label='x-nullcline ($y=0$)')
plt.plot(-1/(3*epsilon), 0, 'ro')  # Equilibrium point X_2
plt.plot(X, -X - 3 * epsilon * X**2, 'g', label='y-nullcline ($x=-1/(3\epsilon)$)')

# Plot vector field along x-nullcline
for i in range(0, len(x_nullcline), 40):
    plt.quiver(x_nullcline[i], 0, 0, 0.1, color='r', scale=20)

# Plot vector field along y-nullcline
y_nullcline_direction = np.sign(-(-1/(3*epsilon)) - 3 * epsilon * (-1/(3*epsilon))**2)
plt.quiver(-1/(3*epsilon), 0, 0, y_nullcline_direction*0.1, color='r', scale=20)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Phase Plane')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.show()