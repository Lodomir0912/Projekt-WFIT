import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation

# Define parameters
t, g = smp.symbols('t g')
m1, m2 = smp.symbols('m1 m2')
L1, L2 = smp.symbols('L1 L2')
the1, the2 = smp.symbols(r'\theta_1, \theta_2', cls=smp.Function)
the1 = the1(t)
the2 = the2(t)

# Define derivatives and second derivatives
the1_d = smp.diff(the1, t)
the2_d = smp.diff(the2, t)
the1_dd = smp.diff(the1_d, t)
the2_dd = smp.diff(the2_d, t)

# Define x1, y1, x2, y2
x1 = L1*smp.sin(the1)
y1 = -L1*smp.cos(the1)
x2 = L1*smp.sin(the1) + L2*smp.sin(the2)
y2 = -L1*smp.cos(the1) - L2*smp.cos(the2)

# Kinetic energy
T1 = 1/2 * m1 * (smp.diff(x1, t)**2 + smp.diff(y1, t)**2)
T2 = 1/2 * m2 * (smp.diff(x2, t)**2 + smp.diff(y2, t)**2)
T = T1 + T2

# Potential energy
V1 = m1 * g * y1
V2 = m2 * g * y2
V = V1 + V2

# Lagrangian
L = T - V

# Lagrange's equations
LE1 = smp.diff(L, the1) - smp.diff(smp.diff(L, the1_d), t).simplify()
LE2 = smp.diff(L, the2) - smp.diff(smp.diff(L, the2_d), t).simplify()

# Solve Lagrange's equations
sols = smp.solve([LE1, LE2], (the1_dd, the2_dd), simplify=False, rational=False)

# Convert symbolic expressions into numeric functions
dz1dt_f = smp.lambdify((t,g,m1,m2,L1,L2,the1,the2,the1_d,the2_d), sols[the1_dd])
dz2dt_f = smp.lambdify((t,g,m1,m2,L1,L2,the1,the2,the1_d,the2_d), sols[the2_dd])
dthe1dt_f = smp.lambdify(the1_d, the1_d)
dthe2dt_f = smp.lambdify(the2_d, the2_d)

# Define dS/dt function
def dSdt(S, t, g, m1, m2, L1, L2):
    the1, z1, the2, z2 = S
    return [
        dthe1dt_f(z1),
        dz1dt_f(t, g, m1, m2, L1, L2, the1, the2, z1, z2),
        dthe2dt_f(z2),
        dz2dt_f(t, g, m1, m2, L1, L2, the1, the2, z1, z2),
    ]

# Get user input for parameters
m1_val = float(input("Enter value of m1: "))
m2_val = float(input("Enter value of m2: "))
L1_val = float(input("Enter value of L1: "))
L2_val = float(input("Enter value of L2: "))
the1_val = float(input("Enter value of angle theta1: "))
the2_val = float(input("Enter value of angle theta2: "))

# Solve ODEs
t = np.linspace(0, 40, 1001)
g = 9.81
ans = odeint(dSdt, y0=[the1_val, 0, the2_val, 0], t=t, args=(g, m1_val, m2_val, L1_val, L2_val))

# Get theta1 and theta2 from the answer
the1 = ans.T[0]
the2 = ans.T[2]

# Create a function to get x1, y1, x2, y2
def get_x1y1x2y2(t, the1, the2, L1, L2):
    return (L1*np.sin(the1),
            -L1*np.cos(the1),
            L1*np.sin(the1) + L2*np.sin(the2),
            -L1*np.cos(the1) - L2*np.cos(the2))

x1, y1, x2, y2 = get_x1y1x2y2(t, ans.T[0], ans.T[2], L1_val, L2_val)

# Create animation
def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])

    # Add track path
    if i > 0:
        ln2.set_data(x2[:i], y2[:i])
        ln2.set_alpha(0.5)


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_facecolor('k')
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ln1, = plt.plot([], [], 'ro--', lw=3, markersize=8)
ln2, = plt.plot([], [], 'w--', lw=1)
ax.set_ylim(-8, 8)
ax.set_xlim(-8, 8)
ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50)
ani.save('pendulum.gif', writer='pillow', fps=25)
plt.show()
