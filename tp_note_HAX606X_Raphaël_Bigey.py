#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm



# %%

def descente(grad,x_init,gamma,maxiter,epsilon):
    x = x_init
    res = []
    for i in range(1,maxiter+1):
        g = grad(x)
        if np.linalg.norm(g,ord=2)**2<=epsilon**2:
            break
        else:
            x = (x[0] - gamma*g[0],x[1]-gamma*g[1])
        res.append(x)
    return res

def ftest(x, y):
    return (x - 1) ** 2 + 3*(y + 1)**2

def fgrad(x):
    return (2*(x[0]-1),6*(x[1]+1))

print("descente: ",descente(grad=fgrad,x_init=(1,1),gamma=0.01,maxiter=500,epsilon=10e-3))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = ftest(X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


# %%

def fab(a,b,x):
    return x[1]**2/a + x[0]**2/b

def fabGrad(x,a,b):
    return (2*x[0]/b,2*x[1]/a)

def descenteFab(grad,x_init,gamma,maxiter,epsilon,a,b):
    x = x_init
    res = []
    for i in range(1,maxiter+1):
        g = grad(x,a,b)
        if np.linalg.norm(g,ord=2)**2<=epsilon**2:
            break
        else:
            x = (x[0] - gamma*g[0],x[1]-gamma*g[1])
        res.append(x)
    return res


X = descenteFab(fabGrad, (1, 1), 0.01, 1000, 0.0001,1,1)
X2 = descenteFab(fabGrad, (1, 1), 0.01, 1000, 0.0001,10,10)
X3 = descenteFab(fabGrad, (1, 1), 0.01, 1000, 0.0001,50,50)
X4 = descenteFab(fabGrad, (1, 1), 0.01, 1000, 0.0001,100,100)
Y = [fab(1,1,x) for x in X]
Y2 = [fab(10,10,x) for x in X2]
Y3 = [fab(50,50,x) for x in X3]
Y4 = [fab(100,100,x) for x in X4]
fig, ax = plt.subplots()
plt.title("Évolution graphique de la valeur de l’objectif au cours des itérations de l’algorithme de descente de gradient")
plt.yscale("log")
plt.ylabel("y")
plt.xlabel("x")
plt.plot(np.linspace(1, len(X), len(X)), Y)
plt.plot(np.linspace(1, len(X2), len(X2)), Y2)
plt.plot(np.linspace(1, len(X3), len(X3)), Y3)
plt.plot(np.linspace(1, len(X4), len(X4)), Y4)

# %%
