#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm



# %%
# QUESTION 1 Coder la descente de gradient

# Implémentation fonction de descente de gradient
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

#Verification avec la fonction f_test
def ftest(x, y):
    return (x - 1) ** 2 + 3*(y + 1)**2

def fgrad(x):
    return (2*(x[0]-1),6*(x[1]+1))


q1 = descente(grad=fgrad,x_init=(1,1),gamma=0.01,maxiter=500,epsilon=10e-3)
print("Liste des itérés: ",q1)
print("Dernier des itérés: ",q1[-1])
#L'algorithme converge bien vers le point (1,-1) minimum global de la fonction


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
# QUESTION 2 Application au cas quadratique


def fab(a,b,x,y):
    return y**2/a + x**2/b

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

#Affichage sur un même graphique la valeur des objectifs au cours des itérations
P = descenteFab(fabGrad, (1, 1), 0.1, 1000, 0.001,1,1)

X2 = descenteFab(fabGrad, (1, 1), 0.01, 1000, 0.001,10,10)
X3 = descenteFab(fabGrad, (1, 1), 0.01, 1000, 0.001,50,50)
X4 = descenteFab(fabGrad, (1, 1), 0.01, 1000, 0.001,100,100)
Y = [fab(1,1,x[0],x[1]) for x in P]
Y2 = [fab(10,10,x[0],x[1]) for x in X2]
Y3 = [fab(50,50,x[0],x[1]) for x in X3]
Y4 = [fab(100,100,x[0],x[1]) for x in X4]
fig, ax = plt.subplots()
ax.set_title("Évolution graphique de la valeur de l’objectif au cours des itérations de l’algorithme de descente de gradient")
ax.set_yscale("log")
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.plot(np.linspace(1, len(P), len(P)), Y)
ax.plot(np.linspace(1, len(X2), len(X2)), Y2)
ax.plot(np.linspace(1, len(X3), len(X3)), Y3)
ax.plot(np.linspace(1, len(X4), len(X4)), Y4)


#Affichage des lignes de niveau de chaque fonctions
x = np.arange(-2, 2, 0.005)
y = np.arange(-2, 2, 0.005)
X, Y = np.meshgrid(x, y)
Z = fab(1,1,X,Y)
fig_level_sets, ax_level_sets = plt.subplots(1, 1, figsize=(3, 3))
ax_level_sets.set_title(r"$a=b=1$: lignes de niveau")
level_sets = ax_level_sets.contourf(X, Y, Z, levels=30, cmap="RdBu_r")
fig_level_sets.colorbar(level_sets, ax=ax_level_sets, fraction=0.046, pad=0.04)
t = [x[0] for x in P]
s = [x[1] for x in P]
ax_level_sets.scatter(t,s, marker=".",color='yellow')
# %%
