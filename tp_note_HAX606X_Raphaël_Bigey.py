#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm



# %%
# QUESTION 1 Coder la descente de gradient

# Implémentation fonction de descente de gradient
def descente(grad,x_init,gamma,maxiter,epsilon):
    x = x_init
    res = [x]
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
    res = [x]
    for i in range(1,maxiter+1):
        g = grad(x,a,b)
        if np.linalg.norm(g,ord=2)**2<=epsilon**2:
            break
        else:
            x = (x[0] - gamma*g[0],x[1]-gamma*g[1])
        res.append(x)
    return res

#Affichage sur un même graphique la valeur des objectifs au cours des itérations
X = descenteFab(fabGrad, (1, 1), 0.01, 1000, 0.001,1,1)
X2 = descenteFab(fabGrad, (1, 1), 0.01, 1000, 0.0001,10,10)
X3 = descenteFab(fabGrad, (1, 1), 0.01, 1000, 0.0001,50,50)
X4 = descenteFab(fabGrad, (1, 1), 0.01, 1000, 0.0001,100,100)
Y = [fab(1,1,x[0],x[1]) for x in X]
Y2 = [fab(10,10,x[0],x[1]) for x in X2]
Y3 = [fab(50,50,x[0],x[1]) for x in X3]
Y4 = [fab(100,100,x[0],x[1]) for x in X4]

fig, ax = plt.subplots()
ax.set_title("Évolution graphique de la valeur de l’objectif au cours des itérations de l’algorithme de descente de gradient")

ax.set_yscale("log")
ax.set_ylabel("valeur des itérés")
ax.set_xlabel("itérations")

ax.plot(np.linspace(1, len(X), len(X)), Y,label='a=1=b')
ax.plot(np.linspace(1, len(X2), len(X2)), Y2,label='a=10=b')
ax.plot(np.linspace(1, len(X3), len(X3)), Y3,label='a=50=b')
ax.plot(np.linspace(1, len(X4), len(X4)), Y4,label='a=100=b')
ax.legend()

#%%
#Affichage des lignes de niveau de chaque fonctions
x = np.linspace(-2,2,1000)
y = np.linspace(-2,2,1000)
X, Y = np.meshgrid(x, y)

li1 = descenteFab(fabGrad, (1, 1), 0.1, 500, 0.0001,1,1)
li2 = descenteFab(fabGrad, (1, 1), 0.1, 500, 0.0001,10,10)
li3 = descenteFab(fabGrad, (1,1), 0.1, 500, 0.0001,50,50)
li4 = descenteFab(fabGrad, (1, 1), 0.1, 500, 0.0001,100,100)
x_li1, y_li1 = zip(*li1)
x_li2, y_li2 = zip(*li2)
x_li3, y_li3 = zip(*li3)
x_li4, y_li4 = zip(*li4)


Z1 = fab(1,1,X,Y)
plt.contour(X, Y, Z1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ligne de niveau de la fonction fab a=1=b')
plt.colorbar(label='Valeurs de f(x, y)')
plt.grid(True)
plt.scatter(x_li1,y_li1,color='red',label='Les points itérés',marker='.')
plt.legend()
plt.show()

Z2 = fab(10,10,X,Y)
plt.contour(X, Y, Z2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ligne de niveau de la fonction fab a=10=b')
plt.colorbar(label='Valeurs de f(x, y)')
plt.grid(True)
plt.scatter(x_li2,y_li2,color='red',label='Les points itérés',marker='.')
plt.legend()
plt.show()

Z3 = fab(50,50,X,Y)
plt.contour(X, Y, Z3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ligne de niveau de la fonction fab a=50=b')
plt.colorbar(label='Valeurs de f(x, y)')
plt.grid(True)
plt.scatter(x_li3,y_li3,color='red',label='Les points itérés',marker='.')
plt.legend()
plt.show()

Z4 = fab(100,100,X,Y)
plt.contour(X, Y, Z4)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ligne de niveau de la fonction fab a=100=b')
plt.colorbar(label='Valeurs de f(x, y)')
plt.grid(True)
plt.scatter(x_li4,y_li4,color='red',label='Les points itérés',marker='.')
plt.legend()
plt.show()

# Remarque sur les vitesses de convergences:
# On remarque que plus la fonction est "plate", plus la méthode de convergence va être longue
# Pour la fonction a=1=b, la méthode converge rapidement
# Pour la fonction a=100=b, la méthode n'arrive même pas à 0 car celle-ci converge trop lentement


# %%
# Affichage de la distance à l’optimum en norme à échelle logarithmique

li1_norm = [np.linalg.norm(v,ord=2) for v in li1]
li2_norm = [np.linalg.norm(v,ord=2) for v in li2]
li3_norm = [np.linalg.norm(v,ord=2) for v in li3]
li4_norm = [np.linalg.norm(v,ord=2) for v in li4]

plt.yscale("log")

plt.ylabel("distance à l'optimum")
plt.xlabel("itérations")

plt.title('Distance à l\'optimum en norme l2')
plt.plot(li1_norm,label='a=1=b')
plt.plot(li2_norm,label='a=10=b')
plt.plot(li3_norm,label='a=50=b')
plt.plot(li4_norm,label='a=100=b')
plt.legend()
# %%
# a = ? ; b = ?
#   
