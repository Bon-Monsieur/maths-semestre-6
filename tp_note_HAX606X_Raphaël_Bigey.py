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

# Plot the surface
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
plt.figure()
plt.contour(X, Y, Z1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ligne de niveau de la fonction fab a=1=b')
plt.colorbar()
plt.grid(True)
plt.scatter(x_li1,y_li1,color='red',label='Les points itérés',marker='.')
plt.show()



Z2 = fab(10,10,X,Y)
plt.figure()
plt.contour(X, Y, Z2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ligne de niveau de la fonction fab a=10=b')
plt.colorbar()
plt.grid(True)
plt.scatter(x_li2,y_li2,color='red',label='Les points itérés',marker='.')
plt.show()



Z3 = fab(50,50,X,Y)
plt.figure()
plt.contour(X, Y, Z3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ligne de niveau de la fonction fab a=50=b')
plt.colorbar()
plt.grid(True)
plt.scatter(x_li3,y_li3,color='red',label='Les points itérés',marker='.')
plt.show()



Z4 = fab(100,100,X,Y)
plt.figure()
plt.contour(X, Y, Z4)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ligne de niveau de la fonction fab a=100=b')
plt.colorbar()
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


plt.figure()
plt.yscale("log")

plt.ylabel("distance à l'optimum")
plt.xlabel("itérations")

plt.title('Distance à l\'optimum en norme l2')
plt.plot(li1_norm,label='a=1=b')
plt.plot(li2_norm,label='a=10=b')
plt.plot(li3_norm,label='a=50=b')
plt.plot(li4_norm,label='a=100=b')
plt.legend()
plt.show()



# %%

# a = 3 ; b = 20
x = np.linspace(-2,2,1000)
y = np.linspace(-2,2,1000)
X, Y = np.meshgrid(x, y)

ex1 = descenteFab(fabGrad, (1, 1), 0.1, 1000, 0.0001,3,20)
x_ex1, y_ex1 = zip(*ex1)

Zex1 = fab(3,20,X,Y)

plt.figure()
plt.contour(X, Y, Zex1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ligne de niveau de la fonction fab a=3;b=20')
plt.colorbar(label=r'Valeurs de f_{a,b}(x, y)')
plt.grid(True)
plt.scatter(x_ex1,y_ex1,color='red',label='Les points itérés',marker='.')
plt.legend()
plt.show()


# a = 17 ; b = 5
x = np.linspace(-2,2,1000)
y = np.linspace(-2,2,1000)
X, Y = np.meshgrid(x, y)

ex2 = descenteFab(fabGrad, (1, 1), 0.1, 1000, 0.0001,17,5)
x_ex2, y_ex2 = zip(*ex2)

Zex2 = fab(17,5,X,Y)
plt.figure()
plt.contour(X, Y, Zex2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ligne de niveau de la fonction fab a=17;b=5')
plt.colorbar(label=r'Valeurs de f_{a,b}(x, y)')
plt.grid(True)
plt.scatter(x_ex2,y_ex2,color='red',label='Les points itérés',marker='.')
plt.legend()
plt.show()






# %%
# QUESTION 3 Descente de gradient par coordonnée

#Methode de la descente de gradient par coordonées avec condition d'arret
def descenteCoordFixe(grad,x_init,gamma,n_iter,epsilon,a,b):
    x = x_init
    res=[x]
    for i in range(1,n_iter+1):
        g = grad(x,a,b)
        if np.linalg.norm(g,ord=2)**2<=epsilon**2:
            break
        else:

            temp = x.copy()
            for j in range(0,len(g)):
                temp[j] = x[j]-gamma*g[j]
                x=temp.copy()
                res.append(x)
    return res



# %%
import time

# Comparaison en temps des deux méthodes pour a=1=b
print("Comparaison des deux méthodes pour a=1=b")

t0 = time.perf_counter()
res1 = descenteFab(fabGrad,[1, 1], 0.01, 1000, 10e-8,1,1)[-1]
temps1 = time.perf_counter()-t0

print(f"Temps d'exécution de la méthode classique: {temps1:.4f}, le dernier élément rajouté est: {res1}.")

t1 = time.perf_counter()
res2 = descenteCoordFixe(fabGrad,[1,1],0.01,1000,10e-8,1,1)[-1]
temps2 = time.perf_counter()-t1
print(f"Temps d'exécution de la méthode par coordonnée: {temps2:.4f}, le dernier élément rajouté est: {res2}.")


tmp = round(temps1/temps2, 3)
print(f"Donc la méthode par coordonnée est {tmp} fois plus rapide que la méthode classique pour a=1=b")


print()
print("=====================================")
print()


# Comparaison en temps des deux méthodes pour a=100=b
print("Comparaison des deux méthodes pour a=100=b")

t2 = time.perf_counter()
res3 = descenteFab(fabGrad,[1, 1], 0.01, 1000, 10e-8,100,100)[-1]
temps3 = time.perf_counter()-t2
print(f"Temps d'exécution de la méthode classique: {temps3:.4f}, le dernier élément rajouté est: {res3}.")


t3 = time.perf_counter()
res4 = descenteCoordFixe(fabGrad,[1,1],0.01,1000,10e-8,100,100)[-1]
temps4 = time.perf_counter()-t3
print(f"Temps d'exécution de la méthode par coordonnée: {temps4:.4f}, le dernier élément rajouté est: {res3}.")

tmp = round(temps3/temps4, 3)
print(f"Donc la méthode par coordonnée est {tmp} fois plus rapide que la méthode classique pour a=100=b")

#%% 
#Affichage pour l'algo de descente classique
x = np.linspace(-2,2,1000)
y = np.linspace(-2,2,1000)
X, Y = np.meshgrid(x, y)

li1 = descenteFab(fabGrad, (1, 1), 0.1, 1000, 0.0001,1,1)
x_li1, y_li1 = zip(*li1)


Z1 = fab(1,1,X,Y)
plt.figure()
plt.contour(X, Y, Z1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Descente gradient classique')
plt.colorbar(label='Valeurs de f(x, y)')
plt.grid(True)
plt.scatter(x_li1,y_li1,color='red',label='Les points itérés',marker='.')
plt.legend()
plt.show()
#%%
#Affichage pour l'algo de descente par coordonnées
li2 = descenteCoordFixe(fabGrad, [1, 1], 0.1, 1000, 0.0001,1,1)
x_li2, y_li2 = zip(*li2)

Z2 = fab(1,1,X,Y)
plt.figure()
plt.contour(X, Y, Z2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Descente gradient coordonées')
plt.colorbar(label='Valeurs de f(x, y)')
plt.grid(True)
plt.scatter(x_li2,y_li2,color='red',label='Les points itérés',marker='.')
plt.legend()
plt.show()



# %%
# QUESTION 4 Avec scipy
# Partie a) Problème convexe

from scipy.optimize import minimize


#Minimisation de la fonction f_20_20 avec la méthode Nelder-Mead
def fab_20_20(v):
    x, y = v
    return y**2/20 + x**2/20



pt = (1,1)
result1 = minimize(fab_20_20, pt, method='nelder-mead',tol=10e-10)

print("Resultat de la minimisation pour Nelder:")
print('Success : %s' % result1.success)

print('Status : %s' % result1.message)
print('Total Evaluations: %d' % result1.nfev)
# evaluate solution
solution1 = result1.x
evaluation1 = fab_20_20(solution1)
print('Solution: f(%s) = %.5f' % (solution1, evaluation1),"\n")

print("=====================================\n")

result2 = minimize(fab_20_20, pt, method='CG',tol=10e-10)
# summarize the result
print("Resultat de la minimisation pour CG:")
print('Success : %s' % result2.success)
print('Status : %s' % result2.message)
print('Total Evaluations: %d' % result2.nfev) #Donne le nombre total d'évaluation
# evaluate solution
solution2 = result2.x
evaluation2 = fab_20_20(solution2)
print('Solution: f_20_20(%s) = %.5f' % (solution2, evaluation2),"\n")
print("Est ce que les deux solutions sont les même à 10e-9 près:",np.isclose(solution1,solution2, atol=10e-9, rtol=10e-9))

'''On remarque dans un premier temps que les deux méthodes n'effectuent pas le même 
nombre d'évaluation. En effet, avec la méthode Nelder-Mead minimize() 
effectue 139 itérations, tandis que pour la méthode CG (conjugate gradient)en effectue 204.

Dans un second temps on remarque que la valeur du minimum renvoyé est le même: 0. 
Cependant la  valeur affichée pour le x auquel la fonction prend cette valeur n'est pas 
exactement la même. Mais l'affichage avec l'appel à np.isclose() permet de s'assurer 
que les deux algorithmes convergent vers "la même solution"

result.success nous donne si la fonction renvoie bien un résultat
result.message explique la cause de terminaison de l'algorithme
result.nfev nous donne le nombre d'evaluation effectuées par minimize()
'''


# %%
# QUESTION 4   
# Partie b) Problème non convexe
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "widget")
from pylab import cm


# Representation fonction de Rosenbrock et ses lignes de niveau sur [-5,5]
def fr(x, y):
    return (1-x)**2 + 100*(y-x)**2


def GradFr(x,y):
    return (-2*(1-x)-200*(y-x) ,200*(y-x))

x1 = np.arange(-5, 5, 0.05)
y1 = np.arange(-5, 5, 0.05)
X, Y = np.meshgrid(x1, y1)
Z = fr(X, Y)


# Figure : lignes de niveau.
fig_level_sets, ax_level_sets = plt.subplots(1, 1, figsize=(3, 3))
ax_level_sets.set_title(r"Ligne de niveau de $ f_r $ sur [-5,5]²")
level_sets = ax_level_sets.contourf(X, Y, Z, levels=20, cmap="RdBu_r")
fig_level_sets.colorbar(level_sets, ax=ax_level_sets, fraction=0.046, pad=0.04)
plt.show()

# Figure : surface
fig_surface, ax_surface = plt.subplots(
    1, 1, figsize=(3, 3), subplot_kw={"projection": "3d"}
)
ax_surface.set_title(r"Rosenbrock surface sur [-5,5]²")
surf = ax_surface.plot_surface(
    X,
    Y,
    Z,
    rstride=1,
    cstride=1,
    cmap=cm.RdBu_r,
    linewidth=0,
    antialiased=True,
    alpha=0.8,
)

# %%
# Représentation des lignes de niveau de f_r sur [0,1.5]
x2 = np.arange(0, 1.5, 0.05)
y2 = np.arange(0, 1.5, 0.05)
X2, Y2 = np.meshgrid(x2, y2)
Z2 = fr(X2,Y2)

fig_level_sets, ax_level_sets = plt.subplots(1, 1, figsize=(3, 3))
ax_level_sets.set_title(r"Ligne de niveau de $ f_r $ sur [0,1.5]²")
level_sets = ax_level_sets.contourf(X2, Y2, Z2, levels=40, cmap="RdBu_r")
fig_level_sets.colorbar(level_sets, ax=ax_level_sets, fraction=0.046, pad=0.04)
plt.show()

#La difficulté d'optimiser la fonction vient de 






# %%
# Minimisation avec la descente de gradient classique
# Premier point initial: (-3,3)
test1 = descente(GradFr,(-3,3),0.01,2000,10e-8)[-1]

