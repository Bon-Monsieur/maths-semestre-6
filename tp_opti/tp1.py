#%%
# Ceci est un commentaire pour l'import des packages.
import numpy as np  # package de calcul scientifique
import matplotlib.pyplot as plt  # package graphique

#%%

entier = 1
print(type(entier))
flottant = 2.5
print(type(flottant))


xmax = np.finfo('float').max
nxmax = 1+np.finfo('float').eps
print(xmax, xmax+1)
#print(1.1*xmax - 1.1*xmax) retourne NAN
print(np.testing.assert_almost_equal(xmax + 1, xmax))

# %%
n_petit = 1200
n_petits = np.logspace(-n_petit, 0, base=2, num=n_petit + 1)
print(n_petits)
print(n_petits[::-1])

for idx, val in enumerate(n_petits[::-1]):  # faire une boucle
    print(idx, val)
    if val <= 0:  # XXX TODO
        break  # on arrête la boucle
print(idx, val)
print(np.finfo(np.float64).smallest_subnormal)


# %%

print(np.isclose(0.6, 0.1 + 0.2 + 0.3, atol=np.finfo(np.float64).smallest_subnormal, rtol=np.finfo(np.float64).smallest_subnormal))
print(np.isclose(0.6, 0.1 + 0.2 + 0.3, atol=2, rtol=np.finfo(np.float64).max))

# %%
# Somme de deux vecteurs
A = np.array([1.0, 2, 3])
B = np.array([-1, -2, -3.0])

# Attribuer à la variable sum_A_B la somme de A et B
sum_A_B =  A+B  # XXX TODO

np.testing.assert_almost_equal(np.zeros((3,)), sum_A_B)
print("it worked")

# Le produit terme à terme avec *
prod_A_B = A*B  # XXX TODO

np.testing.assert_almost_equal(np.array([-1.0, -4, -9]), prod_A_B)
print("it worked")

# Remarque : la même chose fonctionne terme à terme avec /, ** (puissance)
np.testing.assert_almost_equal(np.array([1.0, 4, 9]), A ** 2)
print("it worked: even for powers")


# %%

J = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])

I3 = np.eye(4)

np.testing.assert_almost_equal(I3, np.linalg.matrix_power(J,3)) 
print("it worked: method 1")
np.testing.assert_almost_equal(I3, J@J@J) 
print("it worked: method 2")


# %%

print(f"L'inverse de la matrice: \n {J} \n est \n {np.linalg.inv(J)}")

n = 5  # XXX TODO: tester avec n=100
Jbig = np.roll(np.eye(n), -1, axis=1)  # matrice de permutation de taille n
print(Jbig)

b = np.arange(n)
print(b)

# on peut transposer une matrice facilement de 2 manières :
print(Jbig)
print(Jbig.T)
print(np.transpose(Jbig))

import time
# Résolution de système par une méthode naive: inversion de matrice
t0 = time.perf_counter()  # XXX TODO
y1 = np.linalg.inv(Jbig) @ b
timing_naive = time.perf_counter()-t0
print(
    f"Temps pour résoudre un système avec la formule mathématique: {timing_naive:.4f} s."
)

# Résolution de système par une méthode adaptée: fonctions dédiée de `numpy``
t1 = time.perf_counter()
y2 = np.linalg.solve(Jbig,b)
timinig_optimized = time.perf_counter()-t1
print(
    f"Temps pour résoudre un système avec la formule mathématique: {timinig_optimized:.4f} s.\nC'est donc {timing_naive / timinig_optimized} fois plus rapide d'utiliser la seconde formulation"
)

np.testing.assert_almost_equal(y1, y2)
print("Les deux méthodes trouvent le même résultat")

# %%
J = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])
A = J
print(f"The first column is {A[:, 0]}")

# Afficher la deuxième ligne de A
print(f"The second row is {A[1,:]}")  # XXX TODO

C = np.eye(5, 5)
C[0:5:2,:] = 0  # mettre à zéro une ligne sur deux. # 
print(C)

# %%
# grille linéaire
x = np.linspace(-5, 5, num=100)
print(x)

# grille géométrique
np.logspace(1, 9, num=9)  


# %%
d = np.arange(6)
print(d.shape)
print(d.reshape(2,3))
print(d.reshape(3,2))
print(d.reshape(6,))
print(d.reshape(1,6))


# %%
import matplotlib.pyplot as plt 

x = np.linspace(-10,10,num=10000)
y = np.cos(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r"$f:x\mapsto \cos(x)$", zorder=1)
plt.hlines(
    y=1,
    xmin=-10,
    xmax=10,
    label=r"$\pm 1$",
    color="r",
    zorder=1,
    linestyles="dotted",
)
plt.hlines(
    y=-1,
    xmin=-10,
    xmax=10,
    label=r"$\pm 1$",
    color="r",
    zorder=1,
    linestyles="dotted",)

x_extrema = np.pi * np.arange(-3, 4)
y_extrema = np.cos(x_extrema)

plt.scatter(x_extrema, y_extrema,s=30,color='black')
plt.xlabel("$x$")
plt.ylabel('$y$')
plt.legend()
plt.title('Fonction cosinus')
plt.tight_layout()
plt.show()

# %%
# Liste de couleurs, dégradé de violet :
colors = plt.cm.Purples(np.linspace(0.3, 1, 5))
lambdas = np.arange(1, 6)
x = np.linspace(0, 10, 1000, endpoint=True)

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 8))


for i in range(len(lambdas)):
    y = np.exp(-x * i)
    axs[0].plot(x, y, color=colors[i])
    axs[1].semilogy(x, y, color=colors[i], label=f'$\lambda_{{{i}}}$')

# Titres / sous-titres
fig.suptitle("Décroissance exponentielle")
axs[0].set_title("Échelle classique")
axs[1].set_title("Échelle semi-logarithmique")
axs[1].legend(loc=3)

# %%
# Créer une grille de points avec meshgrid : exemple
x = np.linspace(-10, 10, 11)
y = np.linspace(0, 20, 11)
xx, yy = np.meshgrid(x, y)

# xx est x répété "le nombre de points dans y" fois sur les lignes
# yy est y répété "le nombre de points dans x" fois sur les colonnes

fig_level_set, axs = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

axs[0].plot(xx, yy, ls="None", marker=".")
axs[0].set_aspect('equal')
axs[1].plot(yy, xx, ls="None", marker=".")
axs[1].set_aspect('equal')
plt.show()
# Une fonction à deux variables pourra ainsi être visualisée en l'évaluant
# sur chacun des points d'une grille assez fine.


# %%

from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "widget")
from pylab import cm

import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x**4 + y**4 -2*(x-y)**2

x = np.arange(-5, 5, 0.05)
y = np.arange(-5, 5, 0.05)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Figure : lignes de niveau.
fig_level_sets, ax_level_sets = plt.subplots(1, 1, figsize=(3, 3))
ax_level_sets.set_title(r"$x^2 - y^4$: lignes de niveau")
level_sets = ax_level_sets.contourf(X, Y, Z, levels=30, cmap="RdBu_r")
fig_level_sets.colorbar(level_sets, ax=ax_level_sets, fraction=0.046, pad=0.04)




# Figure : surface
fig_surface, ax_surface = plt.subplots(
    1, 1, figsize=(3, 3), subplot_kw={"projection": "3d"}
)
ax_surface.set_title(r"$x^2 - y^4$: surface")
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

# Ajout du point d'origine sur la première figure
ax_level_sets.scatter(0, 0, color='red')
# Ajout du point d'origine sur la deuxième figure
ax_surface.scatter(0, 0, f(0, 0), color='red')

plt.show()


# %%

# La fonction f_dir admet des dérivées partielles sur R²\{(0,0)} comme quotient
# de fonctions polynomiales qui ne s'annulent jamais. En(0,0) on peut la calculer

# La fonction f_dir n'est pas continue en (0,0) car en (t,t) elle vaut 1/2

def z(x, y):
    return np.where(np.allclose([x, y], 0), 0, x * y / (x ** 2 + y ** 2))


def dx(x, y):
    return np.where(
        np.allclose([x, y], 0),
        0,
        y * (y ** 2 - x ** 2) / (x ** 2 + y ** 2) ** 2,
    )


def dy(x, y):
    return np.where(
        np.allclose([x, y], 0),
        0,
        x * (x ** 2 - y ** 2) / (x ** 2 + y ** 2) ** 2,
    )

x = np.arange(-0.5, 0.5, 0.01)
y = np.arange(-0.5, 0.5, 0.01)
X, Y = np.meshgrid(x, y)
Z = z(X, Y)

dX = dx(X, Y)
dY = dy(X, Y)
speed = np.sqrt(dX * dX + dY * dY)

# Figure 1
fig1 = plt.figure(figsize=(8, 4))
ax1 = fig1.add_subplot(1, 2, 1)
im = ax1.contourf(X, Y, Z, levels=30, cmap="RdBu_r")  # XXX TODO
ax1.streamplot(X, Y, dX, dY, color="k", linewidth=5 * speed / speed.max())
ax1.set_xlim([x.min(), x.max()])
ax1.set_ylim([y.min(), y.max()])
ax1.set_aspect("equal")
cbar = fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_title(r"$\frac{xy}{x^2 + y^2}$: lignes de niveau et gradients")

ax2 = fig1.add_subplot(1, 2, 2, projection="3d")
ax2.set_title(r"$\frac{xy}{x^2 + y^2}$: surface")
surf = ax2.plot_surface(
    X,
    Y,
    Z,
    rstride=1,
    cstride=1,
    cmap=cm.RdBu_r,
    linewidth=0,
    antialiased=False,
)
plt.tight_layout()
plt.show()

# Figure 2
fig2, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.imshow(
    Z, cmap=cm.RdBu_r, origin="lower", extent=[min(x), max(x), min(y), max(y)]
)
CS = ax.contour(X, Y, Z, colors="white", alpha=1, linewidths=0.8)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title("Alternative")
plt.tight_layout(pad=3.0)
plt.show()

# %%
