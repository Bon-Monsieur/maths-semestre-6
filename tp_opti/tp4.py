#%%
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# %% 
# QUESTION 1)
def ftest(v):
    x,y=v
    return (x - 1) ** 2 + 3*(y + 1)**2

def descente(grad, x_init, gamma, maxiter, epsilon): #Methode de descente du tp note
    x = x_init
    results = [x]
    for i in range(1, maxiter + 1):
        g = grad(x)
        if np.square(np.linalg.norm(g)) <= np.square(epsilon):
            break
        else:
            x = x-gamma*g
            results.append(x)
    return results

#Gradient de ftest
def fgrad(x):
    return np.array([2*(x[0]-1),6*(x[1]+1)])

q1 = descente(grad=fgrad,x_init=(1,1),gamma=0.01,maxiter=500,epsilon=1e-10)
print("Liste des itérés: ",q1)
print("Dernier des itérés: ",q1[-1])
# La méthode converge vers (1,-1) qui est bien le minimum

# Utilisation de minimize pour trouver le minimum sur R²+
bnds = ((0,None),(0,None))
res = minimize(ftest, (2,3), method='TNC',tol=1e-10,bounds=bnds)
print(res.x)
# bnds représente les bornes de recherche. On voit que la fonction converge
# vers (1,0), ce qui est normal car le min est (1,-1), mais -1 n'est pas positif



# %%
# QUESTION 2
# Implémentation de la méthode de descente du gradient projeté
def descente_projete(grad,proj,x_init,gamma,maxiter,eps):
    x = x_init
    for i in range(maxiter):
        g = grad(x)
        if np.square(np.linalg.norm(g))<=np.square(eps):
            break
        else:
            x = proj(x-gamma*g)
    return x


#%%
# QUESTION 3
# Fonction de projection sur R²+
def proj1(v):
    return np.array(np.maximum(v,0))

res = descente_projete(fgrad,proj1,[-2,3],10e-3,10000,1e-10)
print("Résultat sous contraite  s.c. x1≥0, x2≥0 :\n",res)
# %%
# QUESTION 4
# Fonction de projection sur la boule unité fermée
def proj2(v):
    return np.array(v / np.maximum(np.square(np.linalg.norm(v)),1))

res = descente_projete(fgrad,proj2,[3,-3],10e-5,10000,1e-10)
print(r'resultat sous contrainte s.c. ||x||²<=1',res)

# %%
# POUR ALLER PLUS LOIN
# Fonction définissant les 2 contraintes
def c1(v):
    x1,x2 = v
    return x1-3*x2+2
def c2(v):
    x1,x2 = v
    return x1+x2

contraintes = [{'type' : 'ineq','fun':c1},{'type':'ineq','fun':c2}]

res = minimize(ftest, (2,3), method='SLSQP',tol=1e-10,constraints=contraintes)
print("Le minimum est en: ",res.x,"et vaut: ",res.fun)

