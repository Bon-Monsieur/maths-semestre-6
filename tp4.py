#%%
import numpy as np
from scipy.optimize import minimize

# %% 
# QUESTION 1)
def ftest(v):
    x,y=v
    return (x - 1) ** 2 + 3*(y + 1)**2

def descente(grad, x_init, gamma, maxiter, epsilon):
    x = x_init
    results = [x]
    for i in range(1, maxiter + 1):
        g = grad(x)
        if np.square(np.linalg.norm(g)) <= np.square(epsilon):  # Norme 2 (voir doc numpy)
            break
        else:
            x = [x[0] - gamma * g[0], x[1] - gamma * g[1]]
            results.append(x)
    return results

def fgrad(x):
    return (2*(x[0]-1),6*(x[1]+1))

q1 = descente(grad=fgrad,x_init=(1,1),gamma=0.01,maxiter=500,epsilon=10e-3)
print("Liste des itérés: ",q1)
print("Dernier des itérés: ",q1[-1])
# Le minimum est (1,-1)

# Utilisation de minimize pour trouver le minimum sur R²+
bnds = ((0,np.inf),(0,np.inf))
res = minimize(ftest, (2,3), method='TNC',tol=10e-9,bounds=bnds)
print(res.x)
# bnds représente les bordes de recherche. On voit que la fonction converge
# vers (1,0), ce qui est normal car le min est (1,-1), mais -1 n'est pas positif



# %%
# QUESTION 2
def descente_projete(grad,proj,x_init,gamma,maxiter,eps):
    x = x_init
    
    for i in range(maxiter):
        g = grad(x)
        if np.linalg.norm(g)**2<=eps**2:
            break
        else:
            tmp = x
            for j in range(len(x)):
                tmp[j] = x[j]-gamma*g[j]
            x = proj(tmp)
    return x


#%%
# QUESTION 3