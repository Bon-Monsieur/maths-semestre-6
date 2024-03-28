#%%
import numpy as np  # package de calcul scientifique
import matplotlib.pyplot as plt  # package graphique
from scipy.optimize import golden
from scipy.optimize import minimize
# %%

def dicho(f,a,b,e):
    while(abs(b-a)>e):
        m = 1/2*(b+a)

        if f(a)*f(m)<=0:
            b=m
        else :
            a=m
    return m

def f_test(x) : return x-1

print(dicho(f_test,-2,2,0.01))

# %%
def dicho_for(f,a,b,nb_it):
    for i in range(nb_it):
        m = 1/2*(b+a)

        if f(a)*f(m)<=0:
            b=m
        else :
            a=m
    return m

print(dicho_for(f_test,-2,2,20))

# %%
x = np.linspace(-2, 2, num=10000)

def f(x):
    return x**4 - (x-1)**2 + 1

def df(x):
    return 4*x**3 - 2*(x-1)

y = df(x)    
import time
t0 = time.perf_counter()
print(dicho_for(df,-2,2,75))
t1 = time.perf_counter()
print(t1-t0)

#Le seul point critique est x=-1
#Le minimum de la fonction est f(-1)=-2

#print(dicho(df,-1/2,1/2,0.0001))
# %%

#Méthode du nombre d'ord

def Or(f,a,b,tol=1e-5,max_iter=100):
    mi = min(a,b)
    ma = max(a,b)
    phi = (1+np.sqrt(5))/2
    x1 = 1/phi*mi + (1-1/phi)*ma
    f1 = f(x1)
    x2 = 1/phi*ma + (1-1/phi)*mi
    f2 = f(x2)

    for _ in range(max_iter):
        if (ma -mi)<=tol:
            break
        elif f1<f2:
            ma = x2
            x2 = x1
            f2 = f1
            x1 = 1/phi*mi + (1-1/phi)*ma
            f1 = f(x1)
        else:
            mi = x1
            x1 = x2
            f1 = f2
            x2 = 1/phi*ma + (1-1/phi)*mi
            f2 = f(x2)
    
    return (ma+mi)/2

def f(x):
    return x**2+2
def g(x):
    return x**6 + 3*np.exp(-x**2) + 1/2*np.sin(5*x/2)


print("ma methode f: ",Or(f,-5,20,0.001))
print("golden: ",golden(f))

print("ma methode g: ",Or(g,-3/2,1.5,0.000001))
print("golden: ",golden(g))
#sur [0.3, 1.5] le min est en 0.88
#sur [-1.5,1.5] le min est en -0.80



# %%
result = minimize(f,x0=0.5,tol=10**-9)
print("le minimum est en: ",result.x," et vaut: ",result.fun)


# %%
