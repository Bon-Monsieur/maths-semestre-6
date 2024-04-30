#%%
import numpy as np  # package de calcul scientifique
import matplotlib.pyplot as plt  # graphiques
import scipy
#%%
#EXERCICE 1
# QUESTION 1)
A = 1

def f_ras(x,y):
    return 2*A + x**2 - A*np.cos(2*np.pi*x) + y**2  - A*np.cos(2*np.pi*y)

def f_ras2(v):
    x,y=v
    return 2*A + x**2 - A*np.cos(2*np.pi*x) + y**2  - A*np.cos(2*np.pi*y)

x = np.linspace(-5.12,5.12,1000)
y = np.linspace(-5.12,5.12,1000)
X, Y = np.meshgrid(x, y)
Z = f_ras(X,Y)

fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1,projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)
ax1.set_title('Surface 3D')

ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X, Y, Z, levels=7)
fig.colorbar(contour, ax=ax2, shrink=0.5, aspect=5)
ax2.set_title('Contour Plot')


# %%
# QUESTION 2)
# La fonction n'est pas convexe à cause des 4 points

#%%
# QUESTION 3)
def f_ras_grad(v):
    x,y=v
    return np.array([2*x+2*np.pi*A*np.sin(2*np.pi*x),2*y+2*np.pi*A*np.sin(2*np.pi*y)])


# %%
#QUESTION 4)
from scipy.optimize import minimize
pt = (-1,1)
result1 = minimize(f_ras2, pt, method='nelder-mead',tol=1e-10)

print("Resultat de la minimisation pour Nelder point(-1,1):")
print('Success : %s' % result1.success)

print('Status : %s' % result1.message)
print('Total Evaluations: %d' % result1.nfev)
# evaluate solution
solution1 = result1.x
evaluation1 = f_ras2(solution1)
print('Solution: f(%s) = %.5f' % (solution1, evaluation1),"\n")

print("===============================\n")

pt = (-1,1)
result2 = minimize(f_ras2, pt, method='powell',tol=1e-10)

print("Resultat de la minimisation pour Powell point(-1,1):")
print('Success : %s' % result2.success)

print('Status : %s' % result2.message)
print('Total Evaluations: %d' % result2.nfev)
# evaluate solution
solution2 = result2.x
evaluation2 = f_ras2(solution2)
print('Solution: f(%s) = %.5f' % (solution2, evaluation2),"\n")

print("====================Second point======================\n")
pt = (2,1.5)
result3 = minimize(f_ras2, pt, method='nelder-mead',tol=1e-10)

print("Resultat de la minimisation pour Nelder point(2,1.5):")
print('Success : %s' % result3.success)

print('Status : %s' % result3.message)
print('Total Evaluations: %d' % result3.nfev)
# evaluate solution
solution3 = result3.x
evaluation3 = f_ras2(solution3)
print('Solution: f(%s) = %.5f' % (solution3, evaluation3),"\n")

print("===============================\n")

pt = (2,1.5)
result4 = minimize(f_ras2, pt, method='Powell',tol=1e-10)

print("Resultat de la minimisation pour Powell point(2,1.5):")
print('Success : %s' % result4.success)

print('Status : %s' % result4.message)
print('Total Evaluations: %d' % result4.nfev)
# evaluate solution
solution4 = result4.x
evaluation4 = f_ras2(solution4)
print('Solution: f(%s) = %.5f' % (solution4, evaluation4),"\n")

# %%
# QUESTION 5)
# Commentaires:
# Plein de minimums locaux => les méthodes ne convergent pas toute
# la seule à converger correctement est la méthode Powell pour le 
# point (-1,1).

#%%
# QUESTION 6)
import torch

def f_ras3(v):
    x,y=v
    return 2*A + x**2 - A*torch.cos(2*torch.pi*x) + y**2  - A*torch.cos(2*torch.pi*y)

v = torch.tensor([-1.0, 1.0],requires_grad=True)
optimizer = torch.optim.SGD([v], lr=0.1)  # descente de gradient
for i in range(101):
    optimizer.zero_grad()  # on remet à 0 l'arbre des gradients
    fx = f_ras3(v)
    fx.backward()  # calcul des gradients
    optimizer.step()  # pas de la descente
print(v.detach().numpy())  # afficher en numpy la solution

v = torch.tensor([-1.0, 1.0],requires_grad=True)
optimizer = torch.optim.RMSprop([v], lr=0.1)  # descente de gradient
for i in range(101):
    optimizer.zero_grad()  # on remet à 0 l'arbre des gradients
    fx = f_ras3(v)
    fx.backward()  # calcul des gradients
    optimizer.step()  # pas de la descente
print(v.detach().numpy())  # afficher en numpy la solution

# Commentaires:
# la méthode 2 est plus efficace car elle trouve le minimum global de la fonction
# La méthode 1 trouve un minimum local


# %%
# QUESTION 7)

def descente(f_ras_grad, x_init, maxiter, epsilon, lr_scheduler_values,func): #Methode de descente du tp note
    x = x_init
    results = [x]
    values = [func(x)]
    for i in range(1, maxiter + 1):
        gamma=lr_scheduler_values[i-1]
        g = f_ras_grad(x)
        if np.square(np.linalg.norm(g)) <= np.square(epsilon):
            break   
        else:
            x = x-gamma*g
            results.append(x)
            values.append(func(x))
    return results,values

#%%
#QUESTION 8)
# On rempli rempli le tableau lr_scheduler_values par un gamma unique. 
# afin que le gamma ne change pas de valeur au cours des itérations

#%%
# QUESTION 9)
lr_scheduler_values = [0.1]*201
print("Descente à pas fixe sur f_Ras:",descente(f_ras_grad,(-1,1),200,1e-10,lr_scheduler_values,f_ras2)[0])
print("Dernier itéré de la descente:",descente(f_ras_grad,(-1,1),200,1e-10,lr_scheduler_values,f_ras2)[0][-1])


# %%
# PROGRAMMER LA VALEUR DU PAS
# QUESTION 10)
# %% Question 10,11,12:
ymin = 0.001
ymax = 0.1
N = 500

lr_constant_min = [ymin] * N
lr_constant_max = [ymax] * N
lr_multistep = [ymax / 10**(i // 100) for i in range(N)]
lr_cosine = [ymin + 1/2 * (ymax - ymin) * (1 + np.cos(i * np.pi / N)) for i in range(1, N + 1)]

plt.plot(lr_constant_max, linestyle='dashed', label=r'$\gamma_{max}$')
plt.plot(lr_constant_min, linestyle='dashed', label=r'$\gamma_{min}$')
plt.plot(lr_multistep, label='multistep')
plt.plot(lr_cosine, label='cosine')

plt.legend()
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Learning Rate')
plt.title('Comparison of learning rate scheduler')
plt.show()



#%%
# QUESTION 13 et 14)
np.random.seed(len("raphaelbigey"))
v = np.array([np.random.uniform(-5.12,5.12),np.random.uniform(-5.12,5.12)])
print(v)

# %%
# QUESTION 15)
descente_gamma_min = descente(f_ras_grad,v,N,1e-10,lr_constant_min,f_ras2)
descente_gamma_max = descente(f_ras_grad,v,N,1e-10,lr_constant_max,f_ras2)
descente_gamma_multi = descente(f_ras_grad,v,N,1e-10,lr_multistep,f_ras2)
descente_gamma_cosine = descente(f_ras_grad,v,N,1e-10,lr_cosine,f_ras2)
plt.plot(descente_gamma_cosine[1])
plt.plot(descente_gamma_min[1])
plt.plot(descente_gamma_max[1])
plt.plot(descente_gamma_multi[1])
plt.yscale('log')
# Rajouter titre + légende

# Commentaire:
# Il y a 3 stratégies de pas pour lesquelles l'objectif converge correctement vers 0.
# Pour les pas fixes, les deux stratégies ne convergent pas correctement

#%%
# QUESTION 16)
x = np.linspace(-5.12,5.12,1000)
y = np.linspace(-5.12,5.12,1000)
X, Y = np.meshgrid(x, y)
Z = f_ras(X,Y)
# COSINE
x_cosine, y_cosine = zip(*descente_gamma_cosine[0])
fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(20, 9))
ax1.contourf(X,Y,Z)
ax1.plot(x_cosine,y_cosine,color='blue',marker='.',linestyle='-')
ax1.set_title('cosine')
ax1.set_aspect('equal')
# MULTISTEP
x_multi, y_multi = zip(*descente_gamma_multi[0])
ax2.contourf(X,Y,Z)
ax2.plot(x_multi,y_multi,color='green',marker='.',linestyle='-')
ax2.set_title('multistep')
ax2.set_aspect('equal')
# LOW
x_low, y_low = zip(*descente_gamma_min[0])
ax3.contourf(X,Y,Z)
ax3.plot(x_low,y_low,color='cyan',marker='.',linestyle='-')
ax3.set_title('low')
ax3.set_aspect('equal')
# HIGH
x_high, y_high = zip(*descente_gamma_max[0])
ax4.contourf(X,Y,Z)
ax4.plot(x_high,y_high,color='pink',marker='.',linestyle='-')
ax4.set_title('high')
ax4.set_aspect('equal')


# %%
#QUESTION 17 refaire un scipy avec un autre point et A=5 et afficher les 4 
# graphique du 16
from scipy.optimize import minimize
A = 5 
pt = (0.5,0)
result1 = minimize(f_ras2, pt, method='nelder-mead',tol=1e-10)

print("Resultat de la minimisation pour Nelder point(-1,1):")
print('Success : %s' % result1.success)

print('Status : %s' % result1.message)
print('Total Evaluations: %d' % result1.nfev)
# evaluate solution
solution1 = result1.x
evaluation1 = f_ras2(solution1)
print('Solution: f(%s) = %.5f' % (solution1, evaluation1),"\n")

print("===============================\n")

pt = (0.5,0)
result2 = minimize(f_ras2, pt, method='powell',tol=1e-10)

print("Resultat de la minimisation pour Powell point(-1,1):")
print('Success : %s' % result2.success)

print('Status : %s' % result2.message)
print('Total Evaluations: %d' % result2.nfev)
# evaluate solution
solution2 = result2.x
evaluation2 = f_ras2(solution2)
print('Solution: f(%s) = %.5f' % (solution2, evaluation2),"\n")

pt = np.array([0.5,0])
descente_gamma_min = descente(f_ras_grad,pt,N,1e-10,lr_constant_min,f_ras2)
descente_gamma_max = descente(f_ras_grad,pt,N,1e-10,lr_constant_max,f_ras2)
descente_gamma_multi = descente(f_ras_grad,pt,N,1e-10,lr_multistep,f_ras2)
descente_gamma_cosine = descente(f_ras_grad,pt,N,1e-10,lr_cosine,f_ras2)

x = np.linspace(-5.12,5.12,1000)
y = np.linspace(-5.12,5.12,1000)
X, Y = np.meshgrid(x, y)
Z = f_ras(X,Y)
# COSINE
x_cosine, y_cosine = zip(*descente_gamma_cosine[0])
fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(20, 9))
ax1.contourf(X,Y,Z)
ax1.plot(x_cosine,y_cosine,color='blue',marker='.',linestyle='-')
ax1.set_title('cosine')
ax1.set_aspect('equal')
# MULTISTEP
x_multi, y_multi = zip(*descente_gamma_multi[0])
ax2.contourf(X,Y,Z)
ax2.plot(x_multi,y_multi,color='green',marker='.',linestyle='-')
ax2.set_title('multistep')
ax2.set_aspect('equal')
# LOW
x_low, y_low = zip(*descente_gamma_min[0])
ax3.contourf(X,Y,Z)
ax3.plot(x_low,y_low,color='cyan',marker='.',linestyle='-')
ax3.set_title('low')
ax3.set_aspect('equal')
# HIGH
x_high, y_high = zip(*descente_gamma_max[0])
ax4.contourf(X,Y,Z)
ax4.plot(x_high,y_high,color='pink',marker='.',linestyle='-')
ax4.set_title('high')
ax4.set_aspect('equal')

# A revoir avec les autres


# %% 
# EXERCICE 2
# QUESTION 18)
def Sim_GDA(x_init,y_init,gamma,maxiter,eps,f_grad_x,f_grad_y):
    x = x_init
    y = y_init
    result = [x]
    for k in range(maxiter):
        nab_x = f_grad_x(x,y)
        nab_y = f_grad_y(x,y)
        if (np.linalg(x,ord=2)**2<=eps**2 or np.linalg(y,ord=2)**2<=eps**2):
            break
        else:
            x = x - gamma*nab_x
            y = y + gamma*nab_y




#%%
# QUESTION 21) 
# reprendre tp4.py pour projeter à chaque itérations notre x dans l'espace des simplexs

#%%
# QUESTION 22)
# Faire tourner les algos avec les points initaux: [1,0,0] et [0,0.5,0.5]

# QUESTION 20) ecrire la grosse somme, puis dériver par p1, p2, p3 puis par q1,q2,q3

# QESTION 23) par 22) on obtient deux vecteurs correspondant à la facon de jouer pour A
# Afin de minimiser sa perte, et joueuse B à maximiser sa perte 