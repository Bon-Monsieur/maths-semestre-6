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

# Plot 3D
fig = plt.figure(figsize=(15,15))
ax1 = fig.add_subplot(1,2,1,projection='3d',label="f(x,y)")
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)
ax1.set_title(r'3D plot of $f_{Ras}$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')

# Lignes de niveau
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X, Y, Z, levels=7)
fig.colorbar(contour, ax=ax2, shrink=0.5, aspect=5)
ax2.set_title(r'Level Set Plot of $f_{Ras}$')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_aspect('equal')

# %%
# QUESTION 2)
# La fonction n'est pas convexe car elle admet de nombreux minimums locaux qui ne sont pas globaux.

#%%
# QUESTION 3)
def f_ras_grad(v):
    x,y=v
    return np.array([2*x+2*np.pi*A*np.sin(2*np.pi*x),2*y+2*np.pi*A*np.sin(2*np.pi*y)])


# %%
#QUESTION 4)
from scipy.optimize import minimize
pt = (-1,1)

# Minimisation méthode nelder-mead
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

# Minimisation méthode powell
result2 = minimize(f_ras2, pt, method='powell',tol=1e-10)

print("Resultat de la minimisation pour Powell point(-1,1):")
print('Success : %s' % result2.success)

print('Status : %s' % result2.message)
print('Total Evaluations: %d' % result2.nfev)
# evaluate solution
solution2 = result2.x
evaluation2 = f_ras2(solution2)
print('Solution: f(%s) = %.5f' % (solution2, evaluation2),"\n")


# Minimisation méthode nelder-mead second point
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


# Minimisation méthode powell second point
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
# On remarque qu'aucune des deux méthodes n'arrivent à converger vers l'objectif. Celles-ci se bloquent
# sur un minimum local de la fonction.

# Explications scipy:
# Success : retourne si la méthode s'est terminée avec succès
# Message : Description de la cause de l'arrêt
# nfev : nombre d'evaluation de la fonction objective
# x : La slution de l'optimisation


#%%
# QUESTION 6)
import torch

def f_ras3(v):
    x,y=v
    return 2*A + x**2 - A*torch.cos(2*torch.pi*x) + y**2  - A*torch.cos(2*torch.pi*y)

# Minimisation avec l'optimiseur SGD
v = torch.tensor([-1.0, 1.0],requires_grad=True)
optimizer = torch.optim.SGD([v], lr=0.1)  # descente de gradient
for i in range(101):
    optimizer.zero_grad()  # on remet à 0 l'arbre des gradients
    fx = f_ras3(v)
    fx.backward()  # calcul des gradients
    optimizer.step()  # pas de la descente
print(v.detach().numpy())  # afficher en numpy la solution

# Minimisation avec l'optimiseur RMSprop
v = torch.tensor([-1.0, 1.0],requires_grad=True)
optimizer = torch.optim.RMSprop([v], lr=0.1)  # descente de gradient
for i in range(101):
    optimizer.zero_grad()  # on remet à 0 l'arbre des gradients
    fx = f_ras3(v)
    fx.backward()  # calcul des gradients
    optimizer.step()  # pas de la descente
print(v.detach().numpy())  # afficher en numpy la solution


# Commentaires:
# La méthode 2 est plus efficace car elle trouve le minimum global de la fonction.
# La méthode 1 trouve un minimum local.


# %%
# QUESTION 7)

def descente(f_ras_grad, x_init, maxiter, epsilon, lr_scheduler_values,func):
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
# On rempli le tableau lr_scheduler_values par un gamma unique afin que le gamma ne change pas 
# de valeur au cours des itérations.

#%%
# QUESTION 9)
lr_scheduler_values = [0.1]*201
print("Descente à pas fixe sur f_Ras:",descente(f_ras_grad,(-1,1),200,1e-10,lr_scheduler_values,f_ras2)[0])
print("Dernier itéré de la descente:",descente(f_ras_grad,(-1,1),200,1e-10,lr_scheduler_values,f_ras2)[0][-1])


# %% Question 10,11,12)
ymin = 0.001
ymax = 0.1
N = 500

# Calcul des schedulers
lr_constant_min = [ymin] * N
lr_constant_max = [ymax] * N
lr_multistep = [ymax / 10**(i // 100) for i in range(N)]
lr_cosine = [ymin + 1/2 * (ymax - ymin) * (1 + np.cos(i * np.pi / N)) for i in range(1, N + 1)]
lr_redemarrage = []
N_intervalsize = 100

for k in range(500//N_intervalsize):
    for i in range(1,N_intervalsize):
        lr_redemarrage.append(ymin + 0.5 * (ymax - ymin) * (1 + np.cos(i * np.pi / N_intervalsize)))
    lr_redemarrage.append(lr_redemarrage[-1] + 0.5 * (ymax - ymin) * (1 - np.cos(np.pi / N_intervalsize)))


# Tracer les tailles de pas
plt.plot(lr_cosine, label='cosine',color="blue")
plt.plot(lr_redemarrage, label="cosine with restart",color="orange")
plt.plot(lr_multistep, label='multistep',color="green")
plt.plot(lr_constant_max, linestyle='dotted', label=r'$\gamma_{max}$',color="purple")
plt.plot(lr_constant_min, linestyle='dotted', label=r'$\gamma_{min}$',color="black")
plt.legend()
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Learning Rate')
plt.title('Comparison of learning rate schedulers')
plt.show()



#%%
# QUESTION 13 et 14)
np.random.seed(len("raphaelbigey"))
v = np.array([np.random.uniform(-5.12,5.12),np.random.uniform(-5.12,5.12)])
print(v)

# %%
# QUESTION 15)

# Lancement de la descente
descente_gamma_min = descente(f_ras_grad,v,N,1e-10,lr_constant_min,f_ras2)
descente_gamma_max = descente(f_ras_grad,v,N,1e-10,lr_constant_max,f_ras2)
descente_gamma_multi = descente(f_ras_grad,v,N,1e-10,lr_multistep,f_ras2)
descente_gamma_cosine = descente(f_ras_grad,v,N,1e-10,lr_cosine,f_ras2)
descente_gamma_cosine_red = descente(f_ras_grad,v,N,1e-10,lr_redemarrage,f_ras2)

# Affichage de la convergence de l'objectif
plt.plot(descente_gamma_cosine[1],label="cosine",color="blue")
plt.plot(descente_gamma_multi[1],color="green",label="multistep")
plt.plot(descente_gamma_cosine_red[1],color="orange",label="restart")
plt.plot(descente_gamma_min[1],label="low",color="red")
plt.plot(descente_gamma_max[1],label="high",color="purple")
plt.legend()
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Objectif')
plt.title("Comparaison de la convergence de l'objectif suivant les stratégies de pas")
plt.show()


# Commentaire:
# Il y a 3 stratégies de pas pour lesquelles l'objectif converge correctement vers 0.
# Pour les schedulers ymax et ymin les deux stratégies ne convergent pas correctement.


#%%
# QUESTION 16)
x = np.linspace(-5.12,5.12,1000)
y = np.linspace(-5.12,5.12,1000)
X, Y = np.meshgrid(x, y)
Z = f_ras(X,Y)
# COSINE
x_cosine, y_cosine = zip(*descente_gamma_cosine[0])
fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5,figsize=(15, 15))
ax1.contourf(X,Y,Z)
ax1.plot(x_cosine,y_cosine,color='blue',marker='.',linestyle='-')
ax1.set_title('cosine')
ax1.set_aspect('equal')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
# MULTISTEP
x_multi, y_multi = zip(*descente_gamma_multi[0])
ax2.contourf(X,Y,Z)
ax2.plot(x_multi,y_multi,color='green',marker='.',linestyle='-')
ax2.set_title('multistep')
ax2.set_aspect('equal')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
# COSINE WITH RESTART
x_high, y_high = zip(*descente_gamma_cosine_red[0])
ax3.contourf(X,Y,Z)
ax3.plot(x_high,y_high,color='red',marker='.',linestyle='-')
ax3.set_title('restart')
ax3.set_aspect('equal')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
# LOW
x_low, y_low = zip(*descente_gamma_min[0])
ax4.contourf(X,Y,Z)
ax4.plot(x_low,y_low,color='cyan',marker='.',linestyle='-')
ax4.set_title('low')
ax4.set_aspect('equal')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
# HIGH
x_high, y_high = zip(*descente_gamma_max[0])
ax5.contourf(X,Y,Z)
ax5.plot(x_high,y_high,color='pink',marker='.',linestyle='-')
ax5.set_title('high')
ax5.set_aspect('equal')
ax5.set_xlabel('x')
ax5.set_ylabel('y')



# %%
#QUESTION 17)
from scipy.optimize import minimize
A = 5 
pt = (-2,-3)


# Minimisation avec la méthode nelder-mead
result1 = minimize(f_ras2, pt, method='nelder-mead',tol=1e-10)

print("Resultat de la minimisation pour Nelder point(-2,-3):")
print('Success : %s' % result1.success)

print('Status : %s' % result1.message)
print('Total Evaluations: %d' % result1.nfev)
# evaluate solution
solution1 = result1.x
evaluation1 = f_ras2(solution1)
print('Solution: f(%s) = %.5f' % (solution1, evaluation1),"\n")

print("===============================\n")


# Minimisation avec la méthode powell
result2 = minimize(f_ras2, pt, method='powell',tol=1e-10)

print("Resultat de la minimisation pour Powell point(-2,-3):")
print('Success : %s' % result2.success)

print('Status : %s' % result2.message)
print('Total Evaluations: %d' % result2.nfev)
# evaluate solution
solution2 = result2.x
evaluation2 = f_ras2(solution2)
print('Solution: f(%s) = %.5f' % (solution2, evaluation2),"\n")


# Affichage des lignes de niveau et avec les itérés
pt = np.array([-2,-3])
descente_gamma_min = descente(f_ras_grad,pt,N,1e-10,lr_constant_min,f_ras2)
descente_gamma_max = descente(f_ras_grad,pt,N,1e-10,lr_constant_max,f_ras2)
descente_gamma_multi = descente(f_ras_grad,pt,N,1e-10,lr_multistep,f_ras2)
descente_gamma_cosine = descente(f_ras_grad,pt,N,1e-10,lr_cosine,f_ras2)
descente_gamma_cosine_red = descente(f_ras_grad,pt,N,1e-10,lr_redemarrage,f_ras2)

x = np.linspace(-10,10,1000)
y = np.linspace(-10,10,1000)
X, Y = np.meshgrid(x, y)
Z = f_ras(X,Y)
# COSINE
x_cosine, y_cosine = zip(*descente_gamma_cosine[0])
fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5,figsize=(15, 15))
ax1.contourf(X,Y,Z)
ax1.plot(x_cosine,y_cosine,color='blue',marker='.',linestyle='-')
ax1.set_title('cosine')
ax1.set_aspect('equal')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
# MULTISTEP
x_multi, y_multi = zip(*descente_gamma_multi[0])
ax2.contourf(X,Y,Z)
ax2.plot(x_multi,y_multi,color='green',marker='.',linestyle='-')
ax2.set_title('multistep')
ax2.set_aspect('equal')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
# COSINE WITH RESTART
x_high, y_high = zip(*descente_gamma_cosine_red[0])
ax3.contourf(X,Y,Z)
ax3.plot(x_high,y_high,color='red',marker='.',linestyle='-')
ax3.set_title('restart')
ax3.set_aspect('equal')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
# LOW
x_low, y_low = zip(*descente_gamma_min[0])
ax4.contourf(X,Y,Z)
ax4.plot(x_low,y_low,color='cyan',marker='.',linestyle='-')
ax4.set_title('low')
ax4.set_aspect('equal')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
# HIGH
x_high, y_high = zip(*descente_gamma_max[0])
ax5.contourf(X,Y,Z)
ax5.plot(x_high,y_high,color='pink',marker='.',linestyle='-')
ax5.set_title('high')
ax5.set_aspect('equal')
ax5.set_xlabel('x')
ax5.set_ylabel('y')

# Commentaires:
# La fonction avec A=5 possède beaucoup plus de minimums locaux. Après chaque itérations on tombe
# dans le voisinage d'un autre minimum. Ainsi la méthode n'arrive pas à converger, d'où le fait que la 
# méthode donne des itérés partout. On remarquera que pour shedulers=low la méthode n'arrive pas à 
# sortir du premier minimum dans lequel on tombe.


# %% 
# EXERCICE 2
# QUESTION 18)
def Sim_GDA(p_init,gamma,maxiter,eps,f_grad_x,f_grad_y):
    x = p_init[0]
    y = p_init[1]
    result = [p_init]
    for k in range(0,maxiter):
        nab_x = f_grad_x(x,y) # Renvoie un np.array
        nab_y = f_grad_y(x,y) # Renvoie un np.array
        if (np.square(np.linalg.norm(x))<=np.square(eps) or np.square(np.linalg.norm(y))<=np.square(eps)):
            break
        else:
            x -= gamma*nab_x
            y += gamma*nab_y
            result.append(np.array([x,y]))
    return result

def Alt_GDA(p_init,gamma,maxiter,eps,f_grad_x,f_grad_y):
    x = p_init[0]
    y = p_init[1]
    result = [p_init]
    for k in range(0,maxiter):
        nab_x = f_grad_x(x,y) # Renvoie un np.array
        if (np.square(np.linalg.norm(x))<=np.square(eps) or np.square(np.linalg.norm(y))<=np.square(eps)):
            break
        else:
            x = x - gamma*nab_x
            nab_y = f_grad_y(x,y) # Renvoie un np.array
            y = y + gamma*nab_y
            result.append(np.array([x,y]))
    return result

# %%
# QUESTION 19) 

fig.clear()
def f1(v):
    x,y=v
    return x*y
def nab_x_f1(x,y):
    return y
def nab_y_f1(x,y):
    return x

def f2(v):
    x,y=v
    return 0.5*x**2+10*x*y-0.5*y**2
def nab_x_f2(x,y):
    return x+10*y
def nab_y_f2(x,y):
    return 10*x-y

# Affichage graphique
pt = np.array([0.5,-0.5])
fig, ([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2,figsize=(9, 9))
sim_f1 = Sim_GDA(p_init=pt,gamma=0.1,maxiter=100,eps=1e-10,f_grad_x=nab_x_f1,f_grad_y=nab_y_f1)
x_f1_sim, y_f1_sim = zip(*sim_f1)
ax1.plot(x_f1_sim,y_f1_sim,marker='.',linestyle='-',color="red")
ax1.set_title('Sim-GDA for f1')

sim_f2 = Sim_GDA(p_init=pt,gamma=0.1,maxiter=100,eps=1e-10,f_grad_x=nab_x_f2,f_grad_y=nab_y_f2)
x_f2_sim, y_f2_sim = zip(*sim_f2)
ax2.plot(x_f2_sim,y_f2_sim,marker='.',linestyle='-',color="blue")
ax2.set_title('Sim-GDA for f2')

alt_f1 = Alt_GDA(p_init=pt,gamma=0.1,maxiter=100,eps=1e-10,f_grad_x=nab_x_f1,f_grad_y=nab_y_f1)
x_f1_alt, y_f1_alt = zip(*alt_f1)
ax3.plot(x_f1_alt,y_f1_alt,marker='.',linestyle='-',color="green")
ax3.set_title('Alt-GDA for f1')

alt_f2 = Alt_GDA(p_init=pt,gamma=0.1,maxiter=100,eps=1e-10,f_grad_x=nab_x_f2,f_grad_y=nab_y_f2)
x_f2_alt, y_f2_alt = zip(*alt_f2)
ax4.plot(x_f2_alt,y_f2_alt,marker='.',linestyle='-',color="purple")
ax4.set_title('Alt-GDA for f2')

# Commentaires:
# On remarque que les deux méthodes n'arrivent pas à converger avec la fonction f1. Pour sim-gda elle semble diverger
# et pour alt-gda elle semble juste tourner en rond. 
# Pour la fonction f2 les deux méthodes arrivent à converger. On notera que la méthode sim-gda semble converger légérement 
# plus rapidement


#%%
# QUESTION 20)
R = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])

def f(p,q):
    return p[0]*q[0]*R[0,0]+p[0]*q[1]*R[0,1]+p[0]*q[2]*R[0,2]+p[1]*q[0]*R[1,0]+p[1]*q[1]*R[1,1]+p[1]*q[2]*R[1,2]+p[2]*q[0]*R[2,0]+p[2]*q[1]*R[2,1]+p[2]*q[2]*R[2,2]

def nab_p_f(p,q):
    return np.array([q[0]*R[0,0]+q[1]*R[0,1]+q[2]*R[0,2],q[0]*R[1,0]+q[1]*R[1,1]+q[2]*R[1,2],q[0]*R[2,0]+q[1]*R[2,1]+q[2]*R[2,2]])

def nab_q_f(p,q):
    return np.array([p[0]*R[0,0]+p[1]*R[1,0]+p[2]*R[2,0],p[0]*R[0,1]+p[1]*R[1,1]+p[2]*R[2,1],p[0]*R[0,2]+p[1]*R[1,2]+p[2]*R[2,2]])



#%%
# QUESTION 21)
def proj(p):
    return np.array([(np.exp(p[k])) / (np.sum(np.exp(p))) for k in range(len(p))])
        
def Sim_proj(p_init,gamma,maxiter,eps,f_grad_x,f_grad_y):
    x = p_init[0]
    y = p_init[1]
    result = [p_init]
    for k in range(0,maxiter):
        nab_x = f_grad_x(x,y) # Renvoie un np.array
        nab_y = f_grad_y(x,y) # Renvoie un np.array
        if (np.square(np.linalg.norm(x,ord=2))<=np.square(eps) and np.square(np.linalg.norm(y,ord=2))<=np.square(eps)):
            break
        else:
            x = proj(x-gamma*nab_x)
            y = proj(y+gamma*nab_y)
            result.append(np.array([x,y]))
    return result

def Alt_proj(p_init,gamma,maxiter,eps,f_grad_x,f_grad_y):
    x = p_init[0]
    y = p_init[1]
    result = [p_init]
    for k in range(0,maxiter):
        nab_x = f_grad_x(x,y) # Renvoie un np.array
        if (np.square(np.linalg.norm(x,ord=2))<=np.square(eps) and np.square(np.linalg.norm(y,ord=2))<=np.square(eps)):
            break
        else:
            x = proj(x-gamma*nab_x)
            nab_y = f_grad_y(x,y) # Renvoie un np.array
            y = proj(y+gamma*nab_y)
            result.append(np.array([x,y]))
    return result
#%%
# QUESTION 22)
point = np.array([np.array([0,0.5,0.5]),np.array([1,0,0])])
sim_proj_res = Sim_proj(p_init=point,gamma=0.01,maxiter=100,eps=1e-10,f_grad_x=nab_p_f,f_grad_y=nab_q_f)
alt_proj_res = Alt_proj(p_init=point,gamma=0.01,maxiter=100,eps=1e-10,f_grad_x=nab_p_f,f_grad_y=nab_q_f)

# %%
# QESTION 23)
print("========Sim_proj========") 
print("Meilleure stratégie joueuse A avec sim_proj:",sim_proj_res[-1][0])
print("Meilleure stratégie joueuse B avec sim_proj:",sim_proj_res[-1][1])
print("Gain moyen de la joueuse A avec sim_proj:",f(sim_proj_res[-1][0],sim_proj_res[-1][1]))
print("Gain moyen de la joueuse B avec sim_proj:",f(sim_proj_res[-1][1],sim_proj_res[-1][0]),"\n")

print("=======Alt_proj======")

print("Meilleure stratégie joueuse A avec alt_proj:",alt_proj_res[-1][0])
print("Meilleure stratégie joueuse B avec alt_proj:",alt_proj_res[-1][1])
print("Gain moyen de la joueuse A avec alt_proj:",f(alt_proj_res[-1][0],alt_proj_res[-1][1]))
print("Gain moyen de la joueuse B avec alt_proj:",f(alt_proj_res[-1][1],alt_proj_res[-1][0]))


# Commentaire:
# Pour les deux méthodes, il est normal de voir apparaitre la stratégie de jouer avec une probabilité d'un tier chaque coup pour les deux joueuses. En effet,
# le pierre feuille ciseaux est un jeu de chance, et il n'y a donc pas de stratégie gagnante. En suivant cette stratégie,
# il est normal de voir un gain nul pour les deux joueuses. En effet, elles ont autant de chance de gagner, de perdre et de faire nul.
# Le gain est donc nullifié.


# %%
# QUESTION 24)
# Si la matrice n'est plus asymétrique, alors le jeu ne serait plus équilibré. Il existerait alors une stratégie gagnante pour l'une des deux joueuses.
# Si les gains ne sont plus +1 ou -1: 
# Si les gains et les pertes restent égaux (par exemple: +2 et -2) alors cela ne changerait rien au jeu. 
# Si les gains et les pertes ne sont plus égaux, alors l'une des deux joueuses aurait des gains supérieurs à l'autre. Une joueuse aurait donc une stratégie 
# qui lui permet de maximiser ses gains.