
#%%
import numpy as np  # package de calcul scientifique
import matplotlib.pyplot as plt  # package graphique
from matplotlib import cm
import scipy.optimize as sp
#%%
print("oh")





# %%

def ftest(x, y):
    return (x - 1) ** 2 + 3*(y + 1)**2

def fGrad (x) :
    return (2*(x[0]-1), 6*(x[1]+1))

def descente (gradiant,xinit,gamma,maxiter,e) :
    x=xinit
    res=[x]
    for i in range (1,maxiter+1) :
        g=gradiant(x)
        if np.linalg.norm(g,ord=2)**2 <= e**2 :
            break
        else :
            x=(x[0]-gamma*g[0],x[1]-gamma*g[1])
            res.append(x)
    return res[-1]

print ("res = ",descente (fGrad,(1,1),0.01,500,0.001))




fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = ftest(X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()






#
#
#
#















# %%

def fab (a,b,x,y) :
    return (y**2)/a + (x**2)/b

def fabGrad (a,b,x) :
    return ((2*x[0])/b,(2*x[1])/a)

def descenteFab (gradiant,xinit,gamma,maxiter,e,a,b) :
    x=xinit
    res=[x]
    for i in range (1,maxiter+1) :
        g=gradiant(a,b,x)
        if np.linalg.norm(g,ord=2)**2 <= e**2 :
            break
        else :
            x=(x[0]-gamma*g[0],x[1]-gamma*g[1])
            res.append(x)
    return res

a=1
b=a
res=descenteFab(fabGrad,(1,1),0.01,1000,0.0001,a,b)
print ("res = ",res)
imgFab=[fab(a,b,x[0],x[1]) for x in res]
print ("imgFab = ",imgFab)




x = np.linspace(1,len(imgFab),len(imgFab))
plt.yscale("log")
plt.xlabel("evaluations")
plt.ylabel("imgFab")
plt.title("f(a,b,x) au cours de la descente de gradiant")
plt.plot(x,imgFab)
plt.subplot()
plt.show()




# %%

a=10
b=a
res1=descenteFab(fabGrad,(1,1),0.01,1000,0.0001,a,b)
imgFab1=[fab(a,b,x[0],x[1]) for x in res1]

a=50
b=a
res2=descenteFab(fabGrad,(1,1),0.01,1000,0.0001,a,b)
imgFab2=[fab(a,b,x[0],x[1]) for x in res2]

a=100
b=a
res3=descenteFab(fabGrad,(1,1),0.01,1000,0.0001,a,b)
imgFab3=[fab(a,b,x[0],x[1]) for x in res3]

fig, ax = plt.subplots()
plt.title("Distance à l’objectif au cours des itérations de l’algorithme de descente de gradient")
plt.yscale("log")
plt.ylabel("y")
plt.xlabel("x")
plt.plot(np.linspace(1, len(res), len(res)), imgFab, label="a=1")
plt.plot(np.linspace(1, len(res1), len(res1)), imgFab1, label="a=10")
plt.plot(np.linspace(1, len(res2), len(res2)), imgFab2, label="a=50")
plt.plot(np.linspace(1, len(res3), len(res3)), imgFab3, label="a=100")
ax.legend()
plt.show()



# %%
x = np.linspace(-2,2,1000)
y = np.linspace(-2,2,1000)
X, Y = np.meshgrid(x, y)

tab1 = fab(1,1,X,Y)
tab2 = fab(10,10,X,Y)
tab3 = fab(50,50,X,Y)
tab4 = fab(100,100,X,Y)


listeDes1 = descenteFab(fabGrad, (1, 1), 0.1, 500, 0.0001,1,1)
x_li, y_li = zip(*listeDes1)  # Séparation des coordonnées x et y
plt.scatter(x_li, y_li, color='red', label='Points de la liste "li"')

plt.contour(X, Y, tab1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour de fab avec alpha et beta = 1')
plt.colorbar(label='f(x, y)')
plt.grid(True)
plt.show()

listeDes2 = descenteFab(fabGrad, (1, 1), 0.1, 500, 0.0001,10,10)
x_li, y_li = zip(*listeDes2)  # Séparation des coordonnées x et y
plt.scatter(x_li, y_li, color='red', label='Points de la liste "li"')

plt.contour(X, Y, tab2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour de fab avec alpha et beta = 10')
plt.colorbar(label='f(x, y)')
plt.grid(True)
plt.show()

listeDes3 = descenteFab(fabGrad, (1, 1), 0.1, 500, 0.0001,50,50)
x_li, y_li = zip(*listeDes3)  # Séparation des coordonnées x et y
plt.scatter(x_li, y_li, color='red', label='Points de la liste "li"')

plt.contour(X, Y, tab3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour de fab avec alpha et beta = 50')
plt.colorbar(label='f(x, y)')
plt.grid(True)
plt.show()

listeDes4 = descenteFab(fabGrad, (1, 1), 0.1, 500, 0.0001,100,100)
x_li, y_li = zip(*listeDes4)  # Séparation des coordonnées x et y
plt.scatter(x_li, y_li, color='red', label='Points de la liste "li"')

plt.contour(X, Y, tab4)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour de fab avec alpha et beta = 100')
plt.colorbar(label='f(x, y)')
plt.grid(True)
plt.show()


# %%
'''
On remarque que plus alpha et beta augmentent, plus la pente diminue donc les pas nous raprochent moins de la solution
'''


#%%

l1Norme = [np.linalg.norm(i,ord=2) for i in listeDes1]
l2Norme = [np.linalg.norm(i,ord=2) for i in listeDes2]
l3Norme = [np.linalg.norm(i,ord=2) for i in listeDes3]
l4Norme = [np.linalg.norm(i,ord=2) for i in listeDes4]

plt.yscale("log")
plt.xlabel("itérations")
plt.ylabel("proximité à la solution")
plt.title("proximité à la solution en fonction de a")
plt.plot(l1Norme,label="a=1")
plt.plot(l2Norme,label="a=10")
plt.plot(l3Norme,label="a=50")
plt.plot(l4Norme,label="a=100")
plt.subplot()
plt.legend()
plt.show()

'''
On remarque que la solution qu'on trouve avec les mêmes parametres est
plus proche de la bonne solution quand alpha et beta sont plus petit
Cela est du au fait que la pente est plus grande lorsque alpha et beta
sont plus petit.
'''


# %%
x = np.linspace(-2,2,1000)
y = np.linspace(-2,2,1000)
X, Y = np.meshgrid(x, y)

tab1 = fab(10,1,X,Y)
tab2 = fab(1,20,X,Y)


listeDes1 = descenteFab(fabGrad, (1, 1), 0.1, 500, 0.0001,10,1)
x_li, y_li = zip(*listeDes1)  # Séparation des coordonnées x et y
plt.scatter(x_li, y_li, color='red', label='Points de la liste "li"')

plt.contour(X, Y, tab1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour de fab avec alpha = 10 et beta = 1')
plt.colorbar(label='f(x, y)')
plt.grid(True)
plt.show()

listeDes2 = descenteFab(fabGrad, (1, 1), 0.1, 500, 0.0001,1,20)
x_li, y_li = zip(*listeDes2)  # Séparation des coordonnées x et y
plt.scatter(x_li, y_li, color='red', label='Points de la liste "li"')

plt.contour(X, Y, tab2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour de fab avec alpha = 1 et beta = 10')
plt.colorbar(label='f(x, y)')
plt.grid(True)
plt.show()

'''
On retourve des problèmes de convergence, dans un sens la pente est plus grande donc
la solution converge rapidement au depart, mais une fois qu'on est assez proche,
la pente est très reduite donc devenir précis prend plus de temps
'''

l1Norme = [np.linalg.norm(i,ord=2) for i in listeDes1]
l2Norme = [np.linalg.norm(i,ord=2) for i in listeDes2]

plt.yscale("log")
plt.xlabel("itérations")
plt.ylabel("proximité à la solution")
plt.title("proximité à la solution en fonction de a")
plt.plot(l1Norme,label="a=10, b=1")
plt.plot(l2Norme,label="a=1, b=20")
plt.subplot()
plt.legend()
plt.show()












#
#
#
#













# %%

import time

def descenteCooFix (grad, xinit, gamma, niter, e, a, b) :
    x=xinit
    res=[x]
    for i in range (1,niter+1) :
        g=grad(a,b,x)
        if np.linalg.norm(g,ord=2)**2 <= e**2 :
            break
        else :
            temp=x.copy()
            for j in range (0,len(g)) :
                temp[j]=x[j]-gamma*g[j]
                x=temp.copy()
                res.append(x)
    return res


x = np.linspace(-2,2,1000)
y = np.linspace(-2,2,1000)
X, Y = np.meshgrid(x, y)

tab1 = fab(1,1,X,Y)
tab2 = fab(1,1,X,Y)


listeDes1 = descenteFab(fabGrad, [1, 1], 0.1, 500, 0.0001,1,1)
x_li, y_li = zip(*listeDes1)  # Séparation des coordonnées x et y
plt.scatter(x_li, y_li, color='red', label='Points de la liste "li"')

plt.contour(X, Y, tab1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour de fab avec alpha = 1 et beta = 1')
plt.colorbar(label='f(x, y)')
plt.grid(True)
plt.show()


listeDes2 = descenteCooFix(fabGrad, [1, 1], 0.1, 500, 0.0001,1,1)
x_li, y_li = zip(*listeDes2)  # Séparation des coordonnées x et y
plt.scatter(x_li, y_li, color='red', label='Points de la liste "li"')

plt.contour(X, Y, tab2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour de fab avec alpha = 1 et beta = 1')
plt.colorbar(label='f(x, y)')
plt.grid(True)
plt.show()


imgFab1=[fab(a,b,x[0],x[1]) for x in listeDes1]
imgFab2=[fab(a,b,x[0],x[1]) for x in listeDes2]

fig, ax = plt.subplots()
plt.title("Distance à l’objectif au cours des itérations de l’algorithme de descente de gradient pour a=b=1")
plt.yscale("log")
plt.ylabel("y")
plt.xlabel("x")
plt.plot(np.linspace(1, len(listeDes1), len(listeDes1)), imgFab1, label="descenteGrad")
plt.plot(np.linspace(1, len(listeDes2), len(listeDes2)), imgFab2, label="descenteGradCooFix")
plt.legend()
plt.show()



x = np.linspace(-2,2,1000)
y = np.linspace(-2,2,1000)
X, Y = np.meshgrid(x, y)

tab1 = fab(10,10,X,Y)
tab2 = fab(10,10,X,Y)

listeDes1 = descenteFab(fabGrad, [1, 1], 0.1, 500, 0.0001,10,10)
x_li, y_li = zip(*listeDes1)  # Séparation des coordonnées x et y
plt.scatter(x_li, y_li, color='red', label='Points de la liste "li"')

plt.contour(X, Y, tab1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour de fab avec alpha = 10 et beta = 10')
plt.colorbar(label='f(x, y)')
plt.grid(True)
plt.show()

listeDes2 = descenteCooFix(fabGrad, [1, 1], 0.1, 500, 0.0001,10,10)
x_li, y_li = zip(*listeDes2)  # Séparation des coordonnées x et y
plt.scatter(x_li, y_li, color='red', label='Points de la liste "li"')

plt.contour(X, Y, tab2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour de fab avec alpha = 10 et beta = 10')
plt.colorbar(label='f(x, y)')
plt.grid(True)
plt.show()


imgFab1=[fab(a,b,x[0],x[1]) for x in listeDes1]
imgFab2=[fab(a,b,x[0],x[1]) for x in listeDes2]

fig, ax = plt.subplots()
plt.title("Distance à l’objectif au cours des itérations de l’algorithme de descente de gradient pour a=b=10")
plt.yscale("log")
plt.ylabel("y")
plt.xlabel("x")
plt.plot(np.linspace(1, len(listeDes1), len(listeDes1)), imgFab1, label="descenteGrad")
plt.plot(np.linspace(1, len(listeDes2), len(listeDes2)), imgFab2, label="descenteGradCooFix")
plt.legend()
plt.show()




'''

Pour les linges de niveau, on constate bien le decalement a chasue étape du
aux modification successives de chaques coordonnée

On constate que l'algo par coordonnée converge moins vite
C'est logique car on ajoute le point à chaque modification
de coordonnée, il faut donc 2 fois plus de temps


'''



# %%


'''
La méthode de desente de gradient ne fait pas partie du
catalogue de méthodes disponibles dans la fonction minimize

'''

def f20(x):
    return fab(20, 20, x[0], x[1])

res1 = sp.minimize(f20, (1, 1), method="Nelder-Mead")

res2 = sp.minimize(f20, (1, 1), method="CG")
print(res1)
print(res2)


# %%
def fr(x):
    return (1-x[0])**2+100*(x[1]-x[0]**2)**2

