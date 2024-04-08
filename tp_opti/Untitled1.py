#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  # package de calcul scientifique
import torch  # librairie pytorch
import matplotlib.pyplot as plt  # graphiques


# In[2]:


data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data, x_data.shape)


# In[3]:


numpy_array = np.array(data)
print(type(numpy_array))
tensor_from_numpy = torch.from_numpy(numpy_array)
print(type(tensor_from_numpy))


# In[7]:


A = torch.ones(20,5)
b = torch.ones(5,1)*2
print(A[:,0])
A[5,2] = 0
print(A@b)


# In[18]:


def f(x, y):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    return torch.exp(1-(2*torch.log(y))/(torch.cos(x)+2))


x = np.arange(-1, 6, 0.1)
y = np.arange(1, 6, 0.1)
X, Y = np.meshgrid(x,y)
Z = f(X,Y)

fig = plt.figure(figsize=(12, 10))
ax = plt.axes(projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.cividis)
ax.set_xlabel("x", labelpad=20)
ax.set_ylabel("y", labelpad=20)
ax.set_zlabel("z", labelpad=20)
ax.view_init(10, 25)  # élévation de 10 degrés et déplacement horizontal de 25 degrés
fig.colorbar(surf, shrink=0.5, aspect=8)
plt.show()


# In[19]:


xy = torch.randn(2, requires_grad=True)  # point initial en (x,y) aléatoir
out_1 = 1 - 2 * torch.log(xy[1]) / (torch.cos(xy[0]) + 2)
out_f = torch.exp(out_1)
print(f"Gradient au cours de la chaîne = {out_1.grad_fn}")
print(f"Gradient final = {out_f.grad_fn}")


# In[32]:


pt1 = torch.tensor([1.5,2.0])   #Définition du point
pt1.requires_grad_(True)  # Activation du suivit du gradient
out_1 = 1 - 2 * torch.log(pt1[1]) / (torch.cos(pt1[0]) + 2)
out_f = torch.exp(out_1) # Evaluation de la fonction en ce point
out_f.backward()   # Fait remonter les étapes 
print("Gradient en (1.5, 2): ",pt1.grad)  # Affiche le gradient en ce point 

pt2 = torch.tensor([0.0,1.0])
pt2.requires_grad_(True)
out_2 = 1 - 2 * torch.log(pt2[1]) / (torch.cos(pt2[0]) + 2)
out_f2 = torch.exp(out_2)
out_f2.backward()
print("Gradient en (0, 1): ",pt2.grad)


# In[33]:


x = torch.randn(1, requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.1)  # descente de gradient
for i in range(101):
    optimizer.zero_grad()  # on remet à 0 l'arbre des gradients
    fx = x**2
    fx.backward()  # calcul des gradients
    optimizer.step()  # pas de la descente
    if i % 10 == 0:
        print(x)  # itérés succesifs toutes les 10 itérations
print(x.detach().numpy())  # afficher en numpy la solution


# In[50]:


def f(x, y):
    return 100*(y-x**2)**2+(1-x)**2+2

ll = []
x = torch.randn(2, requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.001)  # descente de gradient
for i in range(1,101):
    optimizer.zero_grad()  # on remet à 0 l'arbre des gradients
    fx = f(x[0],x[1])
    ll.append(fx)
    fx.backward()  # calcul des gradients
    optimizer.step()  # pas de la descente

print("Le dernier gradient calculé vaut:", x.grad)
print("Le dernier itéré est:", ll[-1])
ll = [z.detach().numpy() for z in ll]
plt.figure() 
plt.plot(list(range(len(ll))), ll, color="blue")
plt.xlabel("k")
plt.ylabel(r"(f(x_k,y_k))")

plt.tight_layout()
plt.show()


# In[42]:





# In[43]:





# In[ ]:




