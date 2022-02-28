from matplotlib import markers
import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt
import random

np.random.seed(1234)
#Fix these parameters in advance.
m,n,s=300,3000,30
mu=1
delta=1e-4

#Construct these signals
x_star=np.zeros((n,1))
x_star_mask=np.random.randn(s).reshape((s,1))
mask=[round(random.uniform(0,3000)) for i in range(s)]
mask

for i in range(len(mask)):
    x_star[mask[i]]=x_star_mask[i]

# x_star.shape
# plt.plot(x_star)
# plt.show()

A=np.random.randn(m*n).reshape((m,n))
b=A.dot(x_star)+0.01*np.random.randn(m).reshape((m,1))

Lip_1=2*mu+norm(A.T.dot(A),ord=2)
Lip_2=mu*(1/delta)+norm(A.T.dot(A),ord=2)
alpha_1=1/Lip_1
alpha_2=1/Lip_2

#Construct obj functions
class func_1():
    def obj(self, x):
       phi_1=norm(x)**2
       f=.5*norm(A.dot(x)-b)**2+mu*phi_1
       return f
    
    def gradient(self,x):
        g_phi_1=2*x
        gra=A.T.dot(A.dot(x)-b)+mu*g_phi_1
        return gra

class func_2():
    def obj(self, x):
       phi_2=self.phi_2(x)
       f=.5*norm(A.dot(x)-b,ord=2)**2+mu*phi_2
       return f
    
    
    def phi_2(self,x):
        temp=0
        #这里的x实际上是个n*1的array，所以导出的是个array
        #如果不改变x的形状，i是个1*1的array，会影响后续
        for i in x:
            if abs(i)<=delta:
                res = (1/(2*delta))*i**2
            else:
                res = abs(i)-0.5*delta
            temp+=res
        return float(temp)

    def g_phi_2(self,x):
        temp=[]
        for i in x:
            if abs(float(i))<=delta:
                temp.append((1/delta)*float(i))
            else:
                if float(i)>0:
                    temp.append(1)
                else:
                    temp.append(-1)
        temp=np.array(temp).reshape((n,1))
        return temp


    def gradient(self,x):        
        g_phi_2=self.g_phi_2(x)
        gra=(A.T.dot(A.dot(x)-b)).reshape((n,1))+mu*g_phi_2
        return gra   

x_0=np.zeros((n,1))
# x=np.array([i for i in range(n)]).reshape((n,1))
# x.reshape((1,n))
# f1=func_1()
# f1.obj(x_0)
# f1.gradient(x_0)

# f2=func_2()
# f2.phi_2(x_0)
# f2.obj(x_0)
# f2.g_phi_2(x_0)
# f2.gradient(x_0)

############################
#Accelerated Gradient Method
############################

def AGM(x_init,phi=1):
    #k=0时:    
    #y1=x0+beta_0*(x[0]-x[0])，公式后面的式子实际上等于0
    if phi==1:
        f=func_1()
        alpha=alpha_1
    elif phi==2: 
        f=func_2()
        alpha=alpha_2

    x_k=x_init #x0
    x_old=x_k #x-1
    y_new=x_k #y1
    x_new=y_new-alpha*f.gradient(y_new) #x1
    
    #print('x_1: '+str(x_new[:5]))

    k=1
    t_old=1 #t0
    #k>=1时:
    x_k=x_new #更新current_x, 即是x1
    
    tol_list=[]
    
    while True:
        t_k=0.5*(1+(1+4*t_old**2)**0.5)
        beta_k=(t_old-1)/t_k
        y_new=x_k+beta_k*(x_k-x_old)
        x_new=y_new-alpha*f.gradient(y_new)
        

        k+=1

        tol=norm(f.gradient(x_new),ord=2)
        tol_list.append(tol)
        #print('iter: '+str(k)+'///// tol: '+str(tol)+'\nx_new: '+str(x_new[:3]))

        if tol<1e-4:
            break
        
        #Update        
        t_old=t_k
        x_old=x_k
        x_k=x_new

    

    print('iter: '+str(k))
    return x_new, tol_list

x_AGM_1, tol_AGM_1=AGM(x_0,1)
x_AGM_2, tol_AGM_2=AGM(x_0,2)

############################
#Std. Gradient Method
############################

def armijo(x,phi=1):
    if phi==1:
        f=func_1()
    elif phi==2:
        f=func_2()
    

    alpha=1
    gamma_arg=0.1
    sigma=0.5

    fxk=f.obj(x)
    grad=f.gradient(x)
    
    while True:
        x_new=x+alpha*(-grad)        
        fxk_new=f.obj(x_new)
        remain=gamma_arg*alpha*grad.T.dot(-grad)
        
        #print('alpha:', alpha)

        if fxk_new<fxk+remain:             
            break

        alpha=alpha*sigma      

    return alpha

def gradient(x_iter,phi=1):
    if phi==1:
        f=func_1()
    elif phi==2:
        f=func_2()
    
    x=x_iter
    iter=1
    gamma_arg=0.1
    sigma=0.5
    
    tol_list=[]
    
    while True:

        fxk=f.obj(x)
        grad=f.gradient(x)
        
        alpha=1

        while True:
            x_new=x+alpha*(-grad)        
            fxk_new=f.obj(x_new)
            remain=gamma_arg*alpha*grad.T.dot(-grad)
            
            if fxk_new<fxk+remain:             
                break

            alpha=alpha*sigma

        x_new=x-alpha*f.gradient(x)
        iter+=1

        tol=norm(f.gradient(x_new),ord=2)
        tol_list.append(tol)
        if iter%100==0:
            print('iter: '+str(iter)+'///// tol: '+str(tol))
        
        
        

        if tol<1e-4:
            break

        x=x_new

    #print('iter: '+str(iter))
    return x_new,tol_list

x_GM_1,tol_GM_1=gradient(x_0,1)
x_GM_2,tol_GM_2=gradient(x_0,2)

############################
#Visualization
############################

def vis_AGM():
    #x_AGM_1=AGM(x_0,1)
    #x_AGM_2=AGM(x_0,2)
    #x_GM_1=gradient(x_0,1)
    #x_GM_2=gradient(x_0,2)



    fig=plt.figure(figsize=(10,10))
    ax1=fig.add_subplot(2,2,1)
    ax1.plot(x_AGM_1,marker="x",color='green',label='Reconstructed Signals')
    ax1.plot(x_star,color='orange',label='True Signals')
    ax1.legend()
    ax1.title.set_text("Using AGM and φ1")

    ax2=fig.add_subplot(2,2,2)
    ax2.plot(x_AGM_2,marker="x",color='green',label='Reconstructed Signals')
    ax2.plot(x_star,color='orange',label='True Signals')
    ax2.legend()
    ax2.title.set_text("Using AGM and φ2")

    ax3=fig.add_subplot(2,2,3)
    ax3.plot(x_GM_1,marker="x",color='green',label='Reconstructed Signals')
    ax3.plot(x_star,color='orange',label='True Signals')
    ax3.legend()
    ax3.title.set_text("Using Std.GM and φ1")    

    ax4=fig.add_subplot(2,2,4)
    ax4.plot(x_GM_2,marker="x",color='green',label='Reconstructed Signals')
    ax4.plot(x_star,color='orange',label='True Signals')
    ax4.legend()
    ax4.title.set_text("Using Std.GM and φ2")

    plt.show()

vis_AGM()


# import pandas as pd
# data=pd.Series(tol_GM_1)
# data.to_csv("C:/Users/selis/Desktop/tol_GM_1.csv")
# data=pd.Series(tol_AGM_1)
# data.to_csv("C:/Users/selis/Desktop/tol_AGM_1.csv")
# data=pd.Series(tol_GM_2)
# data.to_csv("C:/Users/selis/Desktop/tol_GM_2.csv")
# data=pd.Series(tol_AGM_2)
# data.to_csv("C:/Users/selis/Desktop/tol_AGM_2.csv")

# tol_GM_1=pd.read_csv("C:/Users/selis/Desktop/tol_GM_1.csv").iloc[:,1]
# tol_AGM_1=pd.read_csv("C:/Users/selis/Desktop/tol_AGM_1.csv").iloc[:,1]
# tol_GM_2=pd.read_csv("C:/Users/selis/Desktop/tol_GM_2.csv").iloc[:,1]
# tol_AGM_2=pd.read_csv("C:/Users/selis/Desktop/tol_AGM_2.csv").iloc[:,1]

#def comparison():

# fig1=plt.figure(1)
# plt.plot(np.array([i for i in range(len(tol_AGM_1))]),tol_AGM_1,color='green',label="AGM")
# plt.plot(np.array([i for i in range(len(tol_GM_1))]),tol_GM_1,color='orange',label="Std. Gradient")

# plt.yscale('log')
# plt.xlabel('#Iteration')
# plt.ylabel('tol')
# plt.title("Using φ1")
# plt.legend()
def comparison():
    fig=plt.figure(figsize=(10,4))
    ax1=fig.add_subplot(1,2,1)
    ax2=fig.add_subplot(1,2,2)

    ax1.plot(np.arange(len(tol_AGM_1)),tol_AGM_1,color='green',label="AGM")
    ax1.plot(np.arange(len(tol_GM_1)),tol_GM_1,color='orange',label="Std. Gradient")
    ax1.set_yscale('log')
    ax1.set_xlabel('    #Iteration')
    ax1.set_ylabel('tol')
    ax1.title.set_text("Using φ1")
    ax1.legend()

    ax2.plot(np.arange(len(tol_AGM_2)),tol_AGM_2,color='green',label="AGM")
    ax2.plot(np.arange(len(tol_GM_2)),tol_GM_2,color='orange',label="Std. Gradient")
    ax2.set_yscale('log')
    ax2.set_xlabel('#Iteration')
    ax2.set_ylabel('tol')
    ax2.title.set_text("Using φ2")
    ax2.legend()

    plt.show()


comparison()

