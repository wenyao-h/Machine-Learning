import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt
from sympy import *

# def obj(x1,x2):
#     fx=100*(x2-x1**2)**2+(1-x1)**2
#     return fx
# def g1(x1,x2):
#     g1=-400*x1*(-x1**2 + x2) + 2*x1 - 2
#     return g1
# def g2(x1,x2):
#     g2=-200*x1**2 + 200*x2
#     return g2 

# x1,x2=symbols("x1,x2")
# diff(obj(x1,x2),x1)
# diff(g1(x1,x2),x1)

class func():
    def obj(self,x1,x2):
        fx=100*(x2-x1**2)**2+(1-x1)**2
        return fx

    def g1(self,x1,x2):
        g1=-400*x1*(-x1**2 + x2) + 2*x1 - 2
        return g1

    def g2(self,x1,x2):
        g2=-200*x1**2 + 200*x2
        return g2
    
    def h11(self,x1,x2):
        h11=1200*x1**2 - 400*x2 + 2
        return h11
    
    def h12(self,x1,x2):
        h12=-400*x1
        return h12

    def h21(self,x1,x2):
        h12=-400*x1
        return h12
    
    def h22(self,x1,x2):
        h22=200
        return h22

    def hessian(self,x1,x2):
        h=np.array([[self.h11(x1,x2),self.h12(x1,x2)],\
                   [self.h21(x1,x2),self.h22(x1,x2)]])
        return h
    
    def gra(self,x1,x2):
        gra=np.array([self.g1(x1,x2),self.g2(x1,x2)])
        return gra

###########################
#Newton Method
###########################

#return the choice of program on newton or gradient direction
def dir(x1,x2):
    f=func()
    beta1=1e-6
    beta2=0.1
    newton_dir=-inv(f.hessian(x1,x2)).dot(f.gra(x1,x2))
    _LHS=-f.gra(x1,x2).T.dot(newton_dir)
    _RHS=beta1*min(1,norm(newton_dir)**beta2)*norm(newton_dir)**2

    if _LHS>=_RHS and f.gra(x1,x2).T.dot(newton_dir)<0:
        print("use newton direction")
        return newton_dir
    
    else:
        print("use gradient direction")
        return np.array([-f.g1(x1,x2),-f.g2(x1,x2)])

#no change except for default values
def newton_backtrack(x1,x2):
    f=func()

    alpha=1
    gamma_arg=1e-4
    sigma=0.5

    
    fxk=f.obj(x1,x2)

    _dir=dir(x1,x2)

    while True:
        x1_new=x1+alpha*_dir[0]
        x2_new=x2+alpha*_dir[1]
        fxk_new=f.obj(x1_new,x2_new)
        remain=gamma_arg*alpha*(f.gra(x1,x2).T.dot(_dir))      

        if fxk_new<fxk+remain:             
            break
        alpha=alpha*sigma
    return alpha

#比a题多加了一个记录和tol_limit，返回iteration
def newton_compare(x1_iter,x2_iter,tol_limit):
    
    x1=x1_iter
    x2=x2_iter
    iter=1
    
    #增加list记录path
    x1_list=[x1]
    x2_list=[x2]
    tol_list=[]

    while True:
        f=func()
        ss=newton_backtrack(x1,x2)

        #记录步长
        print('alpha: '+str(ss))
        _dir=dir(x1,x2)

        new_x1=x1+ss*_dir[0]
        new_x2=x2+ss*_dir[1]
        iter+=1

        tol=norm(f.gra(new_x1,new_x2))
        tol_list.append(tol)
        #tol=((f.g1(new_x1,new_x2))**2+(f.g2(new_x1,new_x2))**2)**0.5
    
        #print(str(float(tol))+"// x1: "+str(x1)+"// x2: "+str(x2)+"// iter: "+str(iter))
        
        x1=new_x1
        x2=new_x2
        
        #将新的点记录到list中
        x1_list.append(x1)        
        x2_list.append(x2)

        if tol<tol_limit:
            break

    #return x1,x2
    #return x1_list, x2_list
    return tol_list


###########################
#Gradient Method
###########################

#no change except for default values
def armijo(x1,x2):
    f=func()
    alpha=1
    gamma_arg=1e-4
    sigma=0.5

    fxk=f.obj(x1,x2)
    grad1=f.g1(x1,x2)
    grad2=f.g2(x1,x2)   

    while True:
        x1_new=x1+alpha*(-grad1)
        x2_new=x2+alpha*(-grad2)
        fxk_new=f.obj(x1_new,x2_new)
        remain=gamma_arg*alpha*(-grad1**2-grad2**2)

        
        if fxk_new<fxk+remain:             
            break
        alpha=alpha*sigma
    
    return alpha

#多加了一个记录和tol_limit，返回iteration
def gradient_compare(x1_iter,x2_iter,tol_limit):
    f=func()
    x1=x1_iter
    x2=x2_iter
    iter=1
    
    #增加list记录path
    x1_list=[x1]
    x2_list=[x2]
    tol_list=[]

    while True:

        ss=armijo(x1,x2)
        print('alpha: '+str(ss))

        new_x1=x1-ss*f.g1(x1,x2)
        new_x2=x2-ss*f.g2(x1,x2)
        iter+=1

        tol=((f.g1(new_x1,new_x2))**2+(f.g2(new_x1,new_x2))**2)**0.5
        tol_list.append(tol)
        #print(str(float(tol))+"// x1: "+str(x1)+"// x2: "+str(x2)+"// iter: "+str(iter))
        
        x1=new_x1
        x2=new_x2
        
        #将新的点记录到list中
        x1_list.append(x1)        
        x2_list.append(x2)

        if tol<tol_limit:
            break

    #return x1,x2
    #return x1_list, x2_list
    return tol_list


###########################
#Compare the Permance, from point (-1.2,1)
###########################
x1=-1.2
x2=1
tol_limit=[1e-1,1e-3,1e-5]

#Rounds of iteration
len(newton_compare(x1,x2,1e-1))
len(newton_compare(x1,x2,1e-1))

iter_n=[]
iter_g=[]

for i in tol_limit:
    iter_n.append(len(newton_compare(x1,x2,i)))
    iter_g.append(len(gradient_compare(x1,x2,i)))
iter_n
iter_g

#covergence

def visualization():
    iter_g=gradient_compare(-5,0,1e-5)
    iter_n=newton_compare(-5,0,1e-5)
    plt.plot(np.arange(len(iter_g)),iter_g,color='orange',label="Gradient Method")
    plt.plot(np.arange(len(iter_n)),iter_n,color='green',label="Newton Method")
    plt.yscale('log')
    plt.xlabel("#iteration")
    plt.ylabel("tol")
    plt.legend()
    plt.title('Comparison on Convergence between Gradient and Newton')
    plt.show()    

    
visualization()