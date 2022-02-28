import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt


class func():
    def obj(self,x1,x2):
        fx1=-1+x1+((5-x2)*x2-2)*x2
        fx2=-1+x1+((x2+1)*x2-10)*x2
        fx=fx1**2+fx2**2
        return fx

    def g1(self,x1,x2):
        g1=4*x1 + 2*x2*(x2*(5 - x2) - 2) + 2*x2*(x2*(x2 + 1) - 10) - 4
        return g1

    def g2(self,x1,x2):
        g2=(x1 + x2*(x2*(5 - x2) - 2) - 1)*(2*x2*(5 - 2*x2) + 2*x2*(5 - x2) - 4) + (x1 + x2*(x2*(x2 + 1) - 10) - 1)*(2*x2*(x2 + 1) + 2*x2*(2*x2 + 1) - 20)
        return g2
    
    def h11(self,x1,x2):
        h11=4
        return h11
    
    def h12(self,x1,x2):
        h12=2*x2*(5 - 2*x2) + 2*x2*(5 - x2) + 2*x2*(x2 + 1) + 2*x2*(2*x2 + 1) - 24
        return h12

    def h21(self,x1,x2):
        h12=2*x2*(5 - 2*x2) + 2*x2*(5 - x2) + 2*x2*(x2 + 1) + 2*x2*(2*x2 + 1) - 24
        return h12
    
    def h22(self,x1,x2):
        h22=(20 - 12*x2)*(x1 + x2*(x2*(5 - x2) - 2) - 1) + (12*x2 + 4)*(x1 + x2*(x2*(x2 + 1) - 10) - 1) + (x2*(5 - 2*x2) + x2*(5 - x2) - 2)*(2*x2*(5 - 2*x2) + 2*x2*(5 - x2) - 4) + (x2*(x2 + 1) + x2*(2*x2 + 1) - 10)*(2*x2*(x2 + 1) + 2*x2*(2*x2 + 1) - 20)
        return h22

    def hessian(self,x1,x2):
        h=np.array([[self.h11(x1,x2),self.h12(x1,x2)],\
                   [self.h21(x1,x2),self.h22(x1,x2)]])
        return h
    
    def gra(self,x1,x2):
        gra=np.array([self.g1(x1,x2),self.g2(x1,x2)])
        return gra

######################################################
#Choose 12 initial points uniformly along the edge
######################################################

x1_1=[float(-10) for i in range(3)]
x1_2=list(np.arange(-10,0,10/3))
x1_3=[float(0) for i in range(3)]
x1=x1_2+x1_3+x1_2+x1_1

x2_1=list(np.arange(-5,5,10/3))
x2_2=[float(5) for i in range(3)]
x2_3=[float(-5) for i in range(3)]
x2=x2_2+x2_1+x2_3+x2_1


###########################
#Prepare for the path
###########################

#newton direction
def dir(x1,x2):
    f=func()
    beta1=1e-6
    beta2=0.1
    newton_dir=-inv(f.hessian(x1,x2)).dot(f.gra(x1,x2))
    _LHS=-f.gra(x1,x2).T.dot(newton_dir)
    _RHS=beta1*min(1,norm(newton_dir,ord=2)**beta2)*norm(newton_dir,ord=2)**2

    if _LHS>=_RHS and f.gra(x1,x2).T.dot(newton_dir)<0:
        return newton_dir
    
    else:
        return np.array([-f.g1(x1,x2),-f.g2(x1,x2)])

def newton_backtrack(x1,x2):
    f=func()

    alpha=1
    gamma_arg=0.1
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


def newton(x1_iter,x2_iter):
    
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
        _dir=dir(x1,x2)
        

        new_x1=x1+ss*_dir[0]
        new_x2=x2+ss*_dir[1]
        iter+=1

        tol=norm(f.gra(new_x1,new_x2))
        
        #tol=((f.g1(new_x1,new_x2))**2+(f.g2(new_x1,new_x2))**2)**0.5
        
        #print(str(float(tol))+"// x1: "+str(x1)+"// x2: "+str(x2)+"// iter: "+str(iter))
        
        x1=new_x1
        x2=new_x2
        
        #将新的点记录到list中
        x1_list.append(x1)        
        x2_list.append(x2)

        if tol<1e-8:
            break

    #return x1,x2
    return x1_list, x2_list

# i=11
# newton(x1[i],x2[i])
# x1[i],x2[i]

# point_x1, point_x2 = newton(x1[i],x2[i])
# point_x1, point_x2


###########################
#Contour Plot
###########################    

def draw_contour():
    cl_list=['maroon','salmon','rosybrown','sienna','chocolate','sandybrown','peachpuff','burlywood','navajowhite','goldenrod','gold','darkkhaki']

    x1_plt=np.arange(-80,6,0.1)
    x2_plt=np.arange(-6,6,0.1)
    x1_plt,x2_plt=np.meshgrid(x1_plt,x2_plt)

    fx=(-1+x1_plt+((5-x2_plt)*x2_plt-2)*x2_plt)**2+(-1+x1_plt+((x2_plt+1)*x2_plt-10)*x2_plt)**2

    #prepare for the figure
    #contour
    fig=plt.figure()
    plt.contour(x1_plt,x2_plt,fx,1500)

    #paint the path
    for i in range(len(x1)):
        #Generate the path
        point_x1, point_x2 = newton(x1[i],x2[i])

        #initial points    
        plt.plot(x1[i],x2[i],marker="o",color=cl_list[i])

        #ending points
        plt.plot(point_x1[-1],point_x2[-1],marker="x",color=cl_list[i])

        plt.plot(point_x1,point_x2,color=cl_list[i])

    plt.title('Coutour Plot for Newton`s Method')

    plt.show()

draw_contour()

###########################
#Compare the Permance, from point (-5,0)
###########################


#Gradient Method
def armijo(x1,x2):
    f=func()

    alpha=1
    gamma_arg=0.1
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

#修改后的公式用||xk-x*||作为tolerance，并指定不同级数作为limit，返回iter次数
def gradient_compare(x1_iter,x2_iter,tol_limit):
    
    f=func()
    x1=x1_iter
    x2=x2_iter
    iter=1
    
    #增加list记录path
    x1_list=[x1]
    x2_list=[x2]
    tol_list1, tol_list2, tol_list3=[],[],[]
    while True:
        f=func()
        ss=armijo(x1,x2)


        new_x1=x1-ss*f.g1(x1,x2)
        new_x2=x2-ss*f.g2(x1,x2)
        iter+=1

        tol1=((x1-(-11))**2+(x2-(1+5**0.5))**2)**0.5
        tol2=((x1-(-11))**2+(x2-(1-5**0.5))**2)**0.5
        tol3=((x1-(1))**2+(x2-(0))**2)**0.5
        tol_list1.append(tol1)
        tol_list2.append(tol2)
        tol_list3.append(tol3)
        #print("x1: "+str(x1)+"// x2: "+str(x2)+"// iter: "+str(iter)+"// tol: "+str(tol2))
        
        x1=new_x1
        x2=new_x2
        
        #将新的点记录到list中
        x1_list.append(x1)        
        x2_list.append(x2)
        #print(str((tol1<tol_limit or tol2<tol_limit or tol3<tol_limit)))
        if (tol1<tol_limit or tol2<tol_limit or tol3<tol_limit) :
            break
        
    if tol_list1[-1]==min(tol_list1[-1],tol_list2[-1],tol_list3[-1]):
        tol_list=tol_list1
    elif tol_list2[-1]==min(tol_list1[-1],tol_list2[-1],tol_list3[-1]):
        tol_list=tol_list2
    elif tol_list3[-1]==min(tol_list1[-1],tol_list2[-1],tol_list3[-1]):
        tol_list=tol_list3
    #return x1,x2
    #dis=norm(np.array([x1,x2]),)
    return tol_list
#norm(np.array([[-5,0],[-11, 1-5**0.5]]))
#gradient_compare(-5,0,1)

def newton_compare(x1_iter,x2_iter,tol_limit):
    
    x1=x1_iter
    x2=x2_iter
    iter=1
    
    #增加list记录path
    x1_list=[x1]
    x2_list=[x2]
    tol_list1, tol_list2, tol_list3=[],[],[]
    while True:
        f=func()
        ss=newton_backtrack(x1,x2)
        _dir=dir(x1,x2)
        

        new_x1=x1+ss*_dir[0]
        new_x2=x2+ss*_dir[1]
        iter+=1

        #tol=norm(f.gra(new_x1,new_x2))
        tol1=((x1-(-11))**2+(x2-(1+5**0.5))**2)**0.5
        tol2=((x1-(-11))**2+(x2-(1-5**0.5))**2)**0.5
        tol3=((x1-(1))**2+(x2-(0))**2)**0.5
        tol_list1.append(tol1)
        tol_list2.append(tol2)
        tol_list3.append(tol3)       
        
        #print("// x1: "+str(x1)+"// x2: "+str(x2)+"// iter: "+str(iter))
        
        x1=new_x1
        x2=new_x2
        
        #将新的点记录到list中
        x1_list.append(x1)        
        x2_list.append(x2)
        #print(str((tol1<tol_limit or tol2<tol_limit or tol3<tol_limit)))

        if (tol1<tol_limit or tol2<tol_limit or tol3<tol_limit):
            break

    if tol_list1[-1]==min(tol_list1[-1],tol_list2[-1],tol_list3[-1]):
        tol_list=tol_list1
    elif tol_list2[-1]==min(tol_list1[-1],tol_list2[-1],tol_list3[-1]):
        tol_list=tol_list2
    elif tol_list3[-1]==min(tol_list1[-1],tol_list2[-1],tol_list3[-1]):
        tol_list=tol_list3

    #return x1,x2
    return tol_list


# Visualisation on the comparison
# and some comments on the average performance in terms of iterations
def comparison(tol=1e-6):
    iter_g=[]
    iter_n=[]

    for j in range(len(x1)):
        iter_g.append(len(gradient_compare(x1[j],x2[j],tol)))
        iter_n.append(len(newton_compare(x1[j],x2[j],tol)))

    print(f'For Gradient Method, to reach the tolerance of {tol}, on average it requires {np.array(iter_g).mean():.1f} rounds of iteration;\nwhile for Newton Method, on average it requires {np.array(iter_n).mean():.1f} rounds.')
comparison(1e-6)

def visualization():
    iter_g=gradient_compare(-5,0,1e-6)
    iter_n=newton_compare(-5,0,1e-6)
    plt.plot(np.arange(len(iter_g)),iter_g,color='orange',label="Gradient Method")
    plt.plot(np.arange(len(iter_n)),iter_n,color='green',label="Newton Method")
    plt.yscale('log')
    plt.xlabel("#iteration")
    plt.ylabel("tol")
    plt.legend()
    plt.title('Comparison on Convergence between Gradient and Newton')
    plt.show()    

    
visualization()

