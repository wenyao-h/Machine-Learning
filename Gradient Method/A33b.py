######################################################
#Choose 12 initial points uniformly along the edge
######################################################
import numpy as np
import math

x1_1=[float(-10) for i in range(3)]
x1_2=list(np.arange(-10,0,10/3))
x1_3=[float(0) for i in range(3)]
x1=x1_2+x1_3+x1_2+x1_1

x2_1=list(np.arange(-5,5,10/3))
x2_2=[float(5) for i in range(3)]
x2_3=[float(-5) for i in range(3)]
x2=x2_2+x2_1+x2_3+x2_1

###########################
###########################
#Prepare for the path
###########################
###########################
def obj(x1,x2):
    fx1=-1+x1+((5-x2)*x2-2)*x2
    fx2=-1+x1+((x2+1)*x2-10)*x2
    fx=fx1**2+fx2**2
    return float(fx)


def g1(x1,x2):
    g1=4*x1 + 2*x2*(x2*(5 - x2) - 2) + 2*x2*(x2*(x2 + 1) - 10) - 4
    return float(g1)

def g2(x1,x2):
    g2=(x1 + x2*(x2*(5 - x2) - 2) - 1)*(2*x2*(5 - 2*x2) + 2*x2*(5 - x2) - 4) + (x1 + x2*(x2*(x2 + 1) - 10) - 1)*(2*x2*(x2 + 1) + 2*x2*(2*x2 + 1) - 20)
    return float(g2)



def armijo(x1,x2):
    
    alpha=1
    gamma_arg=0.1
    sigma=0.5

    fxk=obj(x1,x2)
    grad1=g1(x1,x2)
    grad2=g2(x1,x2)   

    while True:
        x1_new=x1+alpha*(-grad1)
        x2_new=x2+alpha*(-grad2)
        fxk_new=obj(x1_new,x2_new)
        remain=gamma_arg*alpha*(-grad1**2-grad2**2)

        alpha=alpha*sigma
        if fxk_new<fxk+remain:             
            break
    return alpha


def golden_section(al_iter,ar_iter,x_1,x_2):
    al=al_iter
    ar=ar_iter
    
    def fa(a,x1,x2):
        return float(obj(x1+a*(-g1(x1,x2)),x2+a*(-g2(x1,x2))))

    theta=(3-(5)**0.5)/2     
    
    iter=1
    while ar-al>.000001:
        iter+=1
        #print("iter: "+str(iter))

        new_al=theta*ar+(1-theta)*al
        new_ar=(1-theta)*ar+theta*al
        #print(xr-xl)
        if fa(new_al,x_1,x_2)<fa(new_ar,x_1,x_2):
            ar=new_ar            
            #print("left"+str(ar))               
        else:
            al=new_al                      
            #print("right"+str(al))            

        #if xr-xl<.000001: break
        
    return (al+ar)/2

#这个改了，没有返回收敛点而是返回了iter经过的所有点
#列表起点为initial points，终点为收敛点，init不同，返回的表的元素数量也不同
def gradient(x1_iter,x2_iter,method):
    
    x1=x1_iter
    x2=x2_iter
    iter=1
    
    #增加list记录path
    x1_list=[x1]
    x2_list=[x2]

    while True:
        if method=="backtrack":
            ss=armijo(x1,x2)
        elif method=="dim":
            ss=0.02/math.log(iter+10)
        elif method=="exactline":
            ss=golden_section(0,2,x1,x2)

        new_x1=x1-ss*g1(x1,x2)
        new_x2=x2-ss*g2(x1,x2)
        iter+=1

        tol=((g1(new_x1,new_x2))**2+(g2(new_x1,new_x2))**2)**0.5
        #print(str(float(tol))+"// x1: "+str(x1)+"// x2: "+str(x2)+"// iter: "+str(iter))
        
        x1=new_x1
        x2=new_x2
        
        #将新的点记录到list中
        x1_list.append(x1)        
        x2_list.append(x2)

        if tol<.00001:
            break

    #return x1,x2
    return x1_list, x2_list




###########################
###########################
#现在开始画图
###########################
###########################


import matplotlib.pyplot as plt


#choose the alg
#2 available choices: "exactline","backtrack"
def alg_choosing(alg):
    
    #choose color for path
    cl_list=['maroon','salmon','rosybrown','sienna','chocolate','sandybrown','peachpuff','burlywood','navajowhite','goldenrod','gold','darkkhaki']

    x1_plt=np.arange(-11,6,0.1)
    x2_plt=np.arange(-6,6,0.1)
    x1_plt,x2_plt=np.meshgrid(x1_plt,x2_plt)

    fx=(-1+x1_plt+((5-x2_plt)*x2_plt-2)*x2_plt)**2+(-1+x1_plt+((x2_plt+1)*x2_plt-10)*x2_plt)**2

    #prepare for the figure
    #contour
    fig=plt.figure()
    plt.contour(x1_plt,x2_plt,fx,2500)

    #paint the path
    for i in range(len(x1)):
        #Generate the path
        point_x1, point_x2 = gradient(x1[i],x2[i],alg)

        #initial points    
        plt.plot(x1[i],x2[i],marker="o",color=cl_list[i])

        #ending points
        plt.plot(point_x1[-1],point_x2[-1],marker="x",color=cl_list[i])

        plt.plot(point_x1,point_x2,color=cl_list[i])

    plt.title('Coutour Plot for Gradient Descent\n--using %s method'%alg)

    plt.show()

alg_choosing("exactline")
alg_choosing("backtrack")


