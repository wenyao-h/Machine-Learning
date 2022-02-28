#from sympy import *
import math
# class func():

#     def obj(self,x1,x2):
#         fx1=-1+x1+((5-x2)*x2-2)*x2
#         fx2=-1+x1+((x2+1)*x2-10)*x2
#         fx=fx1**2+fx2**2
#         return fx

#     def g1(self,x_1,x_2):
#         x1,x2=symbols("x1,x2")
#         grad1=diff(self.obj(x1,x2),x1)
#         return grad1.evalf(subs ={'x1':x_1,'x2':x_2})

#     def g2(self,x_1,x_2):
#         x1,x2=symbols("x1,x2")
#         grad2=diff(self.obj(x1,x2),x2)
#         return grad2.evalf(subs ={'x1':x_1,'x2':x_2})

#To save time and space

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

# res=golden_section(0, 2,-5,0)
# fa(res-0.00001,-5,0)

def gradient(x1_iter,x2_iter,method):
    
    x1=x1_iter
    x2=x2_iter
    iter=1

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
        print(str(float(tol))+"// x1: "+str(x1)+"// x2: "+str(x2)+"// iter: "+str(iter))
        
        x1=new_x1
        x2=new_x2

        if tol<.00001:
            break

    return x1, x2


gradient(-5,0,"backtrack")
gradient(-5,0,"dim")
gradient(-5,0,"exactline")