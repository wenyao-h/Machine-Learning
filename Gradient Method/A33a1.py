from sympy import *
import numpy as np
from numpy.linalg import eig

class func():

    def obj(self,x1,x2):
        fx1=-1+x1+((5-x2)*x2-2)*x2
        fx2=-1+x1+((x2+1)*x2-10)*x2
        fx=fx1**2+fx2**2
        return fx

    def g1(self,x_1,x_2):
        x1,x2=symbols("x1,x2")
        grad1=diff(self.obj(x1,x2),x1)
        return grad1.evalf(subs ={'x1':x_1,'x2':x_2})

    def g2(self,x_1,x_2):
        x1,x2=symbols("x1,x2")
        grad2=diff(self.obj(x1,x2),x2)
        return grad2.evalf(subs ={'x1':x_1,'x2':x_2})
    
    def h11(self,x_1,x_2):
        x1,x2=symbols("x1,x2")
        hessian11=diff(self.g1(x1,x2),x1)
        return hessian11.evalf(subs ={'x1':x_1,'x2':x_2})

    def h12(self,x_1,x_2):
        x1,x2=symbols("x1,x2")
        hessian12=diff(self.g1(x1,x2),x2)
        return hessian12.evalf(subs ={'x1':x_1,'x2':x_2})
    
    def h21(self,x_1,x_2):
        x1,x2=symbols("x1,x2")
        hessian21=diff(self.g2(x1,x2),x1)
        return hessian21.evalf(subs ={'x1':x_1,'x2':x_2})

    def h22(self,x_1,x_2):
        x1,x2=symbols("x1,x2")
        hessian22=diff(self.g2(x1,x2),x2)
        return hessian22.evalf(subs ={'x1':x_1,'x2':x_2})


def hessian(x1,x2):
    s=func()
    #res=s.h11(x1,x2)*s.h22(x1,x2)-s.h12(x1,x2)*s.h21(x1,x2)
    mat=np.array([[float(s.h11(x1,x2)),float(s.h12(x1,x2))],
                  [float(s.h21(x1,x2)),float(s.h22(x1,x2))]])
    value,vector=eig(mat)
    return value

s=func()


#x1*
hessian(1,0) #pos definite, local min
s.obj(1,0) #fx=0

#x2*
hessian(-11,1+5**0.5) #pos definite, local min
s.obj(-11,1+5**0.5) 
s.obj(1,0)>s.obj(-11,1+5**0.5)#False, x1* global min 

#x3*
hessian(-11,1-5**0.5) #pos definite, local min
s.obj(-11,1-5**0.5)

#x4*
hessian(-13/3,-2/3) #infinite, saddle point
s.obj(-13/3,-2/3)

#x5*
hessian(1,2) #infinite, saddle point
s.obj(1,2)
