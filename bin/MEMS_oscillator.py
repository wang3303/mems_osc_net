from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import cmath
from mpl_toolkits.mplot3d import Axes3D

class oscillator_solver():
    #Initiate solver
    def __init__(self,n,c,theta0,s,w,mode = "phase",show = False):
        self.n = n #number of MEMS oscilltors has to be larger than 1
        self.c = c # c_i
        self.s = s # s_ij
        self.w = w
        self.t = np.linspace(0,20, 501)
        self.theta0 = theta0
        self.color = ['y','m','c','r','g','b','w','k']
        self.mode = mode #Plot either magnitude or phase of oscillation
        self.res = 0
        self.sol = -1
        self.show = show
        self.string = ""
        if self.show:
            for i in range(self.n):
                self.res+=(self.w[i]-1)**2
                self.string += "Initial Frequency of Oscillator {}:{:.2f}rad/s\n".format(i+1,self.w[i])
                self.string += "Initial Phase of Oscillator {}:{:.2f}\n".format(i+1,self.theta0[i])

    def print_coupling_strength():
        print self.s
    
    # This function is for odeint
    def func(self,theta,t,s):
        dydt = np.zeros(self.n)
        for i in range(self.n):
            dydt[i]= self.w[i]+self.sin_sum(theta,i)            
        return dydt
    
    def plot(self,ax,t,sol,color = 'r',text ='11',):
        if self.mode == "phase":
            ax.plot(self.t,phase_convert(sol), color, label=text)
        else:
            ax.plot(self.t,np.sin(sol), color, label=text)
        ax.legend(loc='best')
        ax.set_xlabel('t')
        ax.grid()
        
    
    def plot_solution(self,sol,):
        if self.show:
            fig1 = plt.figure('Oscillation state')
            plt.clf()
            ax1  = fig1.add_subplot(111)
            for i in range(self.n):
                self.plot(ax1,self.t,sol[:,i],color = self.color[i % 8],text = 'Oscillator'+str(i+1))
            plt.show()
            plt.figure('Degree of Match')
        result = np.exp(1j*sol)
        result = np.abs(np.sum(result,axis = 1))
        result = result/self.n
        if self.show:
            plt.plot(self.t,result, 'b', label='DoM')
            plt.title('Degree of Match')
            plt.show()
        self.sol = np.average(result[400:500])
            
        
    def w_ij(self,i,j):
        return np.abs(self.s[i,j])*np.sqrt(self.c[j]/self.c[i])
    
    def sin_sum(self,theta,i):
        w_i = [self.w_ij(i,j) for j in range(self.n)]
        res = theta - theta[i]
        for j in range(self.n):
            res[j] +=cmath.phase(self.s[i,j])
        res = np.sin(res)
        res = res*w_i
        res = np.sum(res)
        return res
    
    def showsolution(self,):
        self.res = 0
        for i in range(self.n):
            self.res+=(self.theta0[i])**2
        sol = odeint(self.func, self.theta0, self.t, args=(self.s,))
        self.plot_solution(sol)

def phase_convert(rad):
    a = np.rint(rad / np.pi / 2)
    return rad-a*np.pi*2

def symmetrize(a):
        return a + a.T - np.diag(a.diagonal())

def simulate_network(number = 3): 
    n = number #number of MEMS oscilltors has to be larger than 1
    c = np.ones(n)# c_i
    # Remove the # below to add conductance coupling component between oscillators
    s = np.random.randn(n,n)#+1j*np.random.randn(n,n)# s_ij
    s = symmetrize(s)
    np.fill_diagonal(s, 0)
    print "Coupling strength"
    np.set_printoptions(precision=2)
    
    w = np.abs(np.random.randn(n))
    print s
    mode = ""
    theta0 = np.random.rand(n)
    solver = oscillator_solver(n,c,theta0,s,w,mode,show = True)
    #solver.mode = "phase"
    solver.showsolution()
    return solver.string

def DOM_under_different_initial_conditions_3(shift=0):
    kkk = 100
    x = np.zeros(kkk**2)
    y = np.zeros(kkk**2)
    data = np.zeros(kkk**2)
    count= 0
    n = 3 #number of MEMS oscilltors has to be larger than 1
    c = np.ones(n)# c_i
    s = np.zeros((n,n))
    np.fill_diagonal(s, 0)
    mode = ""
    theta0 = np.zeros(n)
    init = "phase"
    solver = oscillator_solver(n,c,theta0,s,[1,1,1],mode)
    for i in range(kkk):
        if i % 20 == 0:
            print "Finish {}%".format(i)
        for j in range(kkk):
            solver.w = np.array([i*0.001-0.05,j*0.001-0.05,shift])
            solver.showsolution()
            x[count] = solver.w[0]
            y[count] = solver.w[1]
            data[count] = solver.sol
            count +=1

    from matplotlib import cm
    fig = plt.figure('3-Oscillator')
    ax = fig.add_subplot(111, projection='3d')
    x =x.reshape((kkk,kkk))
    y =y.reshape((kkk,kkk))
    data =data.reshape((kkk,kkk))
    surf = ax.plot_surface(x,y,data,cmap=cm.jet)
    ax.set_xlabel('Input component x1')
    ax.set_ylabel('Input component x2')
    ax.set_zlabel('Output value')
    fig.colorbar(surf)

def DOM_under_different_initial_conditions_2():
    kkk = 100
    x = np.zeros(kkk**2)
    y = np.zeros(kkk**2)
    data = np.zeros(kkk**2)
    count= 0
    n = 2 #number of MEMS oscilltors has to be larger than 1
    c = np.ones(n)# c_i
    s = np.zeros((n,n))
    np.fill_diagonal(s, 0)
    theta0 = np.zeros(n)
    mode = ""
    solver = oscillator_solver(n,c,theta0,s,[1,1,1],mode)
    #solver.mode = "phase"
    for i in range(kkk):
        if i % 20 == 0:
            print "Finish {}%".format(i)
        for j in range(kkk):
            solver.w = np.array([i*0.001-0.05,j*0.001-0.05])
            solver.showsolution()
            x[count] = solver.w[0]
            y[count] = solver.w[1]
            data[count] = solver.sol
            count +=1

    from matplotlib import cm
    fig = plt.figure('2-Oscillator')
    ax = fig.add_subplot(111, projection='3d')
    x =x.reshape((kkk,kkk))
    y =y.reshape((kkk,kkk))
    data =data.reshape((kkk,kkk))
    surf = ax.plot_surface(x,y,data,cmap=cm.jet)
    ax.set_xlabel('Input component x1')
    ax.set_ylabel('Input component x2')
    ax.set_zlabel('Output value')
    fig.colorbar(surf)

def pixel_convolution(a,b):
    if a > 0.05 or a < -0.05:
        raise ValueError("Data should be in the range (-0.05,0.05)")  
    if b > 0.05 or b < -0.05:
        raise ValueError("Data should be in the range (-0.05,0.05)")  
    # Initiate solver
    n = 2 #number of MEMS oscilltors has to be larger than 1
    c = np.ones(n)# c_i
    s = np.zeros((n,n))
    np.fill_diagonal(s, 0)
    theta0 = np.zeros(n)
    mode = ""
    solver = oscillator_solver(n,c,theta0,s,[1,1,1],mode)
    
    solver.w = np.array([a,b])
    solver.showsolution()
    sol_ab = solver.sol
    
    solver.w = np.array([a,0])
    solver.showsolution()
    sol_a0 = solver.sol
    
    solver.w = np.array([0,b])
    solver.showsolution()
    sol_0b = solver.sol
    
    return (sol_ab-sol_a0-sol_0b+0.993018255)/166.323808


