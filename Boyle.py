import numpy as np
from scipy.stats import norm


def Sholes(S0,K,r,sigma,T):
    d1=(np.log(S0/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1 - sigma*np.sqrt(T)
    c=S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    p=K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)
    return c,p

def prix(S0,sigma,lamda,T,N):
    dt=T/N
    u=np.exp(sigma*lamda*np.sqrt(dt))
    d=1/u
    Stock=np.zeros((N+1,2*N+1))
    Stock[0,N]=S0
    for i in range(1,N+1):
        for j in range(N-i,N+i+1):
            if j>N:
                Stock[i,j]=d*Stock[i-1,j-1]
            elif j<N:
                Stock[i,j]=u*Stock[i-1,j+1]
            else:
                Stock[i,j]=S0
    return Stock

def payoff(S0,sigma,lamda,T,N,r,K):
    Stock=prix(S0,sigma,lamda,T,N)
    ST=Stock[N,:]
    dt=T/N
    u=np.exp(sigma*lamda*np.sqrt(dt))
    d=1/u
    CT=np.maximum(ST-K,0)
    PT=np.maximum(K-ST,0)
    call=np.zeros((N+1,2*N+1))
    put=np.zeros((N+1,2*N+1))
    call[N,:]=CT
    put[N,:]=PT
    M = np.exp(r*dt)
    V = M**2*(np.exp(sigma**2*dt)-1)

    pu = (u*(V + M*(M-1)) - (M-1))/((u-1)**2*(u+1))
    pd = (u**2*(V + M*(M-1)) - u**3*(M-1))/((u-1)**2*(u+1))
    pm = 1 - pu - pd
    for i in range(N-1,-1,-1):
        for j in range(N-i,N+i+1):
            call[i,j]=np.exp(-r*dt)*(pu*call[i+1,j-1] + pm*call[i+1,j] + pd*call[i+1,j+1])
            put[i,j]=np.exp(-r*dt)*(pu*put[i+1,j-1] + pm*put[i+1,j] + pd*put[i+1,j+1])
    return call[0,N],put[0,N]

def payoff_A(S0,sigma,lamda,T,N,r,K):
    Stock=prix(S0,sigma,lamda,T,N)
    ST=Stock[N,:]
    dt=T/N
    u=np.exp(sigma*lamda*np.sqrt(dt))
    d=1/u
    CT=np.maximum(ST-K,0)
    PT=np.maximum(K-ST,0)
    call=np.zeros((N+1,2*N+1))
    put=np.zeros((N+1,2*N+1))
    call[N,:]=CT
    put[N,:]=PT
    M = np.exp(r*dt)
    V = M**2*(np.exp(sigma**2*dt)-1)
    pay_ant=np.maximum(Stock-K,0)
    pay_ant2=np.maximum(K-Stock,0)

    pu = (u*(V + M*(M-1)) - (M-1))/((u-1)**2*(u+1))
    pd = (u**2*(V + M*(M-1)) - u**3*(M-1))/((u-1)**2*(u+1))
    pm = 1 - pu - pd
    for i in range(N-1,-1,-1):
        for j in range(N-i,N+i+1):
            call[i,j]=np.maximum(pay_ant[i,j],np.exp(-r*dt)*(pu*call[i+1,j-1] + pm*call[i+1,j] + pd*call[i+1,j+1]))
            put[i,j]=np.maximum(pay_ant2[i,j],np.exp(-r*dt)*(pu*put[i+1,j-1] + pm*put[i+1,j] + pd*put[i+1,j+1]))

    return call[0,N],put[0,N]
          






S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1
lamda = np.sqrt(3/2)
N=50

c,p=payoff(S0,sigma,lamda,T,N,r,K)
c1,p1=payoff_A(S0,sigma,lamda,T,N,r,K)
print('call eu',c)
print('call Am',c1)
print('put eu',p)
print('put Am',p1)