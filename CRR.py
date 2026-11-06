import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def Sholes(S0,K,r,sigma,T):
    d1=(np.log(S0/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1 - sigma*np.sqrt(T)
    c=S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    p=K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)
    return c,p


def Prix(S0,sigma,T,N):
    dt=T/N
    u=np.exp(sigma*np.sqrt(dt))
    d=1/u
    Stock=np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(i+1):
            Stock[i,j]=S0 * u**(i-j) * d**j
    return Stock

def Payoff(S0,r,sigma,T,N,K):
    dt=T/N
    u=np.exp(sigma*np.sqrt(dt))
    d=1/u
    p=(np.exp(r*dt)-d)/(u-d)
    Stock=Prix(S0,sigma,T,N)
    ST=Stock[N,:]
    CT=np.maximum(ST-K,0)
    PT=np.maximum(K-ST,0)
    call=np.zeros((N+1,N+1))
    put=np.zeros((N+1,N+1))
    call[N,:]=CT
    put[N,:]=PT
    for i in range(N-1,-1,-1):
        for j in range(i+1):
            call[i,j]=np.exp(-r*dt) * (p*call[i+1,j] + (1-p)*call[i+1,j+1])
            put[i,j]=np.exp(-r*dt)*(p*put[i+1,j] + (1-p)*put[i+1,j+1])
    return call[0,0],put[0,0]

def Payoff_A(S0,r,sigma,T,N,K):
    dt=T/N
    u=np.exp(sigma*np.sqrt(dt))
    d=1/u
    p=(np.exp(r*dt)-d)/(u-d)
    Stock=Prix(S0,sigma,T,N)
    ST=Stock[N,:]
    CT=np.maximum(ST-K,0)
    PT=np.maximum(K-ST,0)
    call=np.zeros((N+1,N+1))
    put=np.zeros((N+1,N+1))
    put[N,:]=PT
    
    call[N,:]=CT
    pay_antcall=np.maximum(Stock-K,0)
    pay_antput=np.maximum(K-Stock,0)
    for i in range(N-1,-1,-1):
        for j in range(i+1):
            call[i,j]=max(pay_antcall[i,j],np.exp(-r*dt)*(p*call[i+1,j]+(1-p)*call[i+1,j+1]))
            put[i,j]=max(pay_antput[i,j],np.exp(-r*dt)*(p*put[i+1,j] +(1-p)*put[i+1,j+1]))

    return call[0,0],put[0,0]

def Richardson(S0, r, sigma, T, N, K):
    call_N, put_N = Payoff(S0, r, sigma, T, N, K)
    call_2N, put_2N = Payoff(S0, r, sigma, T, 2*N, K)

    call_R = 2*call_2N - call_N
    put_R = 2*put_2N - put_N

    return call_R, put_R
            


S0 = 100
r = 0.05
sigma = 0.2
T = 1
K = 100

call_BS, put_BS = Sholes(S0,K,r,sigma,T)

print("Call Black-Scholes :", call_BS)
print("Put Black-Scholes  :", put_BS)

print()
print(" N   Erreur Call CRR   Erreur Call R   Erreur Put CRR   Erreur Put R")

for N in [10, 20, 50, 100, 200, 500]:
    call_N, put_N = Payoff(S0, r, sigma, T, N, K)
    call_R, put_R = Richardson(S0, r, sigma, T, N, K)

    erreur_call = abs(call_N - call_BS)
    erreur_call_R = abs(call_R - call_BS)

    erreur_put = abs(put_N - put_BS)
    erreur_put_R = abs(put_R - put_BS)
    print(f"{N:3d}  {erreur_call:15.8f}  {erreur_call_R:13.8f}"
          f"  {erreur_put:15.8f}  {erreur_put_R:13.8f}")