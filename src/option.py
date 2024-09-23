import numpy as np
import scipy.stats
from datetime import datetime as dt
N = scipy.stats.norm.cdf
def n(t):
    n = np.exp(-(t**2)/2)/np.sqrt(2*np.pi)
    return n

class Option:
    def __init__(self, option_type, initial_price, strike_price, interest_rate, volatility, dividend_rate, time, method, start_date=None, end_date=None) -> None:
        self.option_type = option_type
        self.S0 = initial_price
        self.K = strike_price
        self.r = interest_rate
        self.sig = volatility
        self.y = dividend_rate
        self.T = time
        self.phi = 1
        if option_type == 'put':
            self.phi = -1

        self.method_name = method
        self.method = getattr(self, method)
        # self.start_date = start_date
        # self.end_date = end_date

    def setSeedVariables(self, seed, n, dt=None):
        self.n = n
        self.dt = dt
        self.seed = seed
    
    def setTreeSteps(self, n):
        self.n = n

    def setFDVariables(self, S_min, S_max, dt):
        self.S_min = S_min
        self.S_max = S_max
        self.dt = dt

    def priceOption(self, eps=1):
        if self.method_name == 'BSM':
            return self.method()
        
        time_eps = 0.05
        sig_eps = eps / 200
        S = []

        if self.method_name != 'FD':
            V = self.method()
            self.S0 += eps
            Vp = self.method()
            self.S0 -= 2 * eps
            Vm = self.method()
            self.S0 += eps
        else:
            V, S = self.method(return_S = True)
            self.S_min += eps
            self.S_max += eps
            Vp = self.method()
            self.S_min -= 2 * eps
            self.S_max -= 2 * eps
            Vm = self.method()
            self.S_max += eps
            self.S_min += eps

        delta = (Vp - Vm) / 2 / eps
        gamma = (Vp + Vm - 2 * V) / (eps**2)

        self.T += time_eps
        VTp = self.method()
        self.T -= time_eps
        theta = (V - VTp) / time_eps

        self.sig += sig_eps
        VSigp = self.method()
        self.sig -= sig_eps
        vega = (VSigp - V) / sig_eps

        self.r += sig_eps
        Vrp = self.method()
        self.r -= sig_eps
        rho = (Vrp - V) / sig_eps

        if len(S) > 0:
            return {"stock_price": S, "price": np.round(V, 3), "delta": np.round(delta, 3), "gamma": np.round(gamma, 3), "theta": np.round(theta, 3), "vega": np.round(vega, 3), "rho": np.round(rho, 3)}

        return {"price": np.round(V, 3), "delta": np.round(delta, 3), "gamma": np.round(gamma, 3), "theta": np.round(theta, 3), "vega": np.round(vega, 3), "rho": np.round(rho, 3)}

class European_Option(Option):
    def BSM(self):
        zp = (np.log(self.S0 / self.K) + (self.r - self.y) * self.T) / (self.sig * np.sqrt(self.T)) + self.sig * np.sqrt(self.T) / 2
        zm = (np.log(self.S0 / self.K) + (self.r - self.y) * self.T) / (self.sig * np.sqrt(self.T)) - self.sig * np.sqrt(self.T) / 2

        V = self.phi * (self.S0 * np.exp(- self.y * self.T) * N(self.phi * zp) - self.K * np.exp(- self.r * self.T) * N(self.phi * zm))

        delta = self.phi * np.exp(- self.y * self.T) * N(self.phi * zp)
        rho = self.phi * self.K * self.T * np.exp(- self.r * self.T) * N(self.phi * zm)
        gamma = np.exp(- self.y * self.T) * n(zp) / self.sig / np.sqrt(self.T) / self.S0
        deltaK = - self.phi * np.exp(- self.r * self.T) * N(self.phi * zm)
        theta = self.r * self.K * deltaK + self.y * self.S0 * delta - self.sig**2 * self.S0**2 * gamma / 2
        vega = self.S0 * np.sqrt(self.T) * np.exp(- self.r * self.T) * n(zm)
        return {"price": round(V, 3), "delta": round(delta, 3), "gamma": round(gamma, 3), "theta": round(theta, 3), "vega": round(vega, 3), "rho": round(rho, 3)}
    
    def MC(self):
        m = int(self.T / self.dt)
        np.random.seed(self.seed)
        z = np.random.normal(size = (self.n, m))
        mu = self.r - (self.sig**2) / 2

        ST1 = np.zeros((self.n, m+1))
        ST2 = np.zeros((self.n, m+1))

        ST1[:,0] += self.S0
        ST2[:,0] += self.S0

        for i in range(m):
            ST1[:,i+1] = ST1[:,i] * np.exp(mu * self.dt + self.sig * np.sqrt(self.dt) * z[:,i])
            ST2[:,i+1] = ST2[:,i] * np.exp(mu * self.dt - self.sig * np.sqrt(self.dt) * z[:,i])

        if self.option_type == 'call':
            VTa = (np.where(ST1[:,-1] < self.K, 0, ST1[:,-1] - self.K) + np.where(ST2[:,-1] < self.K, 0, ST2[:,-1] - self.K)) / 2
        else:
            VTa = (np.where(ST1[:,-1] > self.K, 0, self.K - ST1[:,-1]) + np.where(ST2[:,-1] > self.K, 0, self.K - ST2[:,-1])) / 2

        Vta_est = np.exp(-self.r * self.T) * VTa.mean()
        return Vta_est
    
    def BT(self):
        dt = self.T / self.n

        u = np.exp(self.sig * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((self.r - self.y) * dt) - d) / (u - d)

        ST = np.array([self.S0 * u**(self.n - i) * d**i for i in range(self.n + 1)])
        
        V = np.maximum(self.phi * (ST - self.K), 0)

        for i in range(self.n - 1, -1, -1):
            Vt = np.zeros(i+1)
            for j in range(i+1):
                Vt[j] = np.exp(-self.r * dt) * (p * V[j] + (1-p) * V[j+1])
            V = Vt
        return V[0]
    
    def TT(self):
        dt = self.T / self.n
        u = np.exp(self.sig * np.sqrt(3 * dt))
        d = 1 / u
        dXu = self.sig * np.sqrt(3*dt)

        gam = self.r - self.y - self.sig**2 / 2
        pd = 0.5 * ((self.sig**2 * dt + gam**2 * dt**2) / dXu**2 - gam * dt / dXu)
        pu = 0.5 * ((self.sig**2 * dt + gam**2 * dt**2) / dXu**2 + gam * dt / dXu)
        pm = 1 - pd - pu

        ST = np.array([self.S0 * u**max(self.n - i, 0) * d**max(i - self.n, 0) for i in range(2 * self.n + 1)])
        V = np.maximum(self.phi * (ST - self.K), 0)

        for i in range(self.n - 1, -1, -1):
            Vt = np.zeros(2 * i + 1)
            for j in range(2 * i + 1):
                Vt[j] = np.exp(-self.r * dt) * (pu * V[j] + pm * V[j+1] + pd * V[j+2])
            V = Vt
        return V[0]
    
    def FD(self, return_S = False):
        M = int(self.T / self.dt)
        dS = 1
        alpha = 0.5

        N = int(np.round((self.S_max - self.S_min) / dS) + 1)
        S = np.linspace(self.S_max, self.S_min, N)
        j = S / dS

        V = np.zeros((N, M))
        V[:, -1] = np.maximum(self.phi * (S - self.K), 0)

        a1 = (self.sig**2 * j**2 + (self.r - self.y) * j) * (1 - alpha) * self.dt / 2
        a2 = - 1 - (self.sig**2 * j**2 + self.r) * (1 - alpha) * self.dt
        a3 = (self.sig**2 * j**2 - (self.r - self.y) * j) * (1 - alpha) * self.dt / 2

        b1 = - ((self.r - self.y) * j + self.sig**2 * j**2) * alpha * self.dt / 2
        b2 = (self.sig**2 * j**2 + self.r) * alpha * self.dt - 1
        b3 = ((self.r - self.y) * j - self.sig**2 * j**2) * alpha * self.dt / 2

        RA = np.zeros((N,N))
        LA = np.zeros((N,N))

        LA[0, 0] = self.phi
        LA[0, 1] = -self.phi
        LA[-1, -1] = -self.phi
        LA[-1, -2] = self.phi

        for i in range(1, N - 1):
            LA[i, i - 1] = a1[i]
            LA[i, i] = a2[i]
            LA[i, i + 1] = a3[i]

            RA[i, i - 1] = b1[i]
            RA[i, i] = b2[i]
            RA[i, i + 1] = b3[i]

        B = np.zeros(N)
        if self.phi == 1:
            B[0] = S[0] - S[1]
        else:
            B[-1] = S[-2] - S[-1]

        for i in range(M-2,-1,-1):
            V[:, i] = np.linalg.solve(LA, np.dot(RA, V[:, i + 1]) + B)
        
        if return_S:
            return V[:, 0], S
        return V[:, 0]
        
class American_Option(Option):
    def BT(self):
        dt = self.T / self.n

        u = np.exp(self.sig * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((self.r - self.y) * dt) - d) / (u - d)

        ST = np.array([self.S0 * u**(self.n - i) * d**i for i in range(self.n + 1)])
        
        V = np.maximum(self.phi * (ST - self.K), 0)

        for i in range(self.n - 1, -1, -1):
            Vt = np.zeros(i+1)
            St = np.array([self.S0 * u**(i - k) * d**k for k in range(i+1)])
            ev = np.maximum(self.phi * (St - self.K),0)
            for j in range(i+1):
                Vt[j] = max(ev[j], np.exp(- self.r * dt) * (p * V[j] + (1-p) * V[j+1]))
            V = Vt
        return V[0]
    
    def TT(self):
        dt = self.T / self.n
        u = np.exp(self.sig * np.sqrt(3 * dt))
        d = 1 / u
        dXu = self.sig * np.sqrt(3 * dt)

        gam = self.r - self.y - self.sig**2 / 2
        pd = 0.5 * ((self.sig**2 * dt + gam**2 * dt**2) / dXu**2 - gam * dt / dXu)
        pu = 0.5 * ((self.sig**2 * dt + gam**2 * dt**2) / dXu**2 + gam * dt / dXu)
        pm = 1 - pd - pu

        ST = np.array([self.S0 * u**max(self.n - i, 0) * d**max(i - self.n, 0) for i in range(2 * self.n + 1)])
        V = np.maximum(self.phi * (ST - self.K), 0)

        for i in range(self.n - 1, -1, -1):
            Vt = np.zeros(2 * i + 1)
            St = np.array([self.S0 * u**max(i - k, 0) * d**max(k - i, 0) for k in range(2 * i + 1)])
            ev = np.maximum(self.phi * (St - self.K), 0)
            for j in range(2 * i + 1):
                Vt[j] = max(ev[j], np.exp(-self.r * dt) * (pu * V[j] + pm * V[j+1] + pd * V[j+2]))
            V = Vt
        return V[0]
    
    def LSMC(self):
        m = int(np.sqrt(self.n))
        dt = self.T / m
        mu = self.r - self.y - self.sig**2 / 2
        
        np.random.seed(self.seed)
        Z = np.random.normal(size=(self.n // 2, m))

        St = np.zeros((self.n, m + 1))
        St[:,0] = self.S0

        Index = np.zeros((self.n, m))

        for i in range(m):
            St[:self.n // 2, i + 1] = St[:self.n // 2, i] * np.exp(mu * dt + self.sig * np.sqrt(dt) * Z[:,i])
            St[self.n // 2:, i + 1] = St[self.n // 2:, i] * np.exp(mu * dt - self.sig * np.sqrt(dt) * Z[:,i])
        St = St[:, 1:]
        St = St / self.K
        
        EV = np.zeros((self.n, m))
        EV[:, -1] = np.maximum(self.phi * (St[:, -1] - 1), 0)
        Index[:, -1] = np.where(EV[:, -1] > 0, 1, 0)

        for i in range(m - 2, -1, -1):
            EV[:, i] = np.maximum(self.phi * (St[:, i] - 1), 0)

            ITM = np.where(EV[:, i] > 0)[0]
            Y = (Index[ITM, i + 1:] * EV[ITM, i + 1:] * np.exp(-self.r * np.arange(1, m - i) * dt)).sum(axis=1)

            # f = poly(St[ITM,i],k)
            f = np.array([St[ITM, i]**j for j in range(4)])
            A = np.dot(f, f.T)
            b = np.dot(f, Y)
            a = np.linalg.solve(A, b)
            ECV = np.dot(f.T, a)

            Index[ITM[EV[ITM, i] >= ECV], i] = 1
            Index[ITM[EV[ITM, i] >= ECV], i + 1:] = 0

        V0 = (Index * EV * np.exp(-self.r * np.arange(1, m + 1) * dt)).sum(axis=1)
        return self.K * V0.mean()
    
    def FD(self, return_S = False):
        M = int(self.T / self.dt)
        dS = 1
        alpha = 0.5

        N = int(np.round((self.S_max - self.S_min) / dS) + 1)
        S = np.linspace(self.S_max, self.S_min, N)
        j = S / dS

        CV = np.zeros((N, M))
        CV[:, -1] = np.maximum(self.phi * (S - self.K), 0)

        a1 = (self.sig**2 * j**2 + (self.r - self.y) * j) * (1 - alpha) * self.dt / 2
        a2 = - 1 - (self.sig**2 * j**2 + self.r) * (1 - alpha) * self.dt
        a3 = (self.sig**2 * j**2 - (self.r - self.y) * j) * (1 - alpha) * self.dt / 2

        b1 = - ((self.r - self.y) * j + self.sig**2 * j**2) * alpha * self.dt / 2
        b2 = (self.sig**2 * j**2 + self.r) * alpha * self.dt - 1
        b3 = ((self.r - self.y) * j - self.sig**2 * j**2) * alpha * self.dt / 2

        RA = np.zeros((N,N))
        LA = np.zeros((N,N))

        LA[0, 0] = self.phi
        LA[0, 1] = -self.phi
        LA[-1, -1] = -self.phi
        LA[-1, -2] = self.phi

        for i in range(1, N - 1):
            LA[i, i - 1] = a1[i]
            LA[i, i] = a2[i]
            LA[i, i + 1] = a3[i]

            RA[i, i - 1] = b1[i]
            RA[i, i] = b2[i]
            RA[i, i + 1] = b3[i]

        B = np.zeros(N)
        if self.phi == 1:
            B[0] = S[0] - S[1]
        else:
            B[-1] = S[-2] - S[-1]

        for i in range(M-2,-1,-1):
            CV[:, i] = np.linalg.solve(LA, np.dot(RA, CV[:, i + 1]) + B)
            CV[:, i] = np.maximum(CV[:, i], CV[:, i + 1])
        
        if return_S:
            return CV[:, 0], S
        return CV[:, 0]