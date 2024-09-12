import numpy as np
import scipy.stats
from datetime import datetime as dt
N = scipy.stats.norm.cdf
def n(t):
    n = np.exp(-(t**2)/2)/np.sqrt(2*np.pi)
    return n

class Option:
    def __init__(self, option_type, initial_price, strike_price, interest_rate, volatility, dividend_rate, time, start_date=None, end_date=None) -> None:
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
        # self.start_date = start_date
        # self.end_date = end_date

    def setSeedVariables(self, n, dt, seed):
        self.n = n
        self.dt = dt
        self.seed = seed
    
    def setTreeSteps(self, n):
        self.n = n

    def greeks(self, eps=0.5, method='BT'):
        time_eps = 0.05
        sig_eps = eps / 100

        O = self.priceOption(method=method)
        self.S0 += eps
        Op = self.priceOption(method=method)
        self.S0 -= 2 * eps
        Om = self.priceOption(method=method)
        self.S0 += eps

        delta = (Op - Om) / 2 / eps
        gamma = (Op + Om - 2 * O) / (eps**2)

        self.T += time_eps
        OTp = self.priceOption(method=method)
        self.T -= time_eps
        theta = (O - OTp) / time_eps

        self.sig += sig_eps
        OSigp = self.priceOption(method=method)
        self.sig -= sig_eps
        vega = (OSigp - O) / sig_eps

        self.r += sig_eps
        Orp = self.priceOption(method=method)
        self.r -= sig_eps
        rho = (Orp - O) / sig_eps 

        return {"price": O, "delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}

    def priceOption(self, greeks=False, method='BSM'):
        if method=='BSM':
            return self.BSM(greeks=greeks)
        if greeks:
            return self.greeks(method=method)
        if method == 'MC':
            return self.MC()
        if method == 'BT':
            return self.BT()
        if method == 'TT':
            return self.TT()

class European_Option(Option):
    def BSM(self, greeks=False):

        zp = (np.log(self.S0 / self.K) + (self.r - self.y) * self.T) / (self.sig * np.sqrt(self.T)) + self.sig * np.sqrt(self.T) / 2
        zm = (np.log(self.S0 / self.K) + (self.r - self.y) * self.T) / (self.sig * np.sqrt(self.T)) - self.sig * np.sqrt(self.T) / 2

        O = self.phi * (self.S0 * np.exp(- self.y * self.T) * N(self.phi * zp) - self.K * np.exp(- self.r * self.T) * N(self.phi * zm))
        if greeks:
            delta = self.phi * np.exp(- self.y * self.T) * N(self.phi * zp)
            rho = self.phi * self.K * self.T * np.exp(- self.r * self.T) * N(self.phi * zm)
            gamma = np.exp(- self.y * self.T) * n(zp) / self.sig / np.sqrt(self.T) / self.S0
            deltaK = - self.phi * np.exp(- self.r * self.T) * N(self.phi * zm)
            theta = self.r * self.K * deltaK + self.y * self.S0 * delta - self.sig**2 * self.S0**2 * gamma / 2
            vega = self.S0 * np.sqrt(self.T) * np.exp(- self.r * self.T) * n(zm)
            return {"price": O, "delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}
        return O
    
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
            OTa = (np.where(ST1[:,-1] < self.K, 0, ST1[:,-1] - self.K) + np.where(ST2[:,-1] < self.K, 0, ST2[:,-1] - self.K)) / 2
        else:
            OTa = (np.where(ST1[:,-1] > self.K, 0, self.K - ST1[:,-1]) + np.where(ST2[:,-1] > self.K, 0, self.K - ST2[:,-1])) / 2

        Ota_est = np.exp(-self.r * self.T) * OTa.mean()
        return Ota_est
    
    def BT(self):
        dt = self.T / self.n

        u = np.exp(self.sig * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((self.r - self.y) * dt) - d) / (u - d)

        ST = np.array([self.S0 * u**(self.n - i) * d**i for i in range(self.n + 1)])
        
        O = np.maximum(self.phi * (ST - self.K), 0)

        for i in range(self.n - 1, -1, -1):
            Ot = np.zeros(i+1)
            for j in range(i+1):
                Ot[j] = np.exp(-self.r * dt) * (p * O[j] + (1-p) * O[j+1])
            O = Ot
        return O[0]
    
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
        O = np.maximum(self.phi * (ST - self.K), 0)

        for i in range(self.n - 1, -1, -1):
            Ot = np.zeros(2 * i + 1)
            for j in range(2 * i + 1):
                Ot[j] = np.exp(-self.r * dt) * (pu * O[j] + pm * O[j+1] + pd * O[j+2])
            O = Ot
        return O[0]
    
class American_Option(Option):
    def BT(self):
        dt = self.T / self.n

        u = np.exp(self.sig * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((self.r - self.y) * dt) - d) / (u - d)

        ST = np.array([self.S0 * u**(self.n - i) * d**i for i in range(self.n + 1)])
        
        O = np.maximum(self.phi * (ST - self.K), 0)

        for i in range(self.n - 1, -1, -1):
            Ot = np.zeros(i+1)
            St = np.array([self.S0 * u**(i - k) * d**k for k in range(i+1)])
            ev = np.maximum(self.phi * (St - self.K),0)
            for j in range(i+1):
                Ot[j] = max(ev[j], np.exp(- self.r * dt) * (p * O[j] + (1-p) * O[j+1]))
            O = Ot
        return O[0]
    
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
        O = np.maximum(self.phi * (ST - self.K), 0)

        for i in range(self.n - 1, -1, -1):
            Ot = np.zeros(2 * i + 1)
            St = np.array([self.S0 * u**max(i - k, 0) * d**max(k - i, 0) for k in range(2 * i + 1)])
            ev = np.maximum(self.phi * (St - self.K), 0)
            for j in range(2 * i + 1):
                Ot[j] = max(ev[j], np.exp(-self.r * dt) * (pu * O[j] + pm * O[j+1] + pd * O[j+2]))
            O = Ot
        return O[0]