import  math
import  numpy   as np
import  matplotlib.pyplot as plt 
import  seaborn as sns
import matplotlib.ticker as mtick

def _pre(U,phi,sai):
    return phi * max(0, math.tanh(U-sai))
        
class LA():
    
    def __init__(self,cs_w=[],p1=[],p2=[],vip=[],pv=[]):   
        
        #Inputs:    [cs, us]
        #Outputs:   [p1, p2]
        #arrays & parameters: 
        #***    p1, p2:   Pyramidal Neurons
        #***    pv:     Parvalbumin Inhibitpry neuron
        #***    vip:    ventroin--- Inhibitpry neuron
        #***    Pre-synaptic transfer function:[sai, phi]
        #***    Tonic firing(T), tau(time const.)   
        self.Tp =   .3
        self.tauP   =   20.6
        self.saiP   =   28.1
        self.phiP   =   35.5

        self.T_vip =   1.3
        self.T_pv  =   5.3        
        self.tau_vip   =   13.3
        self.tau_pv    =   13
        self.sai_vip   =   27.2
        self.sai_pv    =   31.9
        self.phi_vip   =   44.7
        self.phi_pv    =   144

        #p1 : synaptic weight & LTP
        self.p1 =   p1
        self.us_p1    =   1.52
        self.w0_cs  =   0.35
        self.pv_p1  =   -.33
        self.cs_w   =   cs_w
        self.eta    =   .00012
        self.sigma  =   .85

        #p2 : synaptic weight 
        self.p2 =   p2
        self.cs_p2  =   1. 
        self.pv_p2  =   -.4
        
        #pv : synaptic weight 
        self.pv =   pv
        self.cs_pv  =   .505 
        self.vip_pv =   -1.805
        
        #vip : synaptic weight
        self.vip    =   vip
        self.us_vip =   1.8 
    
    def _pre(self,U,phi,sai):
        return phi * max(0, math.tanh(U-sai))

    def cs_p1(self,cs,p1):
        M = 3 * self.w0_cs
        Post = p1
        Pre = cs
        w   = self.cs_w[-1]
        dw  =   self.eta* (M - self.cs_w[-1])* (Post- (self.phiP* self.sigma))* Pre
        if (Post>=self.sigma):
            w += abs(dw)
        #print("[CS, P1]:",[cs,Post])
        #print("dw:", dw)
        return w

    def LAp1(self, us, cs):
        v   =   self.p1[-1]
        lapv=   self.pv[-1]
        cs  =   self._pre(cs,self.phiP,self.saiP ) 
        us  =   self._pre(us,self.phiP,self.saiP ) 
        lapv  =   self._pre(lapv,self.phiP,self.saiP ) 
        I   =   self.cs_w[-1] * cs + self.us_p1 * us + self.pv_p1 * lapv
        return  v + (-v+ self.Tp + I )/ self.tauP

    def LAp2(self, cs):
        cs  =   self._pre(cs,self.phiP,self.saiP ) 
        v   = self.p2[-1]
        lapv= self.pv[-1] 
        lapv  =   self._pre(lapv,self.phiP,self.saiP ) 
        I   =   self.cs_p2 * cs + self.pv_p2 * lapv 
        return v + (-v+ self.Tp + I)/ self.tauP
    
    def LAvip(self, us):
        v   = self.vip[-1]
        us  = self._pre(us, self.phi_vip, self.phi_vip)
        I   = self.us_vip * us 
        #print(v, I, self.tau_vip)
        return (v + (-v+ self.T_vip + I)/ self.tau_vip)
  
    def LApv(self, cs):
        v   = self.pv[-1]
        lavip   = self.vip[-1]
        cs  =   self._pre(cs,self.phi_pv,self.sai_pv)
        I   =   self.vip_pv * lavip + self.cs_pv * cs
        return v + (-v+ self.T_pv + I)/ self.tau_pv

    def main(self, cs, us):
        vip=self.LAvip(us)
        pv = self.LApv(cs)
        p1 = self.LAp1(us, cs)
        p2 = self.LAp2(cs)
        return vip, pv, p1, p2



dt=1
T=np.arange(0,2700,dt)
CS  =   np.zeros(np.size(T))
US  =   np.zeros(np.size(T))
#Conditioning session
CS[90:110], CS[130:150], CS[170:190] =    100,    100,    100
US[108:110], US[148:150], US[188:190] =    100,   100,    100

#Extinction session 1         
CS[400:420], CS[440:460], CS[480:500]   =   100,    100,    100
CS[520:540], CS[560:580], CS[600:620]   =   100,    100,    100
CS[640:660], CS[680:700], CS[720:740]   =   100,    100,    100
CS[760:780], CS[800:820], CS[840:860]   =   100,    100,    100
CS[880:900], CS[920:940], CS[960:980]   =   100,    100,    100
CS[1000:1020], CS[1040:1060], CS[1080:1100] =   100,    100,    100
CS[1120:1140], CS[1160:1180]    =   100,    100

#Extinction session 2
CS[1400:1420], CS[1440:1460], CS[1480:1500] =    100,    100,    100
CS[1520:1540], CS[1560:1580], CS[1600:1620] =    100,    100,    100
CS[1640:1660], CS[1680:1700], CS[1720:1740] =    100,    100,    100
CS[1760:1780], CS[1800:1820], CS[1840:1860] =    100,    100,    100
CS[1880:1900], CS[1920:1940], CS[1960:1980] =    100,    100,    100
CS[21000:21020], CS[21040:21060], CS[21080:21100] =    100,    100,    100
CS[21120:21140], CS[21160:21180]    =    100,    100

#reinstatement 
CS[2400:2420], CS[2580:2600], US[2510:2511] =    100,    100,    100

a=LA()
a.p1.append(0)
a.p2.append(0)
a.pv.append(0)
a.vip.append(0)
a.cs_w.append(a.w0_cs)

us=np.array([])
cs=np.array([])

for  i, j in enumerate(T):
    print("time is:",j)
    #us = np.append(us,_pre(US[i],a.phiP,a.saiP))
    #cs = np.append(cs,_pre(CS[i],a.phiP,a.saiP))
    lavip, lapv, lap1, lap2 = a.main(CS[i], US[i])
    csw = a.cs_p1(CS[i],lap1)
    
    a.cs_w.append(csw)
    a.p1.append(lap1)
    a.p2.append(lap2)
    a.vip.append(lavip)
    a.pv.append(lapv)

plt.figure()
plt.subplot(3,1,1)
plt.plot(CS)
plt.plot(US)

plt.subplot(3,1,2)
plt.plot(a.p1, label="p1")
plt.plot(a.p2, label="p2")
plt.plot(a.pv, label="pv")
plt.plot(a.vip, label="vip")
plt.legend()
plt.subplot(3,1,3)
plt.plot(a.cs_w)

plt.show()
"""
    def LAp2(self):
    def LAvip(self):
    def LApv(self):

class CEA():
    def __init__(self,):
    def CeLON(self):
    def CeLOFF(self):
    def CeM(self):

class BA():
    def __init__(self,):
    def BAp5(self):
    def BAp1(self):
    def BAp2(self):
    def BAp3(self):
    def BAp4(self):
    def BAcck(self):
    def BApv(self):

class PL():
    def __init__(self,):
    def PLs(self,):
    def PLp(self,):
    def PLpv(self,):

class IL():
    def __init__(self,):
    def ILs(self,):
    def ILp(self,):
    def ILpv(self,):

class ITC():
    def __init__(self,):
    def main(self,):


la=LA() 
ba=BA() 
ce=CEA() 
itc=ITC() 
pl=ITC() 
il=ITC() 

"""