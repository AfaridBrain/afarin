import  math
import  numpy   as np
from    scipy   import  signal
import  scipy   as sp
import  matplotlib.pyplot as plt 
import  seaborn as sns
import matplotlib.ticker as mtick
import re

class lap1():

    def __init__(self,CS,US,LApv,CS_w,STEP):
        "transfer function parameters"
        self.T  =   .3
        self.tau=   20.6
        self.sai=   28.1
        self.phi=   50
        self.dt =   STEP

        "synaptic weight & LTP"
        self.us_    =    1.52
        "self.cs_    =    0.35"
        self.cs_    =    0.4
        self.lapv_  =    -.33
        "self.lapv_  =    -.35"
        self.cs_w   =   CS_w
        self.eta    =   .00012
        self.sigma  =   .85

        "input values"
        self.us =    US
        self.cs =    CS
        self.lapv=  LApv        

    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v):
        w=0
        dw  =   self.eta * (3*self.cs_ - self.cs_w) * (v - (self.phi*self.sigma)) * (self.cs)         
        
        if (v>= self.sigma ):
            w   =  self.cs_w + abs(dw) * self.dt 
        else: 
            w   =  self.cs_w                            
        
        I   =   w * self._pre(self.cs) + self.us_ * self._pre(self.us) + self.lapv_ * self._pre(self.lapv) 
        return (-v+self.T+I)/self.tau, w                
class lap2():


    def __init__(self,CS,LApv):
    
        "transfer function parameters"
        self.T  =   .3
        self.tau=   20.6
        "self.tau=   20.1"
        self.sai=   28.1
        self.phi=   50

        "synaptic weight & LTP"
        "self.cs_    =    1."
        self.cs_    =    1. + .1
        self.lapv_  =    -.3


        "input values"
        self.cs =    CS
        self.lapv=  LApv        

    def _pre(self,U):

        return self.phi * max(0, math.tanh(U-self.sai))

    
    def unit(self,v):
        I   =   self.cs_* self._pre(self.cs) + self.lapv_*self._pre(self.lapv) 
        return (-v+self.T+I)/self.tau
class lavip():

    def __init__(self,US):
    
        "transfer function parameters"
        self.T  =   1.3
        self.tau=   13.3 
        self.sai=   27.2
        self.phi=   44.7

        "synaptic weight & LTP"
        "self.us_    =    1.8"
        self.us_    =    1.9
        "self.us_    =    8.08"

        "input values"
        self.us =    US     

    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))
            
    def unit(self,v):
        I   =   self.us_*  self._pre(self.us)
        return (-v+self.T+I)/self.tau
class lapv():

    def __init__(self,CS,LAvip):
    
        "transfer function parameters"
        self.T  =   5.3
        self.tau=   13.0 
        self.sai=   31.9
        self.phi=   144

        "synaptic weight & LTP"
        "self.cs_    =     .505"
        self.cs_    =     .505 
        "self.cs_    =     1"
        self.lavip_ =     -1.805 - .1

        "input values"
        self.cs =   CS     
        self.lavip  =   LAvip     

    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v):
        I   =   self.cs_* self._pre(self.cs) + self.lavip_*self._pre(self.lavip)
        return (-v+self.T+I)/self.tau

class cel_on():

    def __init__(self,LAp1,LAp2,LAp2_w,step):
    
        "transfer function parameters"
        self.T  =   1.3
        self.tau=   13.3
        self.sai=   27.2
        self.phi=   50

        "synaptic weight & LTP"
        self.lap1_  =   2
        self.lap2_  =   2
        "self.lap1_  =   1.12"
        "self.lap2_  =    1.12"
        "self.lap2_  =   1.01"
        self.lap2_w =   LAp2_w
        self.eta    =   .0002
        self.sigma  =   .85
        self.dt  =   step

        "input values"
        self.lap1   =   LAp1     
        self.lap2   =   LAp2     

    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v):
    
        dw  =   self.eta * (3*self.lap2_ - self.lap2_w) * (v - self.phi*self.sigma) * (self.lap2)    
        "w   =  abs (dw) * self.dt + self.lap2_w"
        
        if (v>= self.sigma):
            w   =  abs (dw) * self.dt + self.lap2_w
        else: 
            w   =  self.lap2_w
        
        I   =   w * self._pre(self.lap2) + self.lap1_ * self._pre(self.lap1)

        return (-v+self.T+I)/self.tau , w
class cel_off():

    def __init__(self,CeL_ON):
    
        self.T  =   1.3
        self.tau=   13.0
        "self.tau=   12.8"
        self.sai=   31.9
        self.phi=   144
        
        self.celon_ =   -.7
        "self.celon_ =   -.8"

        self.celon  =   CeL_ON

    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v):
        I   =   self.celon_ * self._pre(self.celon)

        return (-v+self.T+I)/self.tau
class cem():

    def __init__(self,BAp5,BAp4,CeL_OFF,BAp4_w,BAp5_w,step):
    
        "transfer function parameters"
        self.T  =   1.3
        self.tau=   35.6 
        self.sai=   27.5
        self.phi=   25

        "synaptic weight & LTP"
        self.bap4_  =   2.0
        self.bap5_  =   1.8
        self.celoff_=   -2.265
        self.bap4_w =   BAp4_w
        self.bap5_w =   BAp5_w

        self.eta1    =   .000004
        self.eta2    =   .000005
        self.dt  =   step
        "input values"
        self.bap4   =   BAp4     
        self.bap5   =   BAp5     
        self.celoff   =   CeL_OFF     

    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v,DSE):
        if(DSE == 'Yes'):
            w4  =   self.bap4_w + (- self.eta1 * self.bap4_w * self.bap4) * self.dt
            w5  =   self.bap5_w + (- self.eta2 * self.bap5_w * self.bap5) * self.dt
        else:
            w4  =   self.bap4_w 
            w5  =   self.bap5_w 

        I   =   w4  *   self._pre(self.bap4) + w5 * self._pre(self.bap5) + self.celoff_* self._pre(self.celoff)
        return (-v+self.T+I)/self.tau , w4, w5 

class bap5():

    def __init__(self,LAp1):
    
        self.T  =   .3
        self.tau=   31.0
        self.sai=   28.1
        self.phi=   35.5
        
        self.lap1_  =   3.0
        "self.lap1_  =   3.01"
        self.lap1   =   LAp1

    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v):
        I   =   self.lap1_ * self._pre(self.lap1)
        return (-v+self.T+I)/self.tau
class bacck():

    def __init__(self,LAp1):
    
        self.T  =   5.3
        self.tau=   13.0
        "self.sai=   31.9"
        self.sai=   29.9
        self.phi=   144
        
        "self.lap1_  =   1.9"
        self.lap1_  =   2.9
        self.lap1   =   LAp1

    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v):
        I   =   self.lap1_ * self._pre(self.lap1)
        return (-v+self.T+I)/self.tau
class bap1():

    def __init__(self,LAp1,BAcck):
    
        self.T  =   .3
        self.tau=   31.0
        "self.sai=   28.1"
        self.sai=   27.6
        self.phi=   35.5
        
        self.lap1_  =   7.0804
        "self.lap1_  =   7.0807"
        self.bacck_ =   -.7
        self.lap1   =   LAp1
        self.bacck  =   BAcck

    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v):
        I   =   self.lap1_ * self._pre(self.lap1) + self.bacck_ * self._pre(self.bacck)
        return (-v+self.T+I)/self.tau
class bap2():

    def __init__(self,LAp1,BAcck,BAcck_w,STEP):
    
        self.T  =   .3
        self.tau=   15.0
        self.sai=   28.1
        self.phi=   35.5
        
        "self.lap1_  =   5.0"
        self.lap1_  =   5.00
        self.bacck_ =   -.7
        self.lap1   =   LAp1
        self.bacck  =   BAcck

        self.dt     =   STEP
        self.cck_w  =   BAcck_w
        self.eta    =   .0000005


    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v,DSI):

        if(DSI=='Yes'):
            dw  =   ( -self.eta * (self. cck_w) * (self.bacck)) 
            w   =   self.cck_w + dw * self.dt
            "print(DSI,dw,w)"
        else:
            w   =   self.cck_w
            "print(w)"
        I   =   self.lap1_ * self._pre(self.lap1) + (w) * self._pre(self.bacck)
        return (-v+self.T+I)/self.tau,w

class bap3():

    def __init__(self,ILp):
    
        self.T  =   .3
        self.tau=   31
        self.sai=   28.1
        self.phi=   35.5
        
        self.ilp_  =   3.3
        self.ilp   =   ILp


    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v):
        I   =   self.ilp_ * self._pre(self.ilp)
        return (-v+self.T+I)/self.tau
class bap4():

    def __init__(self,PLp,BApv,ITC,PLp_w,STEP):
        "transfer function parameters"
        self.T  =   .3
        self.tau=   31 
        self.sai=   28.1
        self.phi=   35.5
        self.dt=   STEP

        "synaptic weight & LTP"
        self.plp_   =   3.6
        self.bapv_  =   -.6
        self.itc_   =   -8.0
        self.plp_w  =   PLp_w
        self.eta1   =   .001
        self.eta2   =   .00000014
        self.sigma  =   .5

        "input values"
        self.plp    =   PLp
        self.bapv   =   BApv
        self.itc    =   ITC

    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v):
      
        dw1  =   self.eta1 * (3*self.plp_ - self.plp_w) * (v - self.phi*self.sigma) * (self.plp)
        dw2  =   self.eta2 * (3*self.plp_ - self.plp_w) * (v - self.phi*self.sigma) * (self.plp)
        
        if (v> self.sigma):
            "w   =   abs(dw1) * self.dt + self.plp_w"
            w   =   self.plp_w

        else: 
            w   =   -abs(dw2) * self.dt + self.plp_w
            "w   =   self.plp_w"

        I   =   w * self._pre(self.plp) + self.bapv_ * self._pre(self.bapv) + self.itc_ * self._pre(self.itc)

        return (-v+self.T+I)/self.tau, w

class bapv():

    def __init__(self,PLp):
    
        self.T      =   5.3
        self.tau    =   13.0
        self.sai    =   31.9
        self.phi    =   144
        
        self.plp_   =   .8
        self.plp    =   PLp


    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v):
        I   =   self.plp_ * self._pre(self.plp)
        return (-v+self.T+I)/self.tau

class pls():

    def __init__(self,BAp1):
    
        self.T  =   1.3
        self.tau=   29.2
        self.sai=   26.3
        self.phi=   75.5
        
        self.bap1_  =   1.2
        self.bap1   =   BAp1


    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v):
        I   =   self.bap1_ * self._pre(self.bap1)
        return (-v+self.T+I)/self.tau
class plpv():

    def __init__(self,BAp1,PLs):
    
        self.T  =   5.3
        self.tau=   7.7
        self.sai=   30.8
        self.phi=   163.3
        
        self.bap1_  =   1.4
        self.pls_   =   -1.35
        self.bap1   =   BAp1
        self.pls    =   PLs

    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v):
        I   =   self.bap1_ * self._pre(self.bap1) + self.pls_ * self._pre(self.pls)
        return (-v+self.T+I)/self.tau
class plp():

    def __init__(self,BAp1,PLpv):
    
        self.T  =   .3
        self.tau=   19
        self.sai=   22.2
        self.phi=   45.5
        
        self.bap1_  =   2.2
        self.plpv_  =   -1.2
        self.bap1   =   BAp1
        self.plpv   =   PLpv

    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v):
        I   =   self.bap1_ * self._pre(self.bap1) + self.plpv_ * self._pre(self.plpv)
        return (-v+self.T+I)/self.tau

class itc():

    def __init__(self,ILp,BAp3, BAp3_w,STEP):
    
        self.T  =   1.3
        self.tau=   23.5
        self.sai=   41.2
        "self.sai=   45"
        self.phi=   30.4
        self.dt =   STEP
        
        self.ilp_   =   1.0      
        self.bap3_  =   1.0        
        self.bap3_w =   BAp3_w        
        self.bap3   =   BAp3
        self.ilp    =   ILp

        "LTP synaptic weight"
        self.eta    =   .000000018 
        self.sigma  =   .5

    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v):

        dw = self.eta * (3 - self.bap3_w) * (v - self.phi*self.sigma) * self.bap3 
        "w   =  abs(dw) * self.dt + self.bap3_w"
        
        if (v>= self.sigma):
            w   =  abs(dw) * self.dt + self.bap3_w
        else: 
            w   =  self.bap3_w
        
        I   =   self.ilp_ * self._pre(self.ilp) + w * self._pre(self.bap3)
        
        I   =   self.ilp_ * self._pre(self.ilp) + self.bap3_ * self._pre(self.bap3)
        "print(v, I)"
        return (-v+ self.T+ I)/ self.tau, w

class ils():

    def __init__(self,BAp2):
    
        self.T  =   1.3
        self.tau=   29.2
        self.sai=   26.3
        self.phi=   75.5
        
        self.bap2_  =   2.5
        "self.bap2_  =   2.6"
        self.bap2   =   BAp2


    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v):
        I   =   self.bap2_ * self._pre(self.bap2)
        return (-v+self.T+I)/self.tau
class ilpv():

    def __init__(self,BAp2,ILs,US):
    
        self.T  =   5.3
        self.tau=   7.7
        self.sai=   30.8
        self.phi=   163.3
        
        self.bap2_  =   .3
        self.ils_   =   -.5
        "self.ils_   =   -.58"
        self.us_    =   7.0
        self.bap2   =   BAp2
        self.ils    =   ILs
        self.us     =   US

    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v):
        I   =   self.bap2_ * self._pre(self.bap2) + self.ils_ * self._pre(self.ils) + self.us_ *  self._pre(self.us)
        return (-v+self.T+I)/self.tau
class ilp():

    def __init__(self,PLp,BAp2,BAp2_w,ILpv,STEP):
        "transfer function parameters"
        self.T  =   .3
        self.tau=   19 
        self.sai=   22.2
        self.phi=   45.5
        self.dt=   STEP

        "synaptic weight & LTP"
        self.plp_   =   1.2
        self.bap2_  =   3.3
        self.ilpv_  =   -1.5
        self.bap2_w =   BAp2_w
        
        self.eta1   =   .0000002
        self.sigma  =   .5
        self.eta2   =   .0001
        
        "input values"
        self.plp    =   PLp
        self.bap2   =   BAp2      
        self.ilpv   =   ILpv      


    def _pre(self,U):
        return self.phi * max(0, math.tanh(U-self.sai))

    def unit(self,v):

        if (v>= self.sigma):
            dw  =   self.eta1 * (3*self.bap2_ - self.bap2_w) * (v - self.phi*self.sigma) * (self.bap2)
        else: 
            dw  =   self.eta2 * (3*self.bap2_ - self.bap2_w) * (v - self.phi*self.sigma) * (self.bap2)
        
        w   =  dw * self.dt + self.bap2_w
        "w   =   max(0,dw) * self.dt + self.cs_w"
        "print(dw,w)"
        I   =   w * self._pre(self.bap2) + (self.plp_ * self._pre(self.plp)) + (self.ilpv_ * self._pre(self.ilpv))
        return (-v+self.T+I)/self.tau, w

dt=1
T=np.arange(0,2700,dt)
CS  =   np.zeros(np.size(T))
US  =   np.zeros(np.size(T))
#Conditioning session
CS[90:110], CS[130:150], CS[170:190] =    100,    100,    100
US[109:110], US[149:150], US[189:190] =    100,   100,    100

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

LAp1, LAp2, LApv, LAvip =   [], [], [], []
Cel_ON, Cel_OFF, CeM, ITC   =   [], [], [], []
BAp1, BAp2, BAp3, BAp4, BAp5, BAcck, BApv   =   [], [], [], [], [], [], []
PLs, PLpv, PLp  =   [], [], []
ILs, ILpv, ILp  =   [], [], []
w1, w2, w3, w4, w5, w6, w7, w8  =   [], [], [], [], [], [], [], []
k1, k2  =   0, 0
p2_triger  =   "No"
cem_triger  =   "No"

for i,j in enumerate(T):
    if(i!= 0):
        



        dv6, bap3_w =   itc(ILp[i-1],BAp3[i-1],w6[i-1],dt).unit(ITC[i-1])
        V_itc   =   ITC[i-1] + dv6 * dt 
                    
        V_lap2  =   LAp2[i-1] + lap2(CS[i-1], LApv[i-1]).unit(LAp2[i-1]) * dt
        V_lavip =   LAvip[i-1] + lavip(US[i-1]).unit(LAvip[i-1]) * dt
        V_lapv  =   LApv[i-1] + lapv(CS[i-1],LAvip[i-1]).unit(LApv[i-1]) * dt
        dv1, cs_w   =   lap1(CS[i-1],US[i-1],LApv[i-1],w1[i-1],dt).unit(LAp1[i-1])
        V_lap1  =   LAp1[i-1] + dv1 * dt
        
        dv2, lap2_w =   cel_on(LAp1[i-1],LAp2[i-1],w2[i-1],dt).unit(Cel_ON[i-1])
        V_celon     =   Cel_ON[i-1] + dv2 * dt
        V_celoff    =   Cel_OFF[i-1] + cel_off(Cel_ON[i-1]).unit(Cel_OFF[i-1]) * dt
        
        V_bap5  =   BAp5[i-1] + bap5(LAp1[i-1]).unit(BAp5[i-1]) * dt
        V_bap1  =   BAp1[i-1] + bap1(LAp1[i-1],BAcck[i-1]).unit(BAp1[i-1]) * dt
        V_bacck =   BAcck[i-1] + bacck(LAp1[i-1]).unit(BAcck[i-1]) * dt
        
        V_bapv  =   BApv[i-1] + bapv(PLp[i-1]).unit(BApv[i-1]) * dt
        dv3, bacck_w =   bap2(LAp1[i-1],BAcck[i-1],w3[i-1],dt).unit(BAp2[i-1],p2_triger)
        V_bap2  =   BAp2[i-1] + dv3 * dt
        
        dv4, plp_w  =   bap4(PLp[i-1],BApv[i-1],ITC[i-1],w4[i-1],dt).unit(BAp4[i-1])
        V_bap4  =   BAp4[i- 1] + dv4 *dt
        
        V_bap3  =   BAp3[i-1] + bap3(ILp[i-1]).unit(BAp3[i-1]) * dt
        
        V_pls   =   PLs[i-1] + pls(BAp1[i-1]).unit(PLs[i-1]) * dt
        V_plpv  =   PLpv[i-1]+ plpv(BAp1[i-1],PLs[i-1]).unit(PLpv[i-1]) * dt
        V_plp   =   PLp[i-1] + plp(BAp1[i-1],PLpv[i-1]).unit(PLp[i-1]) * dt
        
        V_ils   =   ILs[i-1] + ils(BAp2[i-1]).unit(ILs[i-1]) * dt
        V_ilpv  =   ILpv[i-1]+ ilpv(BAp2[i-1],ILs[i-1],US[i-1]).unit(ILpv[i-1]) * dt
        dv5, bap2_w =   ilp(PLp[i-1],BAp2[i-1],w5[i-1],ILpv[i-1],dt).unit(ILp[i-1])
        V_ilp   =   ILp[i-1] + dv5 * dt
        
        dv7, bap4_w, bap5_w =   cem(BAp5[i-1],BAp4[i-1],Cel_OFF[i-1],w7[i-1],w8[i-1],dt).unit(CeM[i-1],cem_triger)
        V_cem   =   CeM[i-1] + dv7 * dt 
        V_plpv = max(0, V_plpv)
        V_ilpv = max(0, V_ilpv)
        #endocannabinoids receptor plasticity condition
        if(V_bap2>.001):
            k1  +=  1
            if(k1>=1000):
                p2_triger   =   "Yes"
        else: 
            k1  =   0
            p2_triger   =   "No"

        if(V_cem>.001):
            k2  +=  1
            if(k2>=1000):
                cem_triger  =    "Yes"
        else: 
            k2=0
            cem_triger  =   "No"

        
    else:
        cs_w    =   .4
        "cs_w    =   .58"
        lap2_w  =   1
        "lap2_w  =   1.15"
        bacck_w =   -.7
        plp_w   =   3.6
        bap2_w  =   3.3
        bap3_w  =   1.0
        bap4_w  =   2.0
        bap5_w  =   1.80
        V_lap1, V_lap2, V_lavip, V_lapv =   .0000001, .0000001, .0000001 , .0000001
        V_celon, V_celoff, V_cem, V_itc =    .0000001 , .0000001 , .0000001 , .0000001 
        V_bap1, V_bap2, V_bap3, V_bap4, V_bapv, V_bap5, V_bacck =   .0000001, .0000001, .0000001 , .0000001 , .0000001, .0000001, .0000001 
        V_pls, V_plpv, V_plp    =   .0000001, .0000001, .0000001  
        V_ils, V_ilpv, V_ilp    =   .0000001, .0000001, .0000001 
        

    w1      =   np.append(w1, cs_w)
    w2      =   np.append(w2, lap2_w)
    w3      =   np.append(w3, bacck_w)
    w4      =   np.append(w4, plp_w)
    w5      =   np.append(w5, bap2_w)
    w6      =   np.append(w6, bap3_w)
    w7      =   np.append(w7, bap4_w)
    w8      =   np.append(w8, bap5_w)
    LAp1    =   np.append(LAp1,V_lap1)
    LAp2    =   np.append(LAp2,V_lap2)
    LAvip   =   np.append(LAvip,V_lavip)
    LApv    =   np.append(LApv,V_lapv)
    
    Cel_ON  =   np.append(Cel_ON,V_celon)
    Cel_OFF =   np.append(Cel_OFF,max(0,V_celoff))
    CeM     =   np.append(CeM,max(0,V_cem))
    ITC     =   np.append(ITC,V_itc)
    
    BAcck   =   np.append(BAcck,V_bacck)
    BApv    =   np.append(BApv,max(0,V_bapv))
    BAp1    =   np.append(BAp1,V_bap1)
    BAp2    =   np.append(BAp2,V_bap2)
    BAp3    =   np.append(BAp3,V_bap3)
    BAp4    =   np.append(BAp4,max(0,V_bap4))
    BAp5    =   np.append(BAp5,V_bap5)

    PLs     =   np.append(PLs,V_pls)
    PLpv    =   np.append(PLpv,V_plpv)
    PLp     =   np.append(PLp,V_plp)

    ILs     =   np.append(ILs,V_ils)
    ILpv    =   np.append(ILpv,V_ilpv)
    ILp     =   np.append(ILp,max(0,V_ilp))

f0=plt.figure() 
plt.subplot(4,1,1)
plt.title('Dynamics of Units (Lateral, Basal, Central , and Cortex)')
plt.plot(T,CS,label='cs')
plt.plot(T,US,label='us')
plt.plot(T,LApv,label='LApv')
plt.plot(T,LAvip,label='LAvip')
plt.plot(T,LAp2,label='LAp2')
plt.plot(T,LAp1,label='LAp1')
plt.legend(loc='upper right')
plt.subplot(4,1,2)
plt.plot(T,Cel_ON,label='celoN')
plt.plot(T,Cel_OFF,label='celoff')
plt.plot(T,CeM,label='cem')
plt.plot(T,ITC,label='itc')
plt.legend(loc='upper right')
plt.subplot(4,1,3)
plt.plot(T,BAp1,label='BAp1')
plt.plot(T,BAp2,label='BAp2')
plt.plot(T,BAp3,label='BAp3')
plt.plot(T,BAp4,label='BAp4')
plt.plot(T,BAp5,label='BAp5')
plt.plot(T,BApv,label='BApv')
plt.plot(T,BAcck,label='BAcck')
plt.legend(loc='upper right')
plt.subplot(4,1,4)
plt.plot(T,PLs,label='pls')
plt.plot(T,PLpv,label='plpv')
plt.plot(T,PLp,label='plp')
plt.plot(T,ILs,label='ils')
plt.plot(T,ILpv,label='ilpv')
plt.plot(T,ILp,label='ilp')
plt.legend(loc='upper right')

print('LAp1= MAX: ',max(LAp1[:]), 'MIN: ',min(LAp1[:]))
print('CelON= MAX: ',max(Cel_ON[:]), 'MIN: ',min(Cel_ON[:]))
print('BAp1= MAX: ',max(BAp1[:]), 'MIN: ',min(BAp1[:]))
print('BAp2= MAX: ',max(BAp2[:]), 'MIN: ',min(BAp2[:]))
print('BAp5= MAX: ',max(BAp5[:]), 'MIN: ',min(BAp5[:]))
print('BAp4= MAX: ',max(BAp4[:]), 'MIN: ',min(BAp4[:]))
print('BAcck= MAX: ',max(BAcck[:]), 'MIN: ',min(BAcck[:]))

print('ILp= MAX: ',max(ILp[:]), 'MIN: ',min(ILp[:]))
print('PLp= MAX: ',max(PLp[:]), 'MIN: ',min(PLp[:]))
print('ITC= MAX: ',max(ITC[:]), 'MIN: ',min(ITC[:]))


US=((US[:]-min(US[:])) / (max(US[:]) - min(US[:])) )
CS=(CS[:]-min(CS[:])) / (max(CS[:]) - min(CS[:]))  
LAp1=((LAp1[:]-min(LAp1[:])) / (max(LAp1[:]) - min(LAp1[:])) )  
LAp2=(LAp2[:]-min(LAp2[:])) / (max(LAp2[:]) - min(LAp2[:])) 
LApv=(LApv[:]-min(LApv[:])) / (max(LApv[:]) - min(LApv[:]))  
LAvip=(LAvip[:]-min(LAvip[:])) / (max(LAvip[:]) - min(LAvip[:])) 

Cel_OFF=(Cel_OFF[:]-min(Cel_OFF[:])) / (max(Cel_OFF[:]) - min(Cel_OFF[:]))
Cel_ON=(Cel_ON[:]-min(Cel_ON[:])) / (max(Cel_ON[:]) - min(Cel_ON[:])) 
CeM=(CeM[:]-min(CeM[:])) / (max(CeM[:]) - min(CeM[:])) 
ITC=(ITC[:]-min(ITC[:])) / (max(ITC[:]) - min(ITC[:])) 

"""
ILp=ILp.clip(0)
ILpv=ILpv.clip(0)
"""
BAp1=((BAp1[:]-min(BAp1[:])) / (max(BAp1[:]) - min(BAp1[:])) )  
BAp2=((BAp2[:]-min(BAp2[:])) / (max(BAp2[:]) - min(BAp2[:])) )  
BAp3=((BAp3[:]-min(BAp3[:])) / (max(BAp3[:]) - min(BAp3[:])) )  
BAp4=((BAp4[:]-min(BAp4[:])) / (max(BAp4[:]) - min(BAp4[:])) )  
BAp5=((BAp5[:]-min(BAp5[:])) / (max(BAp5[:]) - min(BAp5[:])) )  
BApv=((BApv[:]-min(BApv[:])) / (max(BApv[:]) - min(BApv[:])) )  
BAcck=((BAcck[:]-min(BAcck[:])) / (max(BAcck[:]) - min(BAcck[:])) )  

ILs=((ILs[:]-min(ILs[:])) / (max(ILs[:]) - min(ILs[:])) )  
ILp=((ILp[:]-min(ILp[:])) / (max(ILp[:]) - min(ILp[:])) )  
ILpv=((ILpv[:]-min(ILpv[:])) / (max(ILpv[:]) - min(ILpv[:])) )

PLs=((PLs[:]-min(PLs[:])) / (max(PLs[:]) - min(PLs[:])) )  
PLp=((PLp[:]-min(PLp[:])) / (max(PLp[:]) - min(PLp[:])) )  
PLpv=((PLpv[:]-min(PLpv[:])) / (max(PLpv[:]) - min(PLpv[:])))

h=plt.figure()
arr =   np.vstack((US, CS, Cel_ON, Cel_OFF, LAvip, LApv, LAp1, LAp2, BAp5, BAp1, BAp2, BAp3, BAp4, BAcck, BApv, PLs, PLpv, PLp, ILs, ILpv, ILp, ITC, CeM))
y_axis_labels=['us','cs','CeL-ON','CeL_OFF','LAvip','LApv','LAp1','LAp2','BAp5','BAp1','BAp2', 'BAp3', 'BAp4', 'BAcck', 'BApv', 'PLs', 'PLpv', 'PLp', 'ILs', 'ILpv', 'ILp', 'ITC', 'CeM']

ax = sns.heatmap(arr, xticklabels=500, yticklabels=y_axis_labels, cmap="viridis")
"string='condition        Extinction1                        Extinction2        Reinstatement'"
ax.set_title('conditioning                                          Extinction Session 1                                                          Extinction seasion 2                                          Reinstatement')
locs, labels = plt.xticks()            # Get locations and labels
x_scale=np.arange(0,2700,500)
plt.xticks(locs, x_scale)

f1=plt.figure()
plt.subplot(4,1,1)
plt.title('Dynamics after normalization 0:1 ')
plt.plot(T,CS,label='cs')
plt.plot(T,US,label='us')
plt.plot(T,LApv,label='LApv')
plt.plot(T,LAvip,label='LAvip')
plt.plot(T,LAp2,label='LAp2')
plt.plot(T,LAp1,label='LAp1')
plt.legend(loc='upper right')
plt.subplot(4,1,2)
plt.plot(T,Cel_ON,label='celoN')
plt.plot(T,Cel_OFF,label='celoff')
plt.plot(T,CeM,label='cem')
plt.plot(T,ITC,label='itc')
plt.legend(loc='upper right')
plt.subplot(4,1,3)
plt.plot(T,BAp1,label='BAp1')
plt.plot(T,BAp2,label='BAp2')
plt.plot(T,BAp3,label='BAp3')
plt.plot(T,BAp4,label='BAp4')
plt.plot(T,BAp5,label='BAp5')
plt.plot(T,BApv,label='BApv')
plt.plot(T,BAcck,label='BAcck')
plt.legend(loc='upper right')
plt.subplot(4,1,4)
plt.plot(T,PLs,label='pls')
plt.plot(T,PLpv,label='plpv')
plt.plot(T,PLp,label='plp')
plt.plot(T,ILs,label='ils')
plt.plot(T,ILpv,label='ilpv')
plt.plot(T,ILp,label='ilp')
plt.legend(loc='upper right')

f2=plt.figure()
plt.subplot(2,1,1)
plt.title('synaptic plasticity weight changes ')
plt.plot(T,w1,label='cs to Lap1')
plt.plot(T,w2,label='lap2 to cel-on')
plt.plot(T,w5,label='BAp2 to ILp')
plt.legend(loc='upper right')
"plt.plot(T,w6,label='BAp3 to itc')"
plt.subplot(2,1,2)
plt.plot(T,w3,label='BA-cck to-p2')
plt.plot(T,w4,label='PLp to BAp4')
plt.plot(T,w7,label='BAp5 to Cem')
plt.plot(T,w8,label='BAp4 to CeM')
plt.legend(loc='upper right')

plt.show()

