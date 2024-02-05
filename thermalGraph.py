import math
import pandas as pd 
import numpy as np 
from collections import Counter
import Node
from util import RunTimer
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import LACPP

import ThermalProperty

class ThermalGraph:
    def __init__(self, N3D, battConfig, IndexJoul = None, 
                 b_Cooling = True, b_Wall = False, b_Therm = False, b_CoolPower = False, Timer = None):
        if Timer is None:
            self.Timer = RunTimer()
        else:
            self.Timer = Timer

        list_strTime = ["Therm_simTime", "Therm_InitTime", "Therm_TDFTime", "Therm_PrintTime"]
        self.Timer.Add(list_strTime)

        self.Timer.Start("Therm_InitTime")

        self.N3D = N3D
        N1, N2, N3 = N3D

        self.IndexJoul = IndexJoul

        self.battConfig = battConfig
        battThermProperty = battConfig.battThermProperty
        CellThermProperty = battThermProperty.Cell 
        CPThermProperty = battThermProperty.CP 
        CoolantThermProperty = battThermProperty.Coolant 

        self.CellNodeArr = np.empty((N1, N2, N3), dtype = object)
        self.CellTempArr = []
        self.CoolTempArr = []

        self.HeatRejArr = []
        self.SimTime = []

        self.CreateCellNodes(CellThermProperty)
        self.CreateCPNode(CPThermProperty)
        self.CrateCoolantNode(CoolantThermProperty)
        self.b_CoolingNode = b_Cooling
        self.b_CoolPower = b_CoolPower

        self.N = N1*N2*N3 + 2
        self.NCellNodes = N1*N2*N3

        self.matR = np.zeros((self.N, self.N))
        self.defThermalGraphMat()

        self.simStatus = -1

        self.Timer.Stop("Therm_InitTime")

    def find_IJK_3D(self, idx):
        m, n, l = self.N3D
        if idx >= m*n*l or idx<0:
            raise ValueError("input Index out of range")
        i = idx//(n*l)
        j = (idx%(n*l))//l
        k = idx % l
        return (i,j,k)
    
    def find_Index_3D(self, ijk):
        i, j, k = ijk
        m, n, l = self.N3D

        if not(0<= i <m) or not (0<= j <n) or not (0<= k < l):
            raise ValueError("input IJK out of range")
        idx = i*n*l + j*l + k
        return idx
    
    def TGInitCond(self, TCell0, TCP0 = None, TCool0 = None):
        if TCP0 is None:
            TCP0 = TCell0
        if TCool0 is None:
            TCool0 = TCell0
        
        if (not self.CellTempArr):
            CellTempArr = np.full(self.N, TCell0)
            CellTempArr[-1] = TCP0
            self.CellTempArr.append(CellTempArr)
        if(not self.CoolTempArr):
            self.CoolTempArr.append(TCool0)
        if(not self.HeatRejArr):
            self.HeatRejArr.append(0)
        self.SimTime.append(0)
        if(self.simStatus < 0):
            self.simStatus = 0

    def CreateCellNode(self, CellThermProperty):
        N3D = self.N3D
        N1, N2, N3 = N3D
        
        if(any(x == 0 for x in N3D)):
            raise ZeroDivisionError("The input of N3D has 0")
        
        mass = CellThermProperty['mass']
        L3D = CellThermProperty['L3D']
        dL3D = L3D/N3D
        d1, d2, d3 = dL3D
        
        for i in range(N1):
            x = d1*(0.5 + i)
            for j in range(N2):
                y = d2*(0.5 + j)
                for k in range(N3):
                    z = d3*(0.5 + k)

                    Loc3D = np.array([x, y, z])
                    dmass = self.CalculateDmass(L3D, N3D, Loc3D, mass)
                    A3D = self.CalculateA3D(L3D, N3D, Loc3D)
                    r_Joul = 0
                    if((self.IndexJoul is not None) and ((i,j,k) in self.IndexJoul)):
                        r_Joul = 1.0/len(self.IndexJoul)
                    r_EQC = dmass/mass
                    idx3D = (i,j,k)
                    node = self.AddCellNode(dmass, CellThermProperty, dL3D, Loc3D, A3D, idx3D,
                                            r_EQC = r_EQC, r_Joul = r_Joul)
                    self.CellNodeArr[i,j,k] = node
                    self.CreateCellMap(i,j,k)

    def CreateCPNode(self, CPThermProperty):
        CPnode = self.AddCPNode(CPThermProperty)
        arr = self.CellNodeArr
        N1, N2, N3 = self.N3D
        for i in range(N1):
            for j in range(N2):
                for k in range(N3):
                    if (self.CheckIfCool((i,j,k))):
                        cell = arr[i,j,k]
                        CPnode.LinkAdjCell(cell)
                        print(f"Cold Plate linked to cell {i+1, j+1, k+1}")
        self.CP = CPnode

    def CreateCoolantNode(self, CoolantThermProperty):
        coolant_mass = CoolantThermProperty['coolant_mass']
        coolant_mass /= self.battConfig.N_cells
        coolant_cp = CoolantThermProperty['coolant_cp']
        self.CoolantNode = Node.CoolantNode(coolant_mass, coolant_cp)

    def defThermalGraphMat(self):
        self.defCellParam()
        self.defMatR()

    def defCellParam(self):
        eps = np.finfo(float).eps
        CellNodeArr = self.CellNodeArr
        NCellNodes = self.NCellNodes

        alpha = np.zeros(self.N)
        r_EQC = np.zeros(self.N)
        r_Joul = np.zeros(self.N)

        for idx_x in range(NCellNodes):
            i_tgt, j_tgt, k_tgt = self.find_IJK_3D(idx_x)
            node_tgt = CellNodeArr[i_tgt, j_tgt, k_tgt]

            alpha[idx_x] = 1/2/(node_tgt.mass*node_tgt.cp+eps)
            r_EQC[idx_x] = node_tgt.r_EQC
            r_Joul[idx_x] = node_tgt.r_Joul

        CPNode = self.CP
        alpha[-2] = 1/2/(CPNode.mass * CPNode.cp + eps)

        CoolantNode = self.CoolantNode
        alpha[-1] = 1/2(CoolantNode.mass * CoolantNode.cp + eps)

        self.alpha = alpha
        self.r_EQC = r_EQC
        self.r_Joul = r_Joul

    def defMatR(self):
        eps = np.finfo(float).eps
        CellNodeArr = self.CellNodeArr
        NCellNodes = self.NCellNodes
        matR = self.matR
        
        for idx_x in range(NCellNodes):
            i_tgt, j_tgt, k_tgt = self.find_IJK_3D(idx_x)
            node_tgt = CellNodeArr[i_tgt, j_tgt, k_tgt]
            list_adjCellNodes = node_tgt.AdjacentNodes
            for idx_adj in list_adjCellNodes:
                i_adj, j_adj, k_adj = idx_adj
                idx_y = self.find_Index_3D(idx_adj)
                adjCellNode = CellNodeArr[i_adj, j_adj, k_adj]
                Rt = node_tgt.CalTheramlResist(adjCellNode)
                matR[idx_x, idx_y] = -1/(Rt + eps)
                matR[idx_x, idx_x] += 1/(Rt + eps)

        CPNode = self.CP
        list_adjCellNodesCP = CPNode.AdjacentCells
        for idx_adj in list_adjCellNodesCP:
            i_adj, j_adj, k_adj = idx_adj
            idx_y = self.find_Index_3D(idx_adj)
            adjCellNode = CellNodeArr[i_adj, j_adj, k_adj]
            Rt2CP = adjCellNode.GetRt2CP()
            matR[idx_y, -2] = -1/(Rt2CP + eps)
            matR[idx_y, idx_y] += 1/(Rt2CP + eps)
            matR[-2, idx_y] = -1/(Rt2CP + eps)
            matR[-2, -2] += 1/(Rt2CP + eps)
        

    def addRtCool(self, flowrate):
        eps = np.finfo(float).eps
        matR = self.matR.copy()
        CP = self.CP
        Rt_Coolant = CP.CalRValueCoolant(flowrate)
        matR[-2, -2] += 1/(Rt_Coolant + eps)
        matR[-2, -1] = -1/(Rt_Coolant + eps)
        if self.b_CoolPower:
            matR[-1, -1] += 1/(Rt_Coolant + eps)
            matR[-1, -2] = -1/(Rt_Coolant + eps)
        else:
            matR[-1, -1] = 0
        return matR
    
    def defMatA(self, dt, flowrate):
        I = np.identity(self.N)
        matR = self.addRtCool(flowrate)
        #Alpha = np.diag(self.alpha)
        #matR = np.dot(Alpha, matR)*dt + I
        matA = LACPP.diag_matmul(self.alpha, matR)*dt + I
        return matA
    
    def defMatB(self, dt, flowrate):
        I = np.identity(self.N)
        matR = self.addRtCool(flowrate)
        #Alpha = np.diag(self.alpha)
        #matB = -np.dot(Alpha, matR)*dt +I
        matB = -LACPP.diag_matmul(self.alpha, matR)*dt + I
        return matB
    
    def defb(self, dt, flowrate, Tn):
        matB = self.defMatB(dt, flowrate)

        b = np.dot(matB, Tn)
        return b
    
    def defQ(self, Q_EQC, Q_Joul, Pcool):
        r_EQC = self.r_EQC
        r_Joul = self.r_Joul

        Q = Q_EQC*r_EQC + Q_Joul *r_Joul
        if(self.b_CoolPower):
            Q[-1] += Pcool/self.battConfig.N_cells
        return Q
    
    def defThermalGraphEq(self, dt, flowrate, Tn):
        A = self.defMatA(dt, flowrate)
        b = self.defb(dt, flowrate, Tn)
        return (A, b)
    
    def IntegralUnit(self, dt, Q_EQC, Q_Joul, flowrate, Tcool, Pcool):
        self.Timer.Start("Therm_SimTime")

        # Finish fist step of heat gen, explicit
        Tn = self.CellTempArr[-1].copy()
        Q = self.defQ(Q_EQC, Q_Joul, Pcool)
        alpha = self.alpha
        Tn = np.array(Tn, dtype=np.float64)
        Tn += Q*2*alpha*dt

        #get b after Q
        A, b = self.defThermalGraphEq(dt, flowrate, Tn)

        # Assuming A and b are defined and A is sparse
        # A_sparse = csr_matrix(A)
        # x = spsolve(A_sparse, b)

        x = LACPP.solve_sparse(A, b)
        if not self.b_CoolPower:
            x[-1] = Tcool
        self.CoolTempArr.append(x[-1])
        #x = np.linalg.solve(A, b)
        self.CellTempArr.append(x)

        self.Timer.Stop("Therm_SimTime")

    def HeatRejCal(self, flowrate):
        x = self.CellTempArr[-1]
        TCP = x[-2]
        Tcool = x[-1]
        HeatRej = self.CP.CalCellHeatRej(flowrate, TCP, Tcool)
        self.HeatRejArr.append(HeatRej)

    def AddSimTimeUnit(self, dt):
        simTime = self.SimTime[-1]
        self.SimTime.append(simTime + dt)

    def CheckCool(self, Tcool, Pcool):
        if(self.b_CoolPower):
            if(Pcool is None):
                raise ValueError("No Cooling Power Input")
                return False
        else:
            if(Tcool is None):
                raise ValueError("No Coolant Temp Input")
                return False
        return True
    
def TimeIntegralUnitImplicit(self, dt, Q_EQC, Q_Joul=0, flowrate=20, Tcool = None, Pcool = None):
    if self.CehckCool(Tcool, Pcool):
        self.IntegralUnit(dt, Q_EQC, Q_Joul, flowrate, Tcool, Pcool)
        self.HeatRejCal(flowrate)
        self.AddSimTimeUnit(dt)

def ConvertArr(self):
    self.CellTempArr = np.array(self.CellTempArr)
    self.CoolTempArr = np.array(self.CoolTempArr)
    self.HeatRejArr = np.array(self.HeatRejArr)
    self.SimTime = np.array(self.SimTime)

def TimeIntegralConstParam(self, dt, StopTime, Q_EQC, Q_Joul = 0, flowrate = 20, Tcool = 20, Pcool = None):
    if self.b_CoolPower:
        Tcool = None
    else:
        Pcool = None
    
    


        