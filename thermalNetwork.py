import math
import pandas as pd
import numpy as np 
from collections import Counter
import Node 
from util import RunTimer

class ThermalGraph:
    def __init__(self, n3D, config, b_Cooling = True, b_Wall = False, b_Therm = False, b_CoolPower = False, Timer = None):
        if Timer is None:
            self.Timer = RunTimer()
        else:
            self.Timer = Timer
        list_strTime = ["Therm_SimTime", "Therm_InitTime", "Therm_TDFTime", "Therm_PrintTime"]
        self.Timer.Add(list_strTime)
        self.Timer.Start("Therm_InitTime")

        # Dimension of thermal network in 3D
        self.n3D = n3D
        n1, n2, n3 = n3D

        # Load and register all obj thermal properties
        self.config = config
        thermProperty = config.thermProperty
        objThermProperty = thermProperty.obj
        coldplateThermProperty = thermProperty.coldplate 
        coolantThermProperty = thermProperty.coolant 

        # Create containers for obj nodes
        self.objNodeArr = np.empty((n1,n2,n3))
        self.objTempArr = []
        self.coolantTempArr = []
        self.heatRejArr = []
        self.simTime = []

        # Add each node of cell, coldplate and coolant
        self.CreateObjNodes(objThermProperty)
        self.CreateColdPlateNode(coldplateThermProperty)
        self.CreateCoolantNode(coolantThermProperty)
        self.b_CoolingNode = b_Cooling
        self.b_CoolingPower = b_CoolPower

        # Num of nodes + coldplate and coolant heat transfer as heat source
        self.n = n1*n2*n3 + 2
        self.nObjNodes = n1*n2*n3

        self.matR = np.zeros(self.n, self.n)
        self.Thermal_Network_Mat()

        # Status of simulations
        # -1 for not initiate yet
        # 0 initiated but not run yet
        # 1 complete

        self.simStatus = -1
        self.Timer.Stop("Therm_InitTime")

    def Find_IJK_3D(self, idx):
        m, n, l = self.n3D
        if idx >= m*n*l or idx < 0:
            raise ValueError("Input index out of range")
        i = idx // (n * l)
        j = (idx % (n * l))//l
        k = idx % l
        return (i, j, k)
    
    def Find_Index_3D(self, ijk):
        i, j, k = ijk
        m, n, l = self.n3D
        if not(0 <= i < m) or not(0 <= j < n) or not(0 <= k < l):
            raise ValueError("Input ijk out of range")
        idx = i * n * l + j * l + k
        return idx 
    
    def Init_Cond(self, T0, Tcp0 = None, Tcool0 = None):
        if Tcp0 is None:
            Tcp0 = T0
        if Tcool0 is None:
            Tcool0 = T0

        if (not self.objTempArr):
            objTempArr = np.full(self.n, T0)
            objTempArr[-1] = Tcp0
            self.objTempArr.append(objTempArr)
        if (not self.coolantTempArr):
            self.coolantTempArr.append(Tcool0)
        if (not self.heatRejArr):
            self.heatRejArr.append(0)
        self.simTime.append(0)
        if (self.simStatus < 0):
            self.simStatus = 0

    def Create_Obj_Nodes(self, objThermProperty):
        n3D = self.n3D
        n1, n2, n3 = n3D

        if (any(x == 0 for x in n3D)):
            raise ZeroDivisionError("The input of n3D has 0")
        
        mass = objThermProperty['mass']
        l3D = objThermProperty['l3D']
        dL3D = l3D/n3D
        d1, d2, d3 = dL3D

        for i in range(n1):
            x = d1 * (0.5 + i)
            for j in range(n2):
                y = d2 * (0.5 + j)
                for k in range(n3):
                    z = d3 * (0.5 + k)

                    loc3D = np.array([x, y, z])
                    dmass = self.Calculate_Dmass(l3D, n3D, loc3D, mass)
                    a3D = self.Calculate_A3D(l3D, n3D, loc3D)

                    r_heating = dmass/mass
                    idx3D = (i, j, k)
                    node = self.Add_Obj_Node(dmass, objThermProperty, dL3D, loc3D, a3D, idx3D, r_heating = r_heating)
                    self.objNodeArr[i, j, k] = node
                    self.Create_Obj_Map(i, j, k)

    def Create_ColdPlate_Node(self, coldplateThermProperty):
        coldPlateNode = self.Add_Coldplate_Node(coldplateThermProperty)
        arr = self.objNodeArr
        n1, n2, n3 = self.n3D
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    if(self.Check_If_Cool((i, j, k))):
                        obj = arr[i, j, k]
                        coldPlateNode.Link_Adj_Obj(obj)
                        print(f"Coldplate linked to obj {i+1, j+1, k+1}")
        self.coldPlateNode = coldPlateNode

    def Create_Coolant_Node(self, coolantThermProperty):
        coolantMass = coolantThermProperty['coolantMass']
        coolantMass /= self.config.n_objs
        coolantCp = coolantThermProperty['coolantCP']
        self.coolantNode = Node.Coolant_Node(coolantMass, coolantCp)

    def Thermal_Network_Mat(self):
        self.Cell_Param()
        self.Mat_R()

    def Cell_Param(self):
        eps = np.finfo(float).eps
        objNodeArr = self.objNodeArr
        nOjbNodes = self.nObjNodes

        alpha = np.zeros(self.n)
        r_heating = np.zeros(self.n)

        for idxTgt in range(nOjbNodes):
            iTgt, jTgt, kTgt = self.Find_IJK_3D(idxTgt)
            nodeTgt = objNodeArr[iTgt, jTgt, kTgt]

            alpha[idxTgt] = 1/2/(nodeTgt.mass * nodeTgt.cp + eps)
            r_heating[idxTgt] = nodeTgt.r_heating 

        coldPlateNode = self.coldPlateNode
        alpha[-2] = 1/2/(coldPlateNode.mass * coldPlateNode.cp + eps)

        coolantNode = self.coolantNode
        alpha[-1] = 1/2/(coolantNode.mass * coolantNode.cp + eps)

        self.alpha = alpha
        self.r_heating = r_heating

    def Mat_R(self):
        eps = np.finfo(float).eps
        objNodeArr = self.objNodeArr
        nObjNodes = self.nObjNodes
        matR = self.matR

        for idxX in range(nObjNodes):
            iTgt, jTgt, kTgt = self.Find_IJK_3D(idxX)
            nodeTgt = objNodeArr[iTgt, jTgt, kTgt]
            listAdjObjNodes = nodeTgt.adjNodes 
            for ijkAdj in listAdjObjNodes:
                iAdj, jAdj, kAdj = ijkAdj
                idxY = self.Find_Index_3D(idxY)
                adjObjNode = objNodeArr[iAdj, jAdj, kAdj]
                rt = nodeTgt.Cal_Therm_Resist(adjObjNode)
                matR[idxX, idxY] = -1/(rt + eps)
                matR[idxX, idxX] += 1/(rt + eps)

        coldplatNode = self.coldPlateNode
        listAdjObjNodesColdplate = coldplatNode.adjObjNodes
        for ijkAdj in listAdjObjNodesColdplate:
            iAdj, jAdj, kAdj = ijkAdj
            idxY = self.Find_IJK_3D(ijkAdj)
            adjObjNode = objNodeArr[iAdj, jAdj, kAdj]
            rt2Coldplate = adjObjNode.Get_RT2ColdPlate()
            matR[idxY, -2] = -1(rt2Coldplate + eps)
            matR[idxY, idxY] += 1/(rt2Coldplate + eps)
            matR[-2, idxY] = -1(rt2Coldplate + eps)
            matR[-2, -2] += 1/(rt2Coldplate + eps)


def Add_Rt_Cool(self, flowrate):
    eps = np.finfo(float).eps
    matR = self.matR.copy()
    coldPlateNode = self.coldPlateNode

    rtCoolant = coldPlateNode.Cal_Rt_Coolant(flowrate)
    matR[-2, -2] += 1/(rtCoolant + eps)
    matR[-2, -1] = -1/(rtCoolant + eps)
    
    if self.b_CoolPower:
        matR[-1, -1] += 1/(rtCoolant + eps)
        matR[-1, -2] = -1(rtCoolant + eps)
    else:
        matR[-1, -2] = 0
    return matR

def MatA(self, dt, flowrate):
    I = np.identity(self.n)
    matR = self.Add_Rt_Cool(flowrate)
    alpha = np.diag(self.alpha)
    matA = np.dot(alpha, matR)*dt + I
    #matA = LACPP.diag_matmul(self.alpha, matR) * dt + I
    return matA

