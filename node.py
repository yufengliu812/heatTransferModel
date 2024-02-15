import numpy as np
import math
from util import PreLookup, Util_Methods

class Node:
    def __init__(self, mass, cp):
        self.mass = mass
        self.cp = cp

class CellNode(Node):
    """ A unit heat transfer node of cell"""
    def __init__(self, mass, cp, idx3D):
        super().__init__(mass, cp)
        self.idx3D = idx3D
        self.AdjacentNodes = []
        self.Rt2CP = 1E4
        self.A2Cool = np.finfo(float).eps

    def GetCellNodeIdx(self):
        return self.idx3D
    
    def LinkAdjCell(self, adjCellNode):
        self.AdjacentNodes.append(adjCellNode.GetCellNodeIdx())

    def Lind2AdjCell(self, adjCellNode):
        adjCellNode.AdjacentNodes.apped(self.GetCellNodeIdx())

    def CellThermalResist(self, adjCellNode):
        return self.CalRValue(adjCellNode)
    
    def GetRt2CP(self):
        return self.Rt2CP
    
class CellNodePrismatic(CellNode):
    """ A unit heat transfer node of Prismatic Cell"""
    def __init__(self, mass, cp, kxyz, Locxyz, Lxyz, Axyz, idx3D, kt_jr_ins, r_EQC = 1, r_Joul = 0):
        super().__init__(mass, cp, idx3D)

        self.xyz = Locxyz
        self.kxyz = kxyz
        self.Axyz = Axyz
        self.Lxyz = Lxyz
        self.k_jr_ins = kt_jr_ins[0]
        self.t_jr_ins = kt_jr_ins[1]
        self.r_EQC = r_EQC
        self.r_Joul = r_Joul

    def CalRValue(self, adjCellNode):
        dxyz = adjCellNode.xyz - self.xyz
        Axyz = self.Axyz
        kxyz = self.kxyz
        eps = np.finfo(float).eps
        R3D = np.abs(dxyz)/(kxyz*Axyz + eps)
        return R3D.sum()
    
    def GetA2Cool(self, index2Cool = 2, phi_range = [(0,2*math.pi)], z_range = (0, 0.07)):
        i = index2Cool
        self.A2Cool = self.Axyz[i]
        eps = np.finfo(float).eps
        Rt = self.Lxyz[i]*0.5/(self.kxyz[i] + eps)/(self.A2Cool + eps)
        Rt += self.t_jr_ins/(self.k_jr_ins + eps)/(self.A2Cool + eps)
        self.Rt2CP = Rt
        return True
    
class CellNodeCyl(CellNode):
    """ A unit heat transfer node of cell: cylindrical cell only"""
    def __init__(self, mass, cp, krpz, Locrpz, Lrpz, Arpz, idx3D, kt_jr_ins, r_EQC = 1, r_Joul = 0):
        super().__init__(mass, cp, idx3D)

        self.rpz = Locrpz
        self.krpz = krpz
        self.Lrpz = Lrpz
        self.Arpz = Arpz
        self.k_jr_ins = kt_jr_ins[0]
        self.t_jr_ins = kt_jr_ins[1]
        self.r_EQC = r_EQC
        self.r_Joul = r_Joul
        self.AdjacentNodes = []

    def CalRValue(self, adjCellNode):
        drpz = np.abs(adjCellNode.rpz - self.rpz)
        dphi = min(drpz[1], 2*np.pi - drpz[1])
        r =(self.rpz[0] + adjCellNode.rpz[0])/2
        d3D = (drpz[0], dphi*r, drpz[2])
        Arpz = (self.Arpz + adjCellNode.Arpz)/2
        krpz = self.krpz
        eps = np.finfo(float).eps
        R3D = np.abs(d3D)/(krpz*Arpz + eps)
        return R3D.sum()
    
    def GetA2Cool(self, index2Cool = 0, phi_range = [(0, 2*math.pi)], z_range = (0,0.07)):
        if (index2Cool != 0 and index2Cool != 2):
            raise Exception("Input Cell to Coolant in 3D index not value!")
        b_A2 = False
        if index2Cool == 0:
            r, phi, z = self.rpz
            dr, dphi, dz = self.Lrpz

            phi_span = (phi - dphi*0.5, phi + dphi*0.5)
            z_span = (z - dz*0.5, z + dz*0.5)
            for phi_zone in phi_range:
                phi_comp = (max(phi_span[0], phi_zone[0]), min(phi_span[1], phi_zone[1]))
                z_comp = (max(z_span[0], z_range[0]), min(z_span[1], z_range[1]))

                if (phi_comp[0] < phi_comp[1] and z_comp[0] < z_comp[1]):
                    self.A2Cool += (phi_comp[1] - phi_comp[0]) *(r+dr*0.5)*(z_comp[1] - z_comp[0])
                    b_A2 = True

        else:
            self.A2Cool = self.Arpz[2]
            b_A2 = True

        eps = np.finfo(float).eps
        i = index2Cool
        Rt = self.Lrpz[1]*0.5/(self.krpz[i] + eps)/(self.A2Cool + eps)
        Rt += self.t_jr_ins/(self.k_jr_ins + eps)/(self.A2Cool + eps)
        self.Rt2CP = Rt
        
        return b_A2

class ColdPlateNode(Node):
    """ A unit heat transfer node ofr cold plate"""
    def __init__(self, mass, cp, Rt_tglue, Rt_pc, Rt_Cool_LUT, bp_flowrate):
        super().__init__(mass, cp)
        self.Rt_tglue = Rt_tglue
        self.Rt_pc = Rt_pc
        self.bp_flowrate = bp_flowrate
        self.Rt_Cool_LUT = Rt_Cool_LUT
        self.AdjacentCells = []
        self.HeatRej = np.array([0])

    def RtCoolLUT(self, flowrate):
        """ Coolant resistance lookup table flowrate in unit kg/s"""
        preLookup = PreLookup(self.Rt_Cool_LUT, self.bp_flowrate)
        return preLookup
    
    def LinkAdjCell(self, adjCellNode):
        self.AdjacentCells.append(adjCellNode.GetCellNodeIdx())

    def CalRValueCoolant(self, flowrate):
        Rt_Coolant = self.RtCoolLUT(flowrate) + self.Rt_pc
        return Rt_Coolant

    def CalCellHeatRej(self, flowrate, TCP, TCoolant):
        Rt_Coolant = self.CalRValueCoolant(flowrate)
        Q_Coolant = (TCP - TCoolant)/Rt_Coolant
        return Q_Coolant
    
class CoolantNode(Node):
    """ A unit heat transfer node for coolant"""
    def __init__(self, mass, cp):
        super().__init__(mass, cp)

    def HeatTransfer(self, HeatRej, HeatCool, TCool):
        Q = HeatRej + HeatCool
        return TCool + Q/self.mass/self.cp