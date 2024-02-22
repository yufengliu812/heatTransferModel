import CellModel
import thermalGraph
import ThermalProperty
import battConfig
import numpy as np 
import inspect
import util
from simController import Simplified_Controller

class BattModel:
    def __init__(self, N3D, battConfigName = "Bengal", SOH = 1, IndexJoul = None,
                 b_Cooling = True, b_Wall = False, b_Therm = False, b_CoolPower = False, b_SimController = False, DCFC_LUT_SOC_Crate = None):
        self.N3D = N3D
        self.SOH = SOH
        self.IndexJoul = IndexJoul
        self.b_Cooling = b_Cooling
        self.b_CoolPower = b_CoolPower
        self.b_Wall = b_Wall
        self.b_Therm = b_Therm
        self.b_SimController = b_SimController
        self.Ipack_Delivered = []
        self.Ipack_requested = []
        self.Timer = util.RunTimer()

        self.LoadBattModel(battConfigName)

        if(self.b_SimController):
            self.SC = Simplified_Controller(self.battConfig, self.cellModel, DCFC_LUT_SOC_Crate)


    def LoadBattModel(self, battConfigName):
        N3D = self.N3D
        SOH = self.SOH
        IndexJoul = self.IndexJoul
        b_Cooling = self.b_Cooling
        b_Wall = self.b_Wall
        b_Therm = self.b_Therm
        b_CoolPower = self.b_CoolPower

        if (battConfigName == "Bengal"):
            self.battConfig = battConfig.Bengal()
            self.thermModel = thermalGraph.ThermalGraphPrismaticBottom(N3D, self.battConfig,
                                                                       IndexJoul, b_Cooling, b_Wall, b_Therm,
                                                                       b_CoolPower = b_CoolPower, Tiemr = self.Timer)
            self.cellModel = CellModel.CellModel_Gotion_117Ah(SOH, Timer = self.Timer)

    def BattMdlInit(self, SOC0 = 0.5, TCell0 = 25, TCP0 = None, TCool0 = None):
        self.cellModel.CellInit(SOC0, TCell0)
        if TCP0 is None:
            TCP0 = TCell0
        if TCool0 is None:
            TCool0 = TCell0
        self.thermModel.TGInitCond(TCell0 = TCell0, TCP0 = TCP0, TCool0 = TCool0)


    def TimeIntegrationConstParam(self, StopTime, dt, Ipack, flowrate, Tcool, Vcell_limit = 2.5, 
                                  SOC_LowLimit = 0.0, SOC_UpperLimit = 1.0):
        Icell = Ipack/self.battConfig.N_cells_parallel
        Tcell = self.thermModel.GetCellTempAvg()
        if Ipack>0:
            charge_mode = 1
        else:
            charge_mode = 0
        for i in range(1, int(StopTime/dt)+1):
            self.cellModel.UnitIntegration(dt, charge_mode, Tcell, Icell)
            Qcell = self.cellModel.CellQ[-1]
            self.thermModel.TimeIntegralUnit(dt, Qcell, 0, flowrate, Tcool)
            Tcell = self.thermModel.GetCellTempAvg()
            Vcell = self.cellModel.CellV[-1]
            SOC = self.cellModel.SOC[-1]
            if Vcell < Vcell_limit:
                return
            if SOC < SOC_LowLimit:
                return
            if SOC > SOC_UpperLimit:
                return
            
    def GetIpackDelivered(self, Controller, SOC, Tcell):
        ipack = Controller.IpackCrateDCFCSOC(SOC)
        ipack_Delivered = Controller.IpackDelivered(ipack, Tcell)
        return ipack_Delivered
    
    def TimeIntegration(self, Time, Ipack, Flowrate, Tcool, Pcool, charge_mode = 1, Vcell_limit = (2.5, 4.2), SOC_limit = (0.0, 1.0)):
        self.Check_1DArray(Ipack, Time)
        self.Check_1DArray(Flowrate, Time)
        self.Check_1DArray(Tcool, Time)

        dtTime = Time[1:] - Time[0:-1]
        dtTime = np.insert(dtTime, 0, dtTime[0])
        Icell = Ipack/self.battConfig.N_cells_parallel
        Tcell = self.thermModel.GetCellTempAvg()

        b_CoolPower = self.thermModel.b_CoolPower

        if b_CoolPower:
            Tcool = Pcool
        else:
            Pcool = Tcool

        for dt, icell, flowrate, tcool, pcool in zip(dtTime, Icell, Flowrate, Tcool, Pcool):
            SOC = self.cellModel.SOC[-1]
            Vcell = self.cellModel.CellV[-1]

            if Vcell<Vcell_limit[0] or Vcell>Vcell_limit[1]:
                break
            if SOC < SOC_limit[0] or SOC > SOC_limit[1]:
                break
            
            ipack = icell*self.battConfig.N_cells_parallel
            self.Ipack_requested.append(ipack)

            if (self.b_SimController):
                ipack_Delivered = self.GetIpackDelivered(self.SC, SOC, Tcell)
                self.Ipack_Delivered.append(ipack_Delivered)
            else:
                self.Ipack_Delivered.append(ipack)
            Tcell = self.thermModel.GetCellTempAvg()
            self.cellModel.UnitIntegration(dt, charge_mode, Tcell, icell)
            Qcell = self.cellModel.CellQ[-1]
            QJoul = icell**2*self.battConfig.Rcell_CCA

            if b_CoolPower:
                self.thermModel.TimeIntegralUnitImplicit(dt, Qcell, QJoul, flowrate, Tcool = None, Pcool = pcool)
            else:
                self.thermModel.TimeIntegralUnitImplicit(dt, Qcell, QJoul, flowrate, Tcool = tcool, Pcool = None)

        self.thermModel.OnSimComplete()

    
    def Check_1DArray(self, Time, Array):
        if not(isinstance(Array, np.ndarray) and Array.ndim  == 1 and Array.shape == Time.shape):
            raise TypeError("Input Array should be a 1-col Array with the same dimension as Time")




