import numpy as np 
from scipy.interpolate import RegularGridInterpolator
from util import Util_Methods, RunTimer
from scipy.interpolate import griddata
import itertools

class Container:
    def __call__(self) -> None:
        pass

class Interpolator:
    def __init__(self, strSPList, strLUTField, strLUTList, CellParam):
        self.DimLUT = len(strSPList)
        self.strSPList - strSPList
        self.dataField = getattr(CellParam, strLUTField)
        self.Util_Methods = Util_Methods()
        self.GetLUTSP(strSPList = strSPList, strLUTList = strLUTList)

    def GetSetAttr(self, strContainer, Attr, AttrList):
        for strAttr in AttrList:
            value = getattr(Attr, strAttr)
            container = getattr(self, strContainer)
            setattr(container, strAttr, value)

    def GetLUTSP(self):
        print("Dummy")

    def GetLUTValue(self, strTable, SPList, Timer = None):
        if(len(dir(self.LUT)) <= 0):
            LUT = self.LUT
        else:
            if not isinstance(strTable, str):
                LUT = self.LUT
            else:
                if hasattr(self.LUT, strTable):
                    LUT = getattr(self.LUT, strTable)
                else:
                    raise NameError(f"The input table name {strTable} not exists")
        
        start_indices = []
        ratios = []

        strLUTSearch = "EQC_LUT_Search"
        if Timer is None:
            self.Timer = RunTimer()
        else:
            self.Timer = Timer

        self.Timer.Add(strLUTSearch)
        self.Timer.Start(strLUTSearch)

        for strSP, input_SP in zip(self.strSPList, SPList):
            SP = getattr(self.SP, strSP)
            idx, ratio = Util_Methods.find_idx_ratio_clip(SP, input_SP)

            start_indices.append(idx)
            ratios.append(ratio)

        self.Timer.Stop(strLUTSearch)

        strGridData = "Grid_Data"
        self.Timer.Add(strGridData)
        self.Timer.Start(strGridData)

        grid_dix = [(x, x+1) for x in start_indices]
        #use itertools.product for getting all combinations
        grid_idx_combination = list(itertools.product(*grid_dix))
        
        LUT_value = 0
        for grid in grid_idx_combination:
            grid_value = LUT[grid]
            for i in range(len(start_indices)):
                if grid[i] > start_indices[i]:
                    grid_value *= ratios[i]
                else:
                    grid_value *= (1 - ratios[i])
            LUT_value += grid_value

        self.Timer.Stop(strGridData)

        return LUT_value
    
class AgingFct_Interpolator(Interpolator):
    def __init__(self, strSPList, strLUTField, strLUTList, CellParam):
        super().__init__(strSPList, strLUTField, strLUTList, CellParam)

    def GetLUTSP(self, strSPList = None, strLUTList = None):
        self.LUT = self.dataField.AgingValue.AgingFactor
        self.SP = self.dataField.AgingSetpoint

class RC_Interpolator(Interpolator):
    def __init__(self, strSPList, strLUTField, strLUTList, CellParam):
        super().__init__(strSPList, strLUTField, strLUTList, CellParam)

    def GetLUTSP(self, strSPList = None, strLUTList = None):
        self.LUT = self.dataField.Value
        self.SP = self.dataField.Setpoint

class OCV_Interpolator(Interpolator):
    def __init__(self, strSPList, strLUTField, strLUTList, CellParam):
        super().__init__(strSPList, strLUTField, strLUTList, CellParam)
        self.RefTempC = self.dataField.RefTempC

    def GetLUTSP(self, strSPList, strLUTList = None):
        self.LUT = self.dataField.OCV
        self.SP = Container()
        self.GetSetAttr('SP', self.dataField, strSPList)


class EC_Interpolator(Interpolator):
    def __init__(self, strSPList, strLUTField, strLUTList, CellParam):
        super().__init__(strSPList, strLUTField, strLUTList, CellParam)

    def GetLUTSP(self, strSPList, strLUTList = None):
        self.LUT = self.dataField.EC
        self.SP = Container
        self.GetSetAttr('SP', self.dataField, strSPList)