import Interpolator
import numpy as np 
import scipy.io
import math
from util import RunTimer
import datetime

class EQC:
    def __init__(self, CellParam, Timer = None):
        """ Read RC, AF, EC, OCV """

        list_strTime = ["EQC_InterpTime", "EQC_BuildLUTime"]

        if Timer is None:
            self.Timer = RunTimer()
        else:
            self.Timer = Timer

        self.Timer.Add(list_strTime)
        self.Timer.Start("EQC_BuildLUTTime")

        # RC
        strSPList_RC = ["SOC", "Current", "Temperature"]
        strRCList = ["R0", "R1", "R2", "R3", "C1", "C2", "C3", "tau1", "tau2", "tau3"]
        strLUTField_RC_CH = "ParameterTableCH"
        strLUTField_RC_DCH = "ParameterTableDCH"

        self.RC_CH = Interpolator.RC_Interpolator(strSPList_RC, strLUTField_RC_CH, strRCList, CellParam)
        self.RC_DCH = Interpolator.RC_Interpolator(strSPList_RC, strLUTField_RC_DCH, strRCList, CellParam)

        # AF
        strSPList_AF = ["SOC", "Current", "Temperature", "SOH"]
        strAFList = ["R0", "R1", "R2", "R3", "C1", "C2", "C3", "tau1", "tau2", "tau3"]
        strLUTField_AF_CH = "ParameterTableAgingCH"
        strLUTField_AF_DCH = "ParameterTableAgingDCH"

        self.AF_CH = Interpolator.AgingFct_Interpolator(strSPList_AF, strLUTField_AF_CH, strAFList, CellParam)
        self.AF_DCH = Interpolator.AgingFct_Interpolator(strSPList_AF, strLUTField_AF_DCH, strAFList, CellParam)

        # EC
        strSPList_EC = ["SOC", "SOH"]
        strLUTField_EC_CH = "SOC_EC_CH"
        strLUTField_EC_DCH = "SOC_EC_DCH"

        self.EC_CH = Interpolator.EC_Interpolator(strSPList_EC, strLUTField_EC_CH, strLUTList = None, CellParam= CellParam)
        self.EC_DCH = Interpolator.EC_Interpolator(strSPList_EC, strLUTField_EC_DCH, strLUTField= None, CellParam= CellParam)

        # OCV
        strSPList_OCV = ["SOC", "SOH"]
        strLUTField_OCV_CH = "SOC_OCV_CH"
        strLUTField_OCV_DCH = "SOC_OCV_DCH"
        self.Interpolation_time = 0
        self.OCV_CH = Interpolator.OCV_Interpolator(strSPList_OCV, strLUTField_OCV_CH, strLUTList = None, CellParam= CellParam)
        self.OCV_DCH = Interpolator.OCV_Interpolator(strSPList_OCV, strLUTField_OCV_DCH, strLUTList= None, CellParam= CellParam)

        self.Timer.Stop("EQC_BuildLUTTime")

    def CalEQC(self, charge_mode, SOC, Icell, Tcell, SOH, VCtrl_use_tau_Lookup = 0):

        self.Timer.Start("EQC_InterpTime")
        if charge_mode>0:
            RC = self.RC_CH
            AF = self.AF_CH
            EC = self.EC_CH

        else:
            RC = self.RC_DCH
            AF = self.AF_DCH
            EC = self.EC_DCH

        OCV_CH = self.OCV_CH
        OCV_DCH = self.OCV_DCH
        strRCList =  ["R0", "R1", "R2", "R3", "C1", "C2", "C3", "tau1", "tau2", "tau3"]
        strAFList =  ["R0", "R1", "R2", "R3", "C1", "C2", "C3", "tau1", "tau2", "tau3"]

        glbVar = {}

        SendTimer = self.Timer
        R0 = RC.GetLUTValue("R0", [SOC, Icell, Tcell], Timer = SendTimer)
        R1 = RC.GetLUTValue("R1", [SOC, Icell, Tcell], Timer = SendTimer)
        R2 = RC.GetLUTValue("R2", [SOC, Icell, Tcell], Timer = SendTimer)
        R3 = RC.GetLUTValue("R3", [SOC, Icell, Tcell], Timer = SendTimer)

        if(SOH == 1):
            AFR0 = 1
            AFR1 = 1
            AFR2 = 1
            AFR3 = 1
            AFC1 = 1
            AFC2 = 1
            AFC3 = 1
        else:
            AFR0 = AF.GetLUTValue("R0", [SOC, Icell, Tcell], Timer = SendTimer)
            AFR1 = AF.GetLUTValue("R1", [SOC, Icell, Tcell], Timer = SendTimer)
            AFR2 = AF.GetLUTValue("R2", [SOC, Icell, Tcell], Timer = SendTimer)
            AFR3 = AF.GetLUTValue("R3", [SOC, Icell, Tcell], Timer = SendTimer)
            AFC1 = AF.GetLUTValue("C1", [SOC, Icell, Tcell], Timer = SendTimer)
            AFC2 = AF.GetLUTValue("C2", [SOC, Icell, Tcell], Timer = SendTimer)
            AFC3 = AF.GetLUTValue("C3", [SOC, Icell, Tcell], Timer = SendTimer)\
            
        R123 = np.array([R1, R2, R3])
        AFR123 = np.array([AFR1, AFR2, AFR3])
        AFC123 = np.array([AFC1, AFC2, AFC3])
        R0 = R0*AFR0
        R123 = R123*AFR123

        if VCtrl_use_tau_Lookup>0:
            tau1 = RC.GetLUTValue("tau1", [SOC, Icell, Tcell], Timer = SendTimer)
            tau2 = RC.GetLUTValue("tau2", [SOC, Icell, Tcell], Timer = SendTimer)
            tau3 = RC.GetLUTValue("tau3", [SOC, Icell, Tcell], Timer = SendTimer)
            tau123 = np.array([tau1, tau2, tau3])

            AFRC123 = AFR123*AFC123
            tau123 = AFRC123*tau123

        else:
            C1 = RC.GetLUTValue("C1", [SOC, Icell, Tcell], Timer = SendTimer)
            C2 = RC.GetLUTValue("C2", [SOC, Icell, Tcell], Timer = SendTimer)
            C3 = RC.GetLUTValue("C3", [SOC, Icell, Tcell], Timer = SendTimer)
            C123 = np.array([C1, C2, C3])
            C123 = C123*AFC123
            tau123 = R123*C123

        ECvalue = EC.GetLUTValue(strTable= None, SPList=[SOC, SOH])
        OCV_CH_Value = OCV_CH.GetLUTValue(strTable = None, SPList = [SOC, SOH])
        OCV_DCH_Value = OCV_DCH.GetLUTValue(strTable= None, SPList=[SOC, SOH])
        R03tau13 = np.concatenate((np.array([R0]), R123, tau123), axis=0)
        self.Timer.Stop("EQC_InterpTime")
        return (R03tau13, ECvalue, OCV_CH_Value, OCV_DCH_Value)
    

class CellModel:
    def __init__(self, filename, SOH = 1, Timer = None):
        mat = scipy.io.loadmat(filename, squeeze_me = True, struct_as_record = False)
        CellParam = mat['CellParam']
        if Timer is None:
            self.Timer = RunTimer()
        else:
            self.Timer = Timer
        self.EQCModel = EQC(CellParam, Timer = Timer)

        self.R_cell_weld = CellParam.R_cell_weld
        self.Q_cell_BOL_Ah = CellParam.Q_cell_BOL_Ah
        self.VCtrl_Cell_Hysteresis = CellParam.VCtrl_Cell_Hysteresis
        self.VCtrl_use_tau_lookup = CellParam.VCtrl_use_tau_lookup
        self.VCtrl_use_current_switch = CellParam.VCtrl_user_current_switch
        self.RefTempC_CH = self.EQCModel.OCV_CH.RefTempC
        self.RefTempC_DCH = self.EQCModel.OCV_DCH.RefTempC
        self.SOC = np.array([])
        self.SOH = SOH
        self.CellV = np.array([])
        self.dV123 = np.array([])
        self.CellOCV = np.array([])
        self.CellQ = np.array([])
        self.dOCVdT = np.array([])
        self.SimTime = np.array([])
        self.dV123_ud = np.array([0,0,0])

        self.interpolation_time = 0

    
    def CellInit(self, SOC0, T0):
        self.SOC = np.append(self.SOC, SOC0)
        self.SimTime = np.append(self.SimTime, 0)
        EQCData = self.EQCModel.CalEQC(charge_mode = 1, SOC = SOC0, Icell = 0, Tcell = T0, SOH = self.SOH)
        OCV_CH = EQCData[2]
        OCV_DCH = EQCData[3]
        OCV = (OCV_CH + OCV_DCH)/2
        EC = EQCData[1]
        self.CellV = np.append(self.CellV, OCV)
        self.CellOCV = np.append(self.CellOCV, OCV)
        self.dOCVdT = np.append(self.dOCVdT, EC)
        self.CellQ = np.append(self.CellQ, 0)
        self.dV123 = np.append(self.dV123, 0)
        
        self.HysteresisUnit = Hysteresis(charge_mode0=1, SOC0 = SOC0)

    
    def SOCCal(self, Icell, dt):
        dSOC = Icell/self.Q_cell_BOL_Ah/self.SOH/3600/dt
        self.SOC = np.append(self.SOC, self.SOC[-1] + dSOC)

    def dVR0Val(self, R0, Icell):
        return Icell*R0
    
    def dV123Cal(self, R123, tau123, Icell, dt):
        expTau123dt = np.exp(-dt/tau123)
        dV123 = R123*Icell*(1- expTau123dt) + self.dV123_ud*expTau123dt
        self.dV123_ud = dV123
        return dV123
    
    def dVTotalCal(self, R03tau123, Icell, dt):
        R0 = R03tau123[0]
        R123 = R03tau123[1:4]
        tau123 = R03tau123[4:7]
        dVR0 = self.dVR0Val(R0 , Icell)
        dV123 = self.dV123Cal(R123, tau123, Icell, dt)
        dV123sum = np.sum(dV123)
        self.dV123 = np.append(self.dV123, dV123sum)
        dVTotal = dV123sum + dVR0
        return dVTotal
    
    def OCVCal(self, charge_mode, Tcell, EC, OCV30_CH, OCV30_DCH):
        self.UpdateHysteresis()
        OCV30 = self.HysteresisUnit.OCV_Hysteresis(OCV30_CH, OCV30_DCH)
        if charge_mode > 0:
            RefTempC = self.RefTempC_CH
        else:
            RefTempC = self.RefTempC_DCH
        CellOCV = (Tcell - RefTempC)*EC + OCV30
        return CellOCV
    
    def CellVCal(self, dVTotal, CellOCV):
        return(dVTotal + CellOCV)
    
    def UpdateHysteresis(self):
        SOC = self.SOC[-1]
        if(self.SOC.size == 1):
            dSOC = 0
        else:
            dSOC = self.SOC[-1]  - self.SOC[-2]
        self.HysteresisUnit.Hysteresis_Update(SOC, dSOC)

    def HeatGenCal(self, Tcell, EC, Icell, dVTotal):
        return ((Tcell+273.15) * EC + dVTotal)*Icell
    
    def UnitIntegration(self, dt, charge_mode, Tcell, Icell):
        EQCData = self.EQCModel.CalEQC(charge_mode, self.SOC[-1], Icell, Tcell, SOH = self.SOH)

        self.interpolation_time = self.EQCModel.Interpolation_time
        R03tau123 = EQCData[0]
        EC = EQCData[1]
        OCV30_CH = EQCData[2]
        OCV30_DCH = EQCData[3]

        self.SOCCal(Icell, dt)
        dVTotal = self.dVTotalCal(R03tau123, Icell, dt)
        CellOCV = self.OCVCal(charge_mode, Tcell, EC, OCV30_CH, OCV30_DCH)
        CellV = self.CellVCal(dVTotal, CellOCV)

        CellQ = self.HeatGenCal(Tcell, EC, Icell, dVTotal)

        self.CellV = np.append(self.CellV, CellV)
        self.CellOCV = np.append(self.CellOCV, CellOCV)
        self.CellQ = np.append(self.CellQ, CellQ)
        self.dOCVdT = np.append(self.dOCVdT, EC)
        self.SimTime = np.append(self.SimTime, self.SimTime[-1] + dt)

    def TimeIntegration(self, StopTime, dt, charge_mode, Tcell, Icell):
        for i in range(1, int(StopTime/dt) + 1):
            self.UnitIntegration(dt, charge_mode, Tcell, Icell)

    
class Hysteresis:
    def __init__(self, charge_mode0, SOC0):
        self.lambda_value = 0.0033
        self.hl = -1
        self.hr = 1
        self.hm_CH = -0.999
        self.hm_DCH = 0.999

        self.Hysteresis_Init(charge_mode0, SOC0)

    def Hysteresis_init(self, charge_mode, SOC0):
        if charge_mode > 0:
            hm = self.hm_CH
        else:
            hm = self.hm_DCH
        hc = self.h1*math.exp(-3600*self.lambda_value*SOC0) + 1
        hc = hc - math.exp(-3600*self.lambda_value*SOC0)

        hd = self.hr*math.exp(-3600*self.lambda_value*(1-SOC0)) - 1
        hd = hd + math.exp(-3600*self.lambda_value*(1-SOC0))    

        self.hs_ud = (hm+1)*(hc-hd)/2+hd
        self.hm = hm

    def Hysteresis_Update(self, SOC, dSOC):
        if(dSOC >= 0):
            X = math.exp(-3600*self.lambda_value*(1-SOC))
            ff = self.hs_ud
            if(X<0.999):
                ff = (self.hr - self.hs_ud*X)/(1-X)

        else:
            X = math.exp(-3600*self.lambda_value*SOC)
            ff = self.hs_ud
            if(X<0.999):
                ff = (self.hl - self.hs_ud*X)/(1-X)
        
        e = math.exp(-3600*self.lambda_value*abs(dSOC))
        hs = e*self.hs_ud + (1-e)*ff

        hc = self.hl*math.exp(-3600*self.lambda_value*SOC) + 1 - math.exp(-3600*self.lambda_value*SOC)
        hd = self.hr*math.exp(-3600*self.lambda_value*(1-SOC)) - 1 + math.exp(-3600*self.lambda_value*(1-SOC))

        hs_update = np.median([hs, hc, hd])

        if(abs(hc-hd) < 0.0001):
            if(hc > 0):
                vv = 1
            else:
                vv = -1
        else:
            vv = 2*(hs - hd)/(hc - hd) - 1
        hm_update = np.median([vv, -1, 1])
        self.hs_ud = hs_update
        self.hm = hm_update

    def OCV_Hysteresis(self, OCV30_CH, OCV30_DCH):
        hm = self.hm
        wc = OCV30_CH*(1+hm)/2
        wd = OCV30_DCH*(1-hm)/2
        return (wc+wd)
    

class CellModel_Gotion_117Ah(CellModel):
    def __init__(self, filename, SOH=1, Timer=None):
        filename = 'CellModel\Gotion100Ah\CellParam.mat'
        super().__init__(filename, SOH, Timer)
            






