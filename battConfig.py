import ThermalProperty

class battPack:
    def __init__(self) -> None:
        pass

class Bengal(battPack):
    def __init__(self) -> None:
        super().__init__()
        self.N_cells_parallel = 2
        self.N_cells_serial =126
        self.N_module = 3.5
        self.N_cells = self.N_cells_parallel * self.N_cells_serial

        self.Rpack = 0.9E-3 + 0.02E-3*7
        self.Rcell_CCA = 0.046E-3
        self.RCell_BMS = 5.3E-3/10

        self.CellName = "Gotion_117Ah"
        self.CoolType = "Bottom"
        self.battThermProperty = ThermalProperty.Gotion_117Ah_BottomCool()
        