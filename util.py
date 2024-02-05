import numpy as np
import time
class RunTimer:
    def __init__(self, input = None):
        self.RunTimer = {}
        self.StartTime = {}

        if input is not None:
            self.ADD(input)
    
    def setZero(self, strTime):
        if not strTime in self.RunTime:
            self.RunTimer[strTime] = 0


    def Add(self, input):
        if isinstance(input, list):
            for strTime in input:
                self.setZero(strTime)
        else:
            self.setZero(input)

    def Start(self, input):
        self.StartTime[input] = time.time()
        self.Add(input)

    def Check_Started(self, input):
        b_input_not_started = False
        if isinstance(input, list):
            for element in input:
                if (element not in self.StartTime) or (element not in self.RunTimer):
                    b_input_not_started = True
                    raise(f"The {element} in {input} not started")
        else:
            if (input not in self.StartTime) or (input not in self.RunTimer):
                b_input_not_started = True
                raise(f"The {input} not started")
        return b_input_not_started
    
    def Stop(self, input):
        b_input_not_started = self.Check_Started(input)
        end_time = time.time()
        if not b_input_not_started:
            if isinstance(input, list):
                for element in input:
                    self.RunTimer[element] += end_time - self.StartTime[element]
            else:
                self.RunTimer[input] += end_time - self.StartTime[input]

    def Print(self):
        for strTime in self.RunTimer:
            print(f"The time for {strTime} is {self.RunTimer[strTime]}")
    