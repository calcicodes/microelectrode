import pandas as pd
import numpy as np
from datetime import timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.optimize import curve_fit

import uncertainties as un
import uncertainties.unumpy as unp

def calibration_fn(X, *coeffs):
    log_conc, hrs_from_start = X
    
    # mV = X
    
    A1_0, A2_0, x0_0, dx_0, A1_m, A2_m, x0_m, dx_m  = coeffs
    
    A1 = A1_0 + A1_m * hrs_from_start
    A2 = A2_0 + A2_m * hrs_from_start
    x0 = x0_0 + x0_m * hrs_from_start
    dx = dx_0 + dx_m * hrs_from_start
    
    return A2 + (A1 - A2) / (1 + np.exp((log_conc - x0) / dx))

def u_calibration_fn(X, *coeffs):
    log_conc, hrs_from_start = X
    
    # mV = X
    
    A1_0, A2_0, x0_0, dx_0, A1_m, A2_m, x0_m, dx_m  = coeffs
    
    A1 = A1_0 + A1_m * hrs_from_start
    A2 = A2_0 + A2_m * hrs_from_start
    x0 = x0_0 + x0_m * hrs_from_start
    dx = dx_0 + dx_m * hrs_from_start
    
    return A2 + (A1 - A2) / (1 + unp.exp((log_conc - x0) / dx))

def pred_fn(X, *coeffs):
    mV, hrs_from_start = X
    
    A1_0, A2_0, x0_0, dx_0, A1_m, A2_m, x0_m, dx_m  = coeffs
    
    A1 = A1_0 + A1_m * hrs_from_start
    A2 = A2_0 + A2_m * hrs_from_start
    x0 = x0_0 + x0_m * hrs_from_start
    dx = dx_0 + dx_m * hrs_from_start
    
    # remove impossible values
    remove = (A1 - A2) / (mV - A2) - 1 < 0
    mV[remove] = np.nan
    
    return x0 + dx * unp.log((A1 - A2) / (mV - A2) - 1)

class MicroElectrode:
    def __init__(self, analyte, calibration_data_file, calibration_set_gap_hrs=1, min_conc=None):
        self.calibration_data_file = calibration_data_file
        self.calibration_set_gap_hrs = calibration_set_gap_hrs
        self.min_conc = min_conc
        self.analyte = analyte
        self.log_analyte = f'log_{self.analyte}'
        
        self.calibrate()
        
    def calibrate(self):
        cal = pd.read_csv(self.calibration_data_file)
        
        cal[self.log_analyte] = -np.log10(cal[self.analyte])
        
        cal['datetime'] = pd.to_datetime(cal.datetime, format='%d/%m/%Y %H:%M:%S')
        cal['calibration_set'] = (cal.datetime.diff() > timedelta(hours=self.calibration_set_gap_hrs)).cumsum()
        
        if self.min_conc is not None:
            cal = cal.loc[cal[self.analyte] >= self.min_conc]
    
        self.cal = cal
        
        self.session_start = cal.datetime.min()
        self.cal['hrs_from_start'] = (cal.datetime - self.session_start).dt.total_seconds() / 60 / 60
        
        X = (self.cal[self.log_analyte], cal['hrs_from_start'])
        p, cov = curve_fit(calibration_fn, X, cal['mV'], p0=[self.cal[self.log_analyte].max(), self.cal[self.log_analyte].min(), 0, 1, 0, 0, 0, 0])

        self.calibration_params = un.correlated_values(p, cov)
            
    def set_max_conc(self, max_conc):
        self.max_conc = max_conc
        self.calibrate()
    
    def plot_calibration(self):
        fig, ax = plt.subplots(constrained_layout=True)

        for s, g in self.cal.groupby('calibration_set'):
            ax.scatter(g[self.log_analyte], g['mV'])
            
        new_conc = np.linspace(self.cal[self.log_analyte].min(), self.cal[self.log_analyte].max(), 100)
        # new_mV = self.cal['mV'].values
        upred_start = u_calibration_fn((new_conc, 0), *self.calibration_params)
        ax.plot(new_conc, unp.nominal_values(upred_start), 'C0')
        ax.fill_between(new_conc, 
                        unp.nominal_values(upred_start) - unp.std_devs(upred_start), 
                        unp.nominal_values(upred_start) + unp.std_devs(upred_start), alpha=0.3, color='C0')
        
        upred_end = u_calibration_fn((new_conc, self.cal['hrs_from_start'].max()), *self.calibration_params)
        ax.plot(new_conc, unp.nominal_values(upred_end), 'C1')
        ax.fill_between(new_conc,
                        unp.nominal_values(upred_end) - unp.std_devs(upred_end), 
                        unp.nominal_values(upred_end) + unp.std_devs(upred_end), 
                        alpha=0.3, color='C1')
        
        ax.set_xlabel(self.log_analyte)
        ax.set_ylabel('mV')
        
        return fig, ax
    
    def calculate(self, data_file, time_correct=True):
        self.data_file = data_file
        
        data = pd.read_csv('test_data/ammonium.csv')
        data['datetime'] = pd.to_datetime(self.session_start.strftime('%d/%m/%Y') + ' ' + data.time, format='%d/%m/%Y %H:%M:%S')

        data['hrs_from_start'] = (data.datetime - self.session_start).dt.total_seconds() / 60 / 60
        
        if time_correct:
            X = (data.mV.values, data.hrs_from_start.values)
        else:
            X = (data.mV.values, self.cal['hrs_from_start'].mean())
        
        data[self.log_analyte] = pred_fn(X, *self.calibration_params)
        data[self.analyte] = 10**(-data[self.log_analyte])
        
        self.data = data
        
        return data
    
    def save(self, filename=None):
        data_unpacked = self.data.copy()
        for c, d in self.data.items():
            if np.any(unp.std_devs(d) > 0):
                data_unpacked[f'{c}_std'] = unp.std_devs(d)
                data_unpacked[c] = unp.nominal_values(d)
        if filename is None:
            filename = self.data_file.replace('.csv', '_calibrated.csv')
        
        data_unpacked.to_csv(filename, index=False)


    def plot_calculated(self):
        fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(self.data.datetime, unp.nominal_values(self.data[self.analyte]))

        ax.fill_between(self.data.datetime, 
                        unp.nominal_values(self.data[self.analyte]) - unp.std_devs(self.data[self.analyte]), 
                        unp.nominal_values(self.data[self.analyte]) + unp.std_devs(self.data[self.analyte]), alpha=0.3)

        ax.set_ylabel(self.analyte)
        ax.set_xlabel('Time')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        return fig, ax