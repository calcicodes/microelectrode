from .utils import extract_value_from_region
from .constants import O2_solubility, seawater_density
from .io import get_files, read_timeseries_files, read_profile_file

class OxygenData:
    def __init__(self, dat_file):
        self.ts_files, self.pr_file = get_files(dat_file)

        self.timeseries = read_timeseries_files(self.ts_files)
        self.profiles = None
        if self.pr_file:
            self.profiles = read_profile_file(self.pr_file)

        self.timeseries_comments = self.timeseries.loc[self.timeseries.comment != '']

    def get_timeseries_section(self, commment_start, comment_end=None, number_of_sections=1):
        flag = 'air saturation data from here'

        start = self.timeseries_comments.loc[self.timeseries_comments.comment == commment_start].index.item()
        
        if comment_end:
            stop = self.timeseries_comments.loc[self.timeseries_comments.comment == comment_end].index.item()
        else:
            stop = self.timeseries_comments.loc[start:].iloc[number_of_sections].name

        return self.timeseries.loc[start:stop]
    
    def calibrate(self, air_start_flag, zero_start_flag, window_s=30, tempC=22, sal=35):
        air_sat = O2_solubility(sal, tempC) * seawater_density(sal, tempC)  # in umol/L

        sub_air = self.get_timeseries_section(air_start_flag, number_of_sections=1)
        air_reading = extract_value_from_region(sub_air, 0, window_s)

        sub_zero = self.get_timeseries_section(zero_start_flag, number_of_sections=1)
        zero_reading = extract_value_from_region(sub_zero, 0, window_s)
    
        slope = air_sat / (air_reading - zero_reading)
        intercept = - slope * zero_reading

        self.apply_calibration(slope, intercept)

        return slope, intercept
    
    def apply_calibration(self, slope, intercept):
        self.timeseries['o2_umol_L'] = intercept + self.timeseries['value'] * slope

        if self.profiles:
            for p in self.profiles:
                p.data['o2_umol_L'] = intercept + p.data['value'] * slope