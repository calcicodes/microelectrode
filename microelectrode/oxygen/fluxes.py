import numpy as np

def fit_linear_region(o2_profile, yvar='o2_umol_L', rdiff_threshold=0.05):
    pts = 3
    rdiff = 0
    
    while rdiff < rdiff_threshold:
        sub = o2_profile.iloc[-pts:]
        psub = o2_profile.iloc[-(pts+1)]

        p = np.polyfit(sub['position'], sub[yvar], 1)
        pred = np.polyval(p, psub['position'])
        rdiff = abs((pred - psub[yvar]) / psub[yvar])
        pts += 1

    return p, pts - 1