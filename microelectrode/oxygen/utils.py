import pandas as pd
import uncertainties as un

def extract_value_from_region(data, start_s, window_s, se=False):

    end_s = pd.Timedelta(seconds=start_s + window_s)
    start_s = pd.Timedelta(seconds=start_s)

    dt = data.timestamp - data.timestamp

    mask = (dt >= start_s) & (dt < end_s)

    if se:
        return un.ufloat(
            data.loc[mask, 'value'].mean(),
            data.loc[mask, 'value'].std(ddof=1) / (mask.sum() ** 0.5)
        )
    else:
        return data.loc[mask, 'value'].mean()