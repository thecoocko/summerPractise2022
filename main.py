import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import modules.regression as mreg
import modules.neuron_network as nn
import modules.evolution_algorithm as ea

df = pd.read_csv('../Foreign_Exchange_Rates.csv', parse_dates=True, index_col=0)
print(f"DATASET SHAPE: {df.shape}")




for i in range(1,len(df.columns)):
    df[df.columns[i]] = df[df.columns[i]].replace('ND',0).astype('float')
    
print(df.dtypes)

mreg.hi('mreg')
nn.hi('nn')
ea.hi('ea')