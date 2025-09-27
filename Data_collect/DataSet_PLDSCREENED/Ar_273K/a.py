import pandas as pd 
import os 

DF = pd.read_csv("Ar_273K_Ar_273_0.01_to_Ar_273_15_dataset.csv")
FILENAME  = DF["filename"].values
F = "\n".join(FILENAME)

with open("Ar_273_PLD_SCREENED.txt", "w") as f:
    f.write(F)
