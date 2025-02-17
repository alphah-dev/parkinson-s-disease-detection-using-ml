print("Parkinson's Disease Detection Project Setup Complete!")
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df = pd.read_csv(url)



print(df.info())

print("\nMissing Values:\n", df.isnull().sum())

print("\nClass Distribution:\n", df['status'].value_counts())

