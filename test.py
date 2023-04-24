import pandas as pd 
df = pd.read_csv("dl/sample_text.txt", sep="|")
print(df['label'].unique())

