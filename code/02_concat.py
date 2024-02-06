import pandas as pd
import glob

data_paths = glob.glob('./data_naver/*')
print(data_paths)

df = pd.DataFrame()

for path in data_paths:
    df_temp = pd.read_csv(path)
    df_temp.columns = ['names', 'reviews']
    df_temp.dropna(inplace=True)
    df = pd.concat([df, df_temp], ignore_index=True)

df.drop_duplicates(inplace=True)
df = df.loc[df.groupby('names')['reviews'].apply(lambda x: x.str.len().idxmax())].reset_index(drop=True)
df = df[df['reviews'].apply(lambda x: len(str(x)) > 5000)]
df.info()
df.to_csv('./reviews_naver.csv', index=False)

