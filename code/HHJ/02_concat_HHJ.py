import pandas as pd
import glob

#데이터 폴더 경로 가져오기
path = glob.glob('../data_naver/*')
print(path)

#폴더별 데이터 가져오기
for folder in path:
    folder_name = folder[folder.find('\\')+1:]
    df = pd.DataFrame()
    df_temp = pd.DataFrame()
    names = []
    reviews = []
    file_list = glob.glob(folder+'/*')
    print(file_list)
    for data_path in file_list:
        df_temp = pd.read_csv(data_path)
        names += list(df_temp['names'])
        reviews += list(df_temp['reviews'])

    df['names'] = names
    df['reviews'] = reviews
    df.to_csv(folder+'/'+folder_name+'_concat_data.csv', index=False)
