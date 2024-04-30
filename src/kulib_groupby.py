import pandas as pd

df_1 = pd.read_csv("./data/temp/kulib_library_loan_202307141507.tsv", sep='\t')
df_2 = pd.read_csv("./data/temp/kulib_library_loan_202307141511.tsv", sep='\t')
df_3 = pd.read_csv("./data/temp/kulib_library_loan_202307141514.tsv", sep='\t')
df_4 = pd.read_csv("./data/temp/kulib_library_loan_202307141515.tsv", sep='\t')

df = pd.concat([df_1, df_2, df_3, df_4])
df.sort_values(by='loan_date', inplace=True)
df.to_csv('./data/raw/kulib_loan_list.tsv', sep='\t')

df_groupby = df[['user_id', 'loan_date']].groupby('user_id', as_index=False).count()
df_groupby.to_csv('./data/processed/kulib_loan_list.csv')
