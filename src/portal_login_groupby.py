import pandas as pd

# from portal login to dataframe %% #
df_portal = pd.read_csv('../data/raw/portal_login_log(backdata)_2020_202302.csv'
                        , dtype={'NSSO002_USER_ID': 'int32'}
                        , usecols=['NSSO002_USER_ID', 'NSSO920_ACCESS_DT']
                        # , chunksize=1000000
                        )

df_groupby_pt = df_portal.groupby('NSSO002_USER_ID', as_index=False).count()

df_groupby_pt.to_csv('../data/processed/portal_groupby.csv', sep=',')