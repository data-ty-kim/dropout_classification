# %% import modules %% #
import pandas as pd

# %% from blackboard to dataframe %% #
df_blackboard = pd.read_csv('../data/raw/tomato_saml_auth_log.csv'
                            , dtype={'user_id': 'int32'}
                            , parse_dates=['event_time']
                            , date_parser=lambda x: pd.to_datetime(x, format='%Y%m%d%H%M%S')
                            # , chunksize=1000000
                            )

df_groupby_bb = df_blackboard.groupby('user_id', as_index=False).count()

df_groupby_bb.to_csv('./blackboard_groupby.csv', sep=',')

# %% from portal login to dataframe %% #
df_portal = pd.read_csv('../data/raw/portal_login_log(backdata)_2020_202302.csv'
                        , dtype={'user_id': 'int32'}
                        , parse_dates=['event_time']
                        , date_parser=lambda x: pd.to_datetime(x, format='%Y%m%d%H%M%S')
                        # , chunksize=1000000
                        )

df_groupby_bb = df_blackboard.groupby('user_id', as_index=False).count()

df_groupby_bb.to_csv('./blackboard_groupby.csv', sep=',')