# %%
import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.common.common_util import path_to_project_root
from src.functions_dataframe import *

# %%
# root directory 지정하기
root_dir = path_to_project_root('dropout_classification')

# 데이터 불러오기
df = pd.read_csv(f'{root_dir}/data/processed/df_dropout_classification.csv',
                 dtype={'STD_ID': 'object', 'REC_STS_CD': 'category',
                        'BIRTH': 'int16', 'AGE': 'int16', 'UNIV_FROM': 'int8', 'DEPT_CD': 'category',
                        'ADPT_CD': 'category', 'SEC_REG': 'category', 'DEG_DIV': 'category',
                        'ENT_DIV': 'category', 'PROF': 'object', 'ENT_TERM': 'category', 'ENT_YEAR': 'int16',
                        'SEQ': 'int16', 'COUNT_CHG': 'int16',
                        'SCHOLARSHIP': 'int64', 'THE_NUMBER_OF_FUNDS': 'int64', 'SUM_OF_FUNDS': 'int64',
                        'THE_NUMBER_OF_WORKS': 'int16', 'GPA': 'float16', 'PORTAL_ACCESS': 'int32', 'BB_ACCESS': 'int32'
                        },
                 index_col=0
                 )
df_lib = pd.read_csv(f'{root_dir}/data/processed/kulib_loan_list.csv',
                     dtype={'user_id': 'object', 'loan_date': 'int32'},
                     index_col=0
                     )
df_lib.rename(columns={'user_id': 'STD_ID'}, inplace=True)

# %%
# 재학생, 졸업생 index 따로 저장하기
index_enrolled = get_index(df, '재학생')
index_graduated = get_index(df, '졸업생')

# %%
# 원-핫 인코딩
df = one_hot_at_once(df, dummy=False)

df['SPENT'] = df.apply(lambda x: (2023 - x['ENT_YEAR']) * 12 if x['ENT_TERM'] == 0 else (2023 - x['ENT_YEAR']) * 12 - 6,
                       axis=1
                       )
df.drop(columns=['ENT_YEAR'], inplace=True)
df = pd.get_dummies(df, columns=['DEPT_CD', 'ADPT_CD', 'SEC_REG', 'DEG_DIV', 'ENT_DIV'])

df_new = df.merge(df_lib, how='left', on='STD_ID')
df_new['loan_date'].fillna(0, inplace=True)
df_new['loan_date'] = df_new['loan_date'].astype('int32')
df_new.rename(columns={'loan_date': 'loan'}, inplace=True)

# %%
# log 변환해주기 'PORTAL_ACCESS', 'BB_ACCESS', 'loan'
df_new['PORTAL_ACCESS'] = np.log1p(df_new['PORTAL_ACCESS'])
df_new['BB_ACCESS'] = np.log1p(df_new['BB_ACCESS'])
df_new['loan'] = np.log1p(df_new['loan'])

# 선형성 있는 칼럼 drop ['THE_NUMBER_OF_FUNDS', 'SUM_OF_FUNDS', 'THE_NUMBER_OF_WORKS']
df_new.drop(columns=['THE_NUMBER_OF_WORKS', 'THE_NUMBER_OF_FUNDS'], inplace=True)

# %%
# 제적:1, 재학/졸업:0으로 target 칼럼 만들기
y_label = get_target(df_new, 'binary')
# train set 만들기
x_features = drop_col(df_new)

# 데이터 세트 분할 70:30
x_train_a, x_test_a, y_train_a, y_test_a = train_test_split(x_features.loc[index_enrolled], y_label.loc[index_enrolled],
                                                            test_size=0.3, random_state=156)
x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(x_features.loc[index_graduated], y_label.loc[index_graduated],
                                                            test_size=0.3, random_state=156)

# train, test 데이터셋 만들기
x_train = pd.concat([x_train_a, x_train_b])
x_test = pd.concat([x_test_a, x_test_b])
y_train = pd.concat([y_train_a, y_train_b])
y_test = pd.concat([y_test_a, y_test_b])

# DMatrix 변환
dtrain = xgb.DMatrix(data=x_train, label=y_train,
                     # enable_categorical=True
                     )
dtest = xgb.DMatrix(data=x_test, label=y_test,
                    # enable_categorical=True
                    )

# 하이퍼 파라미터 설정
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'early_stoppings': 100
}
num_rounds = 400

# 모델 학습
# train dataset은 'train'

# evaluation(=test) dataset은 'eval'
wlist = [(dtrain, 'train'), (dtest, 'eval')]

# 하이퍼 파라미터와 조기종료 파라미터를 train()함수의 파라미터로 전달
xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds,
                      early_stopping_rounds=100, evals=wlist)

# xgboost는 확률만 반환함 예측값 결정은 내가 해야 함
pred_probs = xgb_model.predict(dtest)
print('predict() 수행 결괏값을 10개만 표시, 예측 확률값으로 표시됨')
print(np.round(pred_probs[:10], 3))

# %%
# 예측 확률이 0.5보다 크면 1, 그렇지 않으면 0으로 예측값을 결정해 리스트 객체인 preds에 저장
preds = [1 if x > 0.5 else 0 for x in pred_probs]
print('예측값 10개만 표시:', preds[:10])

# get_clf_eval() 함수를 적용해 성능 평가
def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)

    print('오차 행렬')
    print(confusion)
    print(f'정확도: {accuracy:.4f}, 정밀도: {precision:.4f}')
    print(f'재현율: {recall:.4f}, F1: {f1:.4f}')
    print(f'AUC: {roc_auc:.4f}')


get_clf_eval(y_test, preds, pred_probs)

# %%
# 내장된 시각화 기능
fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(xgb_model, ax=ax)
fig.show()

# %%
# 도서 대출빈도 계산해보면 엄청나게 치우쳐져 있다.

sns.histplot(df_lib['loan_date'], kde=True)
plt.show()

# %%
df_temp = np.log1p(df_lib['loan_date'])
sns.histplot(df_temp, kde=True)
plt.show()

# %%
# 'PORTAL_ACCESS', 'BB_ACCESS'
df_temp = np.log1p(df['PORTAL_ACCESS'])
sns.histplot(df_temp, kde=True)
plt.show()

# %%
# 'PORTAL_ACCESS', 'BB_ACCESS'
df_temp = np.log1p(df['BB_ACCESS'])
sns.histplot(df_temp, kde=True)
plt.show()

# %%
sns.histplot(df['SCHOLARSHIP'], kde=True)
plt.show()


# %%
# 'SCHOLARSHIP', 'SUM_OF_FUNDS'
df_temp = np.log1p(df['SCHOLARSHIP'])
sns.histplot(df_temp, kde=True)
plt.show()

# %%
sns.histplot(df['SUM_OF_FUNDS'], kde=True)
plt.show()


# %%
# 'SCHOLARSHIP', 'SUM_OF_FUNDS'
df_temp = np.log1p(df['SUM_OF_FUNDS'])
sns.histplot(df_temp, kde=True)
plt.show()

# %%
# df.columns
# 'THE_NUMBER_OF_FUNDS', 'SUM_OF_FUNDS', 'THE_NUMBER_OF_WORKS'
df_temp = df[['THE_NUMBER_OF_FUNDS', 'SUM_OF_FUNDS', 'THE_NUMBER_OF_WORKS']]
corr = df_temp.corr(method='pearson')
print(corr)