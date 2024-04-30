# %%
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score
from joblib import dump
from src.common.common_util import path_to_project_root
from src.functions_dataframe import *
import shap

# root directory 지정하기
root_dir = path_to_project_root('dropout_classification')

# 데이터 불러오기
df = pd.read_csv(f'{root_dir}/data/processed/df_final.csv',
                 dtype={'STD_ID': 'object', 'REC_STS_CD': 'category',
                        'BIRTH': 'int16', 'AGE': 'int16', 'UNIV_FROM': 'int8', 'DEPT_CD': 'category',
                        'ADPT_CD': 'category', 'SEC_REG': 'category', 'DEG_DIV': 'category',
                        'ENT_DIV': 'category', 'PROF': 'object', 'ENT_TERM': 'category', 'ENT_YEAR': 'int16',
                        'SEQ': 'int16', 'COUNT_CHG': 'int16', 'loan': 'int32',
                        'SCHOLARSHIP': 'int64', 'THE_NUMBER_OF_FUNDS': 'int64', 'SUM_OF_FUNDS': 'int64',
                        'THE_NUMBER_OF_WORKS': 'int16', 'GPA': 'float16', 'PORTAL_ACCESS': 'int32', 'BB_ACCESS': 'int32'
                        },
                 index_col=0
                 )

# log 변환해주기 'PORTAL_ACCESS', 'BB_ACCESS', 'loan'
df['PORTAL_ACCESS'] = np.log1p(df['PORTAL_ACCESS'])
df['BB_ACCESS'] = np.log1p(df['BB_ACCESS'])
df['loan'] = np.log1p(df['loan'])

# 원-핫 인코딩
df = one_hot_at_once(df, dummy=True)

# 재학생, 졸업생 index 따로 저장하기
index_enrolled = get_index(df, '재학생')
index_graduated = get_index(df, '졸업생')

# 제적:1, 재학/졸업:0으로 target 칼럼 만들기
y_label = get_target(df, 'binary')
# train set 만들기
x_features = drop_col(df)

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

# %%
# XGBoost 최종 모델 훈련과 예측 수행
xgb_params = {
    'colsample_bytree': 1,
    'eval_metric': 'logloss',
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_child_weight': 3,
    'objective': 'binary:logistic',
    'early_stoppings': 100
}
xgb_wrapper = XGBClassifier(n_estimators=400)
xgb_wrapper.fit(x_train, y_train, verbose=True)
w_preds = xgb_wrapper.predict(x_test)
w_pred_probs = xgb_wrapper.predict_proba(x_test)[:, 1]

# %%
# 예측 확률이 0.42보다 크면 1, 그렇지 않으면 0으로 예측값을 결정해 리스트 객체인 preds에 저장
preds = [1 if x > 0.42 else 0 for x in w_pred_probs]
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


get_clf_eval(y_test, preds, w_pred_probs)

# %%

explainer = shap.TreeExplainer(xgb_wrapper)
shap_values = explainer.shap_values(x_train)

# %%
shap.summary_plot(shap_values, x_train, plot_type='bar')

# %%
shap.summary_plot(shap_values, x_train)
