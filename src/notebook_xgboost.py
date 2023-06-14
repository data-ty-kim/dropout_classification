# %%
import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from matplotlib.pyplot as plt

# %%
dataset = load_breast_cancer()
x_features = dataset.data # 30개가 넘는다.
y_label = dataset.target
cancer_df = pd.DataFrame(data=x_features, columns=dataset.feature_names)
cancer_df['target'] = y_label

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(cancer_df.head(3))

# %%
# Target 이름와 분포 확인
print(dataset.target_names)
print(cancer_df['target'].value_counts())

# %%
# 데이터 세트 분할 80:20
x_train, x_test, y_train, y_test = train_test_split(x_features, y_label, test_size=0.2, random_state=156)
print(x_train.shape, x_test.shape)

# %%
# DMatrix 변환
dtrain = xgb.DMatrix(data=x_train, label=y_train)
dtest = xgb.DMatrix(data=x_test, label=y_test)

# %%
# 하이퍼 파라미터 설정
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'early_stoppings': 100
}
num_rounds = 400

# %%
# 모델 학습
# train dataset은 'train'

# evaluation(=test) dataset은 'eval'
wlist = [(dtrain, 'train'), (dtest, 'eval')]

# 하이퍼 파라미터와 조기종료 파라미터를 train()함수의 파라미터로 전달
xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds,
                      early_stopping_rounds=100, evals=wlist)

# %%
# xgboost는 확률만 반환함 예측값 결정은 내가 해야 함
pred_probs = xgb_model.predict(dtest)
print('predict() 수행 결괏값을 10개만 표시, 예측 확률값으로 표시됨')
print(np.round(pred_probs[:10], 3))

# 예측 확률이 0.5보다 크면 1, 그렇지 않으면 0으로 예측값을 결정해 리스트 객체인 preds에 저장
preds = [1 if x>0.5 else 0 for x in pred_probs]
print('예측값 10개만 표시:', preds[:10])

# %%
# get_clf_eval() 함수를 적용해 성능 평가
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    print('오차 행렬')
    print(confusion)
    print(f'정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}')

get_clf_eval(y_test, preds)

# %%
# 내장된 시각화 기능
fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(xgb_model, ax=ax)
