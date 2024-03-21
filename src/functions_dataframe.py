import pandas as pd


def get_names(df: pd.DataFrame, col_name: str):
    """
    칼럼명을 넣으면 칼럼에 있는 코드와 이름을 dictionary 로 뱉는 함수
    입력값은 'REC_STS', 'DEPT', 'ADPT', 'DEG', 'ENT'
    """
    col_name = col_name.upper()

    dict_code_1 = {
        'REC': 'REC_STS_CD', 'STS': 'REC_STS_CD',
        'DEPT': 'DEPT_CD', 'ADPT': 'ADPT_CD', 'SEC': 'SEC_REG',
        'DEG': 'DEG_DIV', 'ENT': 'ENT_DIV'
    }

    dict_code_2 = {
        'REC_STS_CD': 'REC_STS_NM', 'DEPT_CD': 'DEPT_NM',
        'ADPT_CD': 'ADPT_NM', 'SEC_REG': 'SEC_NM', 'DEG_DIV': 'DEG_NM', 'ENT_DIV': 'ENT_NM'
    }

    if col_name not in dict_code_1:
        print("입력값이 잘못되었습니다. 다시 확인해주세요.")

    else:
        df_code_name = (
            df[[dict_code_1[col_name], dict_code_2[dict_code_1[col_name]]]]
                .groupby(dict_code_1[col_name])
                .max()
        )
        print("다음과 같은 dataframe 을 반환합니다. (아래는 상위 5개 출력)")
        print(df_code_name.head())
        return df_code_name


def get_index(df, student_condition):
    """
    :param student_condition: '재학생' or '졸업생'
    :param df: dataframe
    :return: '재학생' 또는 '졸업생' 의 index
    """
    if df[df['구분'] == student_condition].index.empty:
        print("입력을 잘못하셨습니다. '재학생' 또는 '졸업생' 을 입력해주세요.")
    else:
        print(f"{student_condition}의 index 를 반환합니다.")
        return df[df['구분'] == student_condition].index


def one_hot_at_once(df, dummy=True):
    """
    1. 'ENT_TERM'에서 '1R'은 0으로 '2R'은 1로 변환
    2. 교수 칼럼을 없애고, 제적생이 많은 지도교수, 지도교수가 없는 경우, 일반적인 경우 셋으로 나눈 onehot encoding 진행
    3. 'ENT_YEAR', 'DEPT_CD', 'ADPT_CD', 'SEC_REG', 'DEG_DIV', 'ENT_DIV' 바꾸기

    """
    # 1. 학기 바꾸기
    df.replace({'ENT_TERM': {'1R': 0, '2R': 1}}, inplace=True)
    df = df.astype({'ENT_TERM': 'int8'})

    # 2-1. 지나치게 많은 제적학생을 둔 지도교수 list 로 저장
    df_outlier = (
        df
        .loc[(df['REC_STS_CD'] == '401') | (df['REC_STS_CD'] == '402'), ['PROF', 'STD_ID']]
        .groupby(['PROF']).count()
    )
    lowerbound = (df_outlier.describe().loc['mean'][0] + 3 * df_outlier.describe().loc['std'][0])
    list_prof = df_outlier.loc[df_outlier['STD_ID'] > lowerbound].index.to_list()
    # 2-2. df['PROF']에서 결측치 제거하고 one-hot encoding 진행
    df['PROF'].fillna(0, inplace=True)
    df['PROF'] = df['PROF'].astype(int)
    df['PROF_0'] = df['PROF'].apply(lambda x: 1 if x == 0 else 0)
    df['PROF_1'] = df['PROF'].apply(lambda x: 1 if (x not in list_prof) & (x != 0) else 0)
    df['PROF_2'] = df['PROF'].apply(lambda x: 1 if x in list_prof else 0)

    # 3. 연도, 학과명, 단과대, 계열, 학위과정, 입학전형 바꾸기
    if dummy is True:
        return pd.get_dummies(df, columns=['ENT_YEAR', 'DEPT_CD', 'ADPT_CD', 'SEC_REG', 'DEG_DIV', 'ENT_DIV'])

    else:
        return df

def get_target(df, input_type):
    """
    target label 만들기
    :param input_type: 'binary' or 'multi'
    :param df: dataframe
    :return: target label
    """
    if input_type == 'binary':
        df['target'] = df['REC_STS_CD'].apply(lambda x: 1 if (x == '401') | (x == '402') else 0)
        df.drop(columns='REC_STS_CD', inplace=True)
        return df['target']
    elif input_type == 'multi':
        df['target'] = (
            df['REC_STS_CD']
            .apply(lambda x: 2 if (x == '401') | (x == '402') else (1 if x in ['303', '304', '501'] else 0))
        )
        df.drop(columns='REC_STS_CD', inplace=True)
        return df['target']
    else:
        print("입력값을 잘못 입력하였습니다. binary 혹은 multi 라고 입력해주세요.")


def drop_col(df):
    """
    한글명칭 저장된 칼럼은 모두 삭제
    :return: dataframe
    """
    df.drop(labels=['구분', 'STD_ID', 'REC_STS_NM', 'BIRTH', 'DEPT_NM', 'ADPT_NM', 'SEC_NM', 'DEG_NM', 'ENT_NM',
                    'PROF', 'target'],
            axis=1, inplace=True
            )
    return df
