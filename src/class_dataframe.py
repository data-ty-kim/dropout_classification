import pandas as pd


class PreDataframe:
    def __init__(self, csv_dir, datatype, index_col=0):
        self.df = pd.read_csv(csv_dir, dtype=datatype, index_col=index_col)

    def get_names(self, col_name):
        """
        칼럼명을 넣으면 칼럼에 있는 코드와 이름을 dictionary 로 뱉는 함수
        입력값은 'REC_STS', 'DEPT', 'ADPT', 'DEG', 'ENT'
        """
        col_name = col_name.upper()

        dict_code_1 = {
            'REC': 'REC_STS_CD', 'STS': 'REC_STS_CD',
            'DEPT': 'DEPT_CD', 'ADPT': 'ADPT_CD',
            'DEG': 'DEG_DIV', 'ENT': 'ENT_DIV'
        }

        dict_code_2 = {
            'REC_STS_CD': 'REC_STS_NM', 'DEPT_CD': 'DEPT_NM',
            'ADPT_CD': 'ADPT_NM', 'DEG_DIV': 'DEG_NM', 'ENT_DIV': 'ENT_NM'
        }
        try:
            df_code_name = (
                self.df[[dict_code_1[col_name], dict_code_2[dict_code_1[col_name]]]]
                    .groupby(dict_code_1[col_name])
                    .max()
            )

        except KeyError:
            print("입력값이 잘못되었습니다. 다시 확인해주세요.")

        print("다음과 같은 dataframe 을 반환합니다. (아래는 상위 5개 출력)")
        print(df_code_name.head())
        return df_code_name

    def get_index(self, student_condition):
        """
        :param student_condition: '재학생' or '졸업생'
        :return: '재학생' 또는 '졸업생' 의 index
        """
        if self.df[self.df['구분'] == student_condition].index.empty:
            print("입력을 잘못하셨습니다. '재학생' 또는 '졸업생' 을 입력해주세요.")
        else:
            print(f"{student_condition}의 index 를 반환합니다.")
            return self.df[self.df['구분'] == student_condition].index

    def onehot_ent(self):
        """
        :return: 'ENT_TERM'에서 '1R'은 0으로 '2R'은 1로 변환
        """
        self.df.replace({'ENT_TERM': {'1R': 0, '2R': 1}}, inplace=True)

    def onehot_prof(self):
        """
        :return: 교수 칼럼을 없애고, 제적생이 많은 지도교수, 지도교수가 없는 경우, 일반적인 경우 셋으로 나눈 onehot encoding 진행
        """
        # 지나치게 많은 제적학생을 둔 지도교수 lis 로 저장
        df_outlier = (
            self.df
                .loc[(self.df['REC_STS_CD'] == '401') | (self.df['REC_STS_CD'] == '402'), ['PROF', 'STD_ID']]
                .groupby(['PROF']).count()
        )
        lowerbound = (df_outlier.describe().loc['mean'][0] + 3 * df_outlier.describe().loc['std'][0])
        list_prof = df_outlier.loc[df_outlier['STD_ID'] > lowerbound].index.to_list()
        # df['PROF']에서 결측치 제거하고 one-hot encoding 진행
        self.df['PROF'].fillna(0, inplace=True)
        self.df['PROF'] = self.df['PROF'].astype(int)
        self.df['PROF_0'] = self.df['PROF'].apply(lambda x: 1 if x == 0 else 0)
        self.df['PROF_1'] = self.df['PROF'].apply(lambda x: 1 if (x not in list_prof) & (x != 0) else 0)
        self.df['PROF_2'] = self.df['PROF'].apply(lambda x: 1 if x in list_prof else 0)
        # 원래 쓰던 df['PROF'] 열 삭제
        self.df.drop(columns='PROF', inplace=True)

    def onehot_year(self):
        self.df = pd.get_dummies(self.df, columns=['ENT_YEAR'])
        self.df.drop(columns='ENT_YEAR', inplace=True)

    # 원핫인코딩 칼럼
    # columns=['DEPT_CD', 'ADPT_CD', 'SEC_REG', 'DEG_DIV', 'ENT_DIV']

    def get_target(self, input_type):
        """
        target label 만들기
        :param input_type: 'binary' or 'multi'
        :return: target label
        """
        if input_type == 'binary':
            self.df['target'] = self.df['REC_STS_CD'].apply(lambda x: 1 if (x == '401') | (x == '402') else 0)
            self.df.drop(columns='REC_STS_CD', inplace=True)
        elif input_type == 'multi':
            self.df['target_multi'] = (
                self.df['REC_STS_CD']
                .apply(lambda x: 2 if (x == '401') | (x == '402') else (1 if x in ['303', '304', '501'] else 0))
            )
            self.df.drop(columns='REC_STS_CD', inplace=True)
        else:
            print("입력값을 잘못 입력하였습니다. binary 혹은 multi 라고 입력해주세요.")

    def drop_col(self):
        """
        한글명칭 저장된 칼럼은 모두 삭제
        :return: nothing but act
        """
        self.df.drop(columns=['구분', 'STD_ID', 'REC_STS_NM', 'BIRTH', 'DEPT_NM', 'ADPT_NM', 'SEC_NM', 'DEG_NM',
                              'ENT_NM'], inplace=True)
