import requests
import xmltodict
import time

# 기준일 설정
date = "2020-11-01"

# 프로그램 작동 시간 측정
start_time_check = time.time()

# API 접속하여 xml 받기
xml_list = []

url = 'http://apis.data.go.kr/1352000/ODMS_COVID_12/callCovid12Api'
params = {'serviceKey': '',
          'pageNo': '1', 'numOfRows': '500', 'std_day': date, 'apiType': 'xml'}

response = requests.get(url, params=params)
req = response.content
xmlObject = xmltodict.parse(req)

print(xmlObject)
