# Daegu City Civil complaint Category Classifier Using Language Intelligence Artificial Intelligence Model
<img src="https://img.shields.io/badge/flask-000000?style=for-the-badge&logo=flask&logoColor=white"><img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"><img src="https://img.shields.io/badge/html5-E34F26?style=for-the-badge&logo=html5&logoColor=white"><img src="https://img.shields.io/badge/css-1572B6?style=for-the-badge&logo=css3&logoColor=white"><img src="https://img.shields.io/badge/javascript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black">


<img src="https://img.shields.io/badge/scikitlearn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=Tensorflow&logoColor=white"><img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white">

## 민원 내용을 판단, 부서를 분류하는 AI 모델 개발

### 회의록
https://www.notion.so/b5c03d4474544d2588ef0012033cb9a7?v=e89cdbe2f9204f0b98d737345e070bef

### 데이터 수집
<a href="http://dongjak.eminwon.seoul.kr/emwp/gov/mogaha/ntis/web/emwp/cmmpotal/action/EmwpMainMgtAction.do">새올 전자 민원 창구</a> 
- 셀레니움을 사용하여 대구광역시 모든 자치구들의 민원 데이터를 수집

### 전처리
- 결측치, 불용어 제거(이모티콘, 한자, \n 등), 맞춤법 검사, 토큰화 
- 읍, 면, 동 이관 민원 -> 동사무소 이관으로 통합
- 부서명의 변화를 고려해 현재 편제에 맞도록 부서명 변경
- 민원건수가 매우 적은 부서들은 클래스에서 제외

### 모델링
- word2vec 모델사용, skip gram 방식 문장 임베딩
- pycaret을 사용하여 가장 성능이 높은 LDA 를 분류 모델로 선정

### 웹서비스 구현
- 모델의 작동 방식, 분류 성능 확인


![북구_접수 1](https://user-images.githubusercontent.com/80496813/236376991-50cefb77-cc45-41e6-8dac-3301af2b0f09.png)
![북구_접수2](https://user-images.githubusercontent.com/80496813/236377000-4970d349-38e8-4420-afeb-ddcfd1b9f34a.png)
![북구_result_1](https://user-images.githubusercontent.com/80496813/236376529-f4f5d139-1f6d-4a27-9a89-d828d00708c1.png)
![북구_result_2](https://user-images.githubusercontent.com/80496813/236376978-2fed0ac3-636f-42f0-bb71-726c5cfe5133.png)
![북구_manager](https://user-images.githubusercontent.com/80496813/236377011-c8d853ee-67d8-4127-bff3-c5584968df0c.png)



