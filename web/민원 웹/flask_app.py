from flask import Flask, render_template, request
import joblib
from kiwipiepy import Kiwi
from gensim.models import Word2Vec
import numpy as np

# Flask app 초기화
app = Flask(__name__)

# 모델 로드
BukGu_model = joblib.load('BukGu_model.joblib')
NamGu_model = joblib.load('NamGu_model.joblib')
DalseongGun_model = joblib.load('DalseongGun_model.joblib')
DalsuGu_model = joblib.load('DalsuGu_model.joblib')
JungGu_model = joblib.load('JungGu_model.joblib')
SeoGu_model = joblib.load('SeoGu_model.joblib')
Suseonggu_model = joblib.load('Suseonggu_model.joblib')
DongGu_model = joblib.load('DongGu_model.joblib')

# w2v 모델 로드
wv_model = joblib.load('new_wv_model.joblib')

# 라우트
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 사용자 입력 받기
    sentence = request.form['sentence']
    text = sentence
    
    subscription = request.form.get('subscription')
    
    # 구별로 다른 모델 사용하기 위한 코드
    if subscription == 'BukGu':
        model = BukGu_model
    elif subscription == 'NamGu':
        model = NamGu_model
    elif subscription == 'DalseongGun':
        model = DalseongGun_model
    elif subscription == 'DalsuGu':
        model = DalsuGu_model
    elif subscription == 'ungGu':
        model = JungGu_model
    elif subscription == 'SeoGu':
        model = SeoGu_model
    elif subscription == 'Suseonggu':
        model = Suseonggu_model
    elif subscription == 'DongGu':
        model = DongGu_model
    else:
        # subscription이 존재하지 않는 경우 기본값인 BukGu로 설정
        model = BukGu_model
        
    # 형태소 분석 후 토큰화
    kiwi = Kiwi()
    sentence = kiwi.tokenize(sentence)
    sentence_token = []
    for s in sentence:
        if s[1] in ['NNG', 'NNP', 'NNB', 'NR', 'NP', 'VV', 'VA', 'VCP', 'VCN']:
            sentence_token.append(s[0])
    
    # 리스트를 numpy 배열로 변환
    sentence_token = np.array(sentence_token).tolist()
    
    def get_sent_embedding(model, embedding_size, tokenized_words):
        feature_vec = np.zeros((embedding_size,), dtype="float32")
        n_words = 0
        for word in tokenized_words:
            if word in model.wv.key_to_index:
                n_words += 1
                feature_vec = np.add(feature_vec, model.wv[word])
        
        # 단어 개수가 0보다 큰 경우 벡터를 단어 개수로 나눠줌 (평균 임베딩 벡터 계산)
        if n_words > 0:
            feature_vec = np.divide(feature_vec, n_words)
        
        return feature_vec
    
    def get_dataset(sentences, model, num_features):
        dataset = list()
        
        # 각 문장을 벡터화해서 리스트에 저장
        for sent in sentences:
            sent_embedding_vectors = get_sent_embedding(model, num_features, sent)
            dataset.append(sent_embedding_vectors)  
              
        # 리스트를 numpy 배열로 변환하여 반환
        sent_embedding_vectors = np.stack(dataset)
        
        return sent_embedding_vectors
    
    sentence_token_vc = get_dataset([sentence_token], wv_model, 1400)
    
    # 모델로 예측하기
    category = model.predict(sentence_token_vc)[0]
    
    # 예측 확률
    pred = model.predict_proba(sentence_token_vc)[0].max()
    # pred = model.predict_proba(sentence_token_vc)[0]
    
    # 결과 result.html 페이지로 보내기
    return render_template('result.html', category=category, text=text, pred=pred, subscription=subscription)


@app.route('/predict/manager')
def manager():
    return render_template('manager.html')

# 앱 실행
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)