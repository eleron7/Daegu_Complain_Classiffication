from flask import Flask, render_template, request
import joblib
from kiwipiepy import Kiwi
from gensim.models import Word2Vec
import numpy as np

# Flask app 초기화
app = Flask(__name__)

# 모델 로드
model = joblib.load('model.joblib')

# 라우트
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 사용자 입력 받기
    sentence = request.form['sentence']
    text = sentence
    # 형태소 분석 후 토큰화
    kiwi = Kiwi()
    sentence = kiwi.tokenize(sentence)
    sentence_token = []
    for s in sentence:
        if s[1] in ['NNG', 'NNP', 'NNB', 'NR', 'NP', 'VV', 'VA', 'VCP', 'VCN']:
            sentence_token.append(s[0])
    
    # 리스트를 numpy 배열로 변환
    sentence_token = np.array(sentence_token).tolist()
    
    # 벡터화
    wv_model = Word2Vec(sentences=[sentence_token], vector_size=1000, window=5, min_count=5, workers=4, sg=1)
    
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
            dataset.append(get_sent_embedding(model, num_features, sent))
        
        # 리스트를 numpy 배열로 변환하여 반환
        sent_embedding_vectors = np.stack(dataset)
        
        return sent_embedding_vectors
    
    sentence_token_vc = get_dataset([sentence_token], wv_model, 1000)
    
    # 모델로 예측하기
    category = model.predict(sentence_token_vc)[0]
    
    # 결과 리턴하기
    return render_template('result.html', category=category, text=text)

# 앱 실행
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)