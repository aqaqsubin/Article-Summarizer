## **워드 임베딩(Word Embedding)**
 단어를 벡터로 표현하는 방법으로, 단어를 밀집 표현으로 변환한다. 이 밀집 벡터를 임베딩 벡터라고도 한다.    

 워드 임베딩 방법론으로는 LSA, [Word2Vec](./Word2Vec.md), FastText, GloVe 등이 있다. 
 
 케라스의 ```Embedding()``` 이라는 도구는 이러한 방법론을 사용하지는 않지만, 단어를 밀집 벡터로 변환한 후에 벡터 값을 학습하는 방법을 사용한다.

위키피디어 등과 같은 방대한 corpus를 가지고 ```Word2vec```, ```FastText```, ```GloVe``` 등을 통해서 사전 훈련된 임베딩 벡터를 불러오는 방법을 사용할 수도 있다.

이 문서에서는 사전 훈련된 워드 임베딩을 가져와 사용하는 방법과 케라스의 `Embedding()` 사용하는 방법을 알아본다.

<br>

- 케라스의 임베딩 층 (Keras Embedding Layer)  
- 사전 훈련된 워드 임베딩(Pre-trained word embedding)

<br>

### **Keras Embedding Layer**  

임베딩 층의 입력으로 사용하기 위해서는 입력 시퀀스의 각 단어들은 정수 인코딩이 되어 있어야 한다.

임베딩 층은 룩업 테이블(Lookup Table)의 개념이며, 어떤 단어 `x`에 매핑된 정수 `n`값에 따라 테이블로부터 인덱스 `n`에 위치한 임베딩 벡터를 꺼내온다.

이 임베딩 벡터는 모델의 입력이 되고, 역전파 과정에서 단어의 임베딩 벡터값이 학습된다.  


```
model = Sequential()
model.add(Embedding(vocab_size, 4, input_length=max_len)) 
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
```
<br>

### **Pre-Trained Word Embedding**  

이미 훈련되어져 있는 워드 임베딩을 불러서 임베딩 벡터로 사용하는 방법이다.  
훈련 데이터가 적은 상황일 경우, 다른 텍스트 데이터를 Word2Vec, GloVe 등으로 사전 훈련된 임베딩 벡터를 불러오는 것이 성능의 개선을 가져올 수 있다.

<br>

#### **사전 훈련된 GloVe 사용**

아래 코드를 통해 glove.6B.zip 파일을 다운받고, 압축 해제하여 텍스트 파일을 읽는다.


```
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove*.zip
```

```
['the', '-0.038194', '-0.24487', '0.72812', ... 중략... '0.8278', '0.27062']
[',', '-0.10767', '0.11053', '0.59812', ... 중략 ... '0.45293', '0.082577']
```
텍스트 파일 안에는 리스트들이 있으며, 단어와 임베딩 벡터를 제공한다.    

이를 기반으로 `embedding_matrix`를 구성하여 가중치 입력으로 사용한다.

사전 훈련된 임베딩 벡터가 100차원의 값이므로 임베딩 층의 `output_dim`은 100이어야 한다.

사전 훈련된 워드 임베딩을 그대로 사용할 것이므로, 더 이상 훈련을 하지 않기 위해 `trainable=False` 옵션을 선택한다.

```
model = Sequential()
model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_len, trainable=False))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=100, verbose=2)
```

<br>


#### **사전 훈련된 Word2Vec 사용**

구글의 사전 훈련된 Word2Vec을 다운받고, Word2Vec 모델을 로드한다.

```
!wget "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
```

```
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True) 
```

`word2vec_model`의 `get_vector(word)`를 통해 해당 단어의 임베딩 벡터를 구할 수 있으며, 
 이를 기반으로 `embedding_matrix`를 구성하여 가중치 입력으로 사용한다.

사전 훈련된 임베딩 벡터가 300 차원의 값이므로 임베딩 층의 `output_dim`은 300 이다.

마찬가지로 사전 훈련된 워드 임베딩을 그대로 사용할 것이기 때문에 `trainable=False` 옵션을 선택한다.

```
model = Sequential()
model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_len, trainable=False))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=100, verbose=2)
```

