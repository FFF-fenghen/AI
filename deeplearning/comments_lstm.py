import shopping_data
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import chinese_vec
import numpy as np

x_train, y_train, x_test, y_test = shopping_data.load_data()

print('x_train:', x_train)
print('y_train:', y_train)
print('x_test:', x_test)
print('y_test:', y_test)

print(x_train[0])
print(y_train[0])

vocalen, word_index = shopping_data.createWordIndex(x_train, x_test)  #
# vocalen 表示词典中的词汇数量
# word_index 表示训练集和测试集全部语料的词典，就是所有的词汇，以及编号
# print(word_index)
print('vocalen:', vocalen)

x_train_index = shopping_data.word2Index(x_train, word_index)  # 获取训练数据的索引表
# x_train_index 表示的是索引向量. 句子x_train被分词产生众多词汇，找到这个句子中所有词汇对应的索引号，组成索引表
x_test_index = shopping_data.word2Index(x_test, word_index)  # 训练测试数据的索引表
# print(x_test_index)

# 统一句子的向量长度
maxlen = 25
x_train_index = sequence.pad_sequences(x_train_index, maxlen=maxlen)
x_test_index = sequence.pad_sequences(x_test_index, maxlen=maxlen)

# 自行构建词嵌入矩阵
word_vec = chinese_vec.load_word_vecs()  # 读取预训练好的词向量
embedding_metrix = np.zeros((vocalen, 300))

for word, i in word_index.items():  # 从自己已有的词汇中去除词汇，取出索引号
    embedding_vector = word_vec.get(word)  # 从训练好的词向量中，去除对应词的词向量
    if embedding_vector is not None:
        embedding_metrix[i] = embedding_vector



# 开始模型训练
model = Sequential()
model.add(Embedding(trainable=False, weights=[embedding_metrix], input_dim=vocalen, output_dim=300, input_length=maxlen))  # input_length 是序列长度
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train_index, y_train, batch_size=512, epochs=50)
score, acc = model.evaluate(x_test_index, y_test)

print(score)
print(acc)
