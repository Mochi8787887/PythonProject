# 參考網站: 
# 1. https://ithelp.ithome.com.tw/articles/10327903

# 引入函式庫
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# -----------------------------設定參數-----------------------------
num_words = 10000      # 使用前10000個常見詞彙
maxlen = 200           # 截斷或填充評論的最大長度
embedding_dim = 64     # 嵌入層的維度
rnn_units = 32         # RNN隱層的維度


# -----------------------------定義模型的結構和訓練過程-----------------------------

# 載入IMDB數據集並進行數據的預處理
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# 使用pad_sequences函數將序列填充或截斷為相同的長度，以確保它們具有相同的維度
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)


# -----------------------------建立RNN模型-----------------------------
model = Sequential()

# 整數索引轉換為密集向量表示
model.add(Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=maxlen)) 

# 處理序列數據
model.add(SimpleRNN(units=rnn_units, activation='relu'))   

# 二元分類（正面或負面情感）
model.add(Dense(units=1, activation='sigmoid'))


# -----------------------------編譯模型-----------------------------

# 優化器（adam）、損失函數（binary_crossentropy）和準確率（accuracy）
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# -----------------------------訓練模型-----------------------------
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2, callbacks=[early_stopping])


# -----------------------------評估模型性能-----------------------------
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"測試準確率: {test_accuracy*100:.2f}%")


# -----------------------------繪製Loss圖-----------------------------

# 建立 LOSS 圖表

# train loss
plt.plot(history.history['loss'])

#test loss
plt.plot(history.history['val_loss'])

#標題
plt.title('Model loss')

#y軸標籤
plt.ylabel('Loss')

#x軸標籤
plt.xlabel('Epoch')

#顯示折線的名稱
plt.legend(['Train', 'Test'], loc='upper left')

#顯示折線圖
plt.show()
