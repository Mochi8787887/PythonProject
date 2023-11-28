# 參考網站: 
# 1. https://ithelp.ithome.com.tw/articles/10328808

# 引入函式庫
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------設定參數-----------------------------
num_words = 10000      # 使用前10000個常見詞彙
maxlen = 200           # 截斷或填充評論的最大長度
embedding_dim = 64     # 嵌入層的維度
lstm_units = 32        # LSTM隱層的維度

# 載入IMDB數據集並進行數據的預處理
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)


# -----------------------------建立LSTM模型-----------------------------
model = Sequential()

# 嵌入層將整數索引轉換為密集向量表示
model.add(Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=maxlen))

# LSTM層用於處理序列數據
model.add(LSTM(units=lstm_units))

# 全連接層用於二元分類（正面或負面情感）
model.add(Dense(units=1, activation='sigmoid'))

# -----------------------------編譯模型-----------------------------

# 優化器（adam）、損失函數（binary_crossentropy）和準確率（accuracy）
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 打印模型摘要(顯示模型的結構和參數數量)
model.summary()

# 回調函數(避免過擬合)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# -----------------------------訓練模型-----------------------------
# 模型訓練10個epoch，每個batch的大小為128，使用上EarlyStopping回調函數
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2, callbacks=[early_stopping])


# -----------------------------繪製Loss圖和情感詞彙分布圖-----------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
emotions = ['Positive', 'Negative']
num_words = 10000
weights = np.random.rand(num_words, len(emotions))
for i, emotion in enumerate(emotions):
    x = weights[:, i]
    y = np.random.rand(len(x))
    y = y + i * 0.2
    plt.scatter(x, y, label=emotion, alpha=0.5)
plt.legend()
plt.title('Vocabulary Emotion Distribution')
plt.xlabel('Emotion Score')

plt.tight_layout()
plt.show()
