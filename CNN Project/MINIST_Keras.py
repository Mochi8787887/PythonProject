# 參考網站: 
# 1. https://ithelp.ithome.com.tw/articles/10326669

# 引入函式庫
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# -----------------------------資料處理-----------------------------

# 載入 Fashion MNIST 數據集
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 數據預處理
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


# -----------------------------建立模型-----------------------------

# 創建卷積神經網絡（CNN）模型
model = Sequential()

# 卷積層 32特徵,卷積核3*3,激勵函數,輸入數據大小(28*28*1) 28*28圖 1為灰階(前面說過1大多黑白3為彩色)
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))) 

#池化 矩陣大小2*2
model.add(MaxPooling2D(pool_size=(2, 2))) 

# Conv2D(特徵數量, 卷積核3*3, 激勵函數,) Conv2D Keras中用於添加二維卷積層的函數
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

#池化 矩陣大小2*2
model.add(MaxPooling2D(pool_size=(2, 2))) 

#攤平
model.add(Flatten()) 

#全連接層
model.add(Dense(128, activation='relu')) 

# Dropout（輸出層隨機失活）是一種正則化技術，用於減少神經網絡的過度擬合
model.add(Dropout(0.5)) 

#全連接層
model.add(Dense(10, activation='softmax')) 


# -----------------------------其他定義與設定-----------------------------

# 編譯模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 定義早停回調函數以監控驗證集的損失
# 訓練模型過程，若在設定次數內，模型沒有再進步則停止訓練，可以避免過擬合          
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# 訓練模型並使用早停回調函數
history = model.fit(x_train, y_train, batch_size=64, epochs=30, validation_split=0.2, callbacks=[early_stopping])

# 在測試集上評估模型
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"測試集上的準確率：{test_accuracy}")

# 輸出最後一個驗證集的損失
val_loss = history.history['val_loss'][-1]
print(f"最後一個驗證集的損失：{val_loss}")


# -----------------------------繪製Loss圖-----------------------------

# 建立 LOSS 圖表

#train loss
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
