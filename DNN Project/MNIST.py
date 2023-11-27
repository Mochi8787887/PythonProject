# 參考網站: 
# 1. https://ithelp.ithome.com.tw/articles/10325896

# 前置操作: 安裝 matplotlib庫
pip install matplotlib

# 引入函式庫
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 載入MNIST數據集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 數據預處理:

# 展平成一維數組
train_images = train_images.reshape((60000, 28 * 28)) 
test_images = test_images.reshape((10000, 28 * 28))

# 正規化像素值
train_images = train_images.astype('float32') / 255 
test_images = test_images.astype('float32') / 255

# 建立DNN模型
model = Sequential()
model.add(Dense(128, input_shape=(28 * 28,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 訓練模型
# 使用 validation_data 同時驗證損失
history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))  

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

# 評估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
