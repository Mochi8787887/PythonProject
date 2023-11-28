# MNIST數據集包含 70000 個 28x28 像素手寫數字圖像，用於深度學習模型的字符識別和圖像分類。
# 參考網站: 
# 1. https://ithelp.ithome.com.tw/articles/10323443
# 2. https://ithelp.ithome.com.tw/articles/10324070


# 引入 NumPy庫並使用 "np" 別名，用於數組和矩陣處理
import numpy as np

# 引入 TensorFlow深度學習框架並使用 "tf" 別名
import tensorflow as tf

# 引入 TensorFlow 的 Keras 模型和層次結構
from tensorflow.keras import layers, models

# 引入手寫數字 MNIST數據集
from tensorflow.keras.datasets import mnist

# 引入將標籤轉換為獨熱編碼的函數
from tensorflow.keras.utils import to_categorical

# 載入MNIST數據集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 數據預處理

# 圖片形狀調整為(樣本數, 圖像長度, 圖像寬度, 圖像通道數)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 歸一化(Normalization)
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# 標籤轉換
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 建立神經網絡模型

# 創建了一個名為 model 的神經網絡模型
model = models.Sequential()

# 將圖像展平成一維數組
model.add(layers.Flatten(input_shape=(28, 28, 1)))  

# 512個神經元的全連接層
model.add(layers.Dense(512, activation='relu'))  

# 10個神經元的全連接層（用於10個類別的分類）
model.add(layers.Dense(10, activation='softmax'))  

# 編譯模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
# 參數對應(丟入訓練集images檔, 丟入訓練集labels檔, 訓練次數, 每次訓練的檔案數）
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# 評估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
