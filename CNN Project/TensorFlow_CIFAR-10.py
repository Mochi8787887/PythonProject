# 利用TensorFlow + 資料集—CIFAR-10 做CNN實作
# 參考網站: 
# 1. https://ithelp.ithome.com.tw/articles/10315087
# 2. https://www.tensorflow.org/tutorials/images/cnn

# 從TensorFlow中引入CIFAR-10資料集
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 讓像素的數值介於0~1之間(將train_images以及test_images除以255.0)
train_images, test_images = train_images / 255.0, test_images / 255.0

# ---------------------------------------------------

# 利用python的matplotlib.pyplot檢視一下資料集中的內容
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

plt.figure(figsize = (10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
    
plt.show()


# ----------------------------------【建立CNN】----------------------------------】----------------------------------
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = "relu"))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(10))

model.summary()

# ---------------------------------------------------

# 將訓練資料給丟進去做訓練，並且指定test_images為驗證
model.compile(optimizer = "adam", 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
              metrics = ["accuracy"])

history = model.fit(train_images, train_labels, epochs = 10, 
                    validation_data = (test_images, test_labels))
                    
# ---------------------------------------------------
               
# 利用python的matplotlib.pyplot繪製隨著訓練的epoch提升，準確度如何變化的2D圖               
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# ---------------------------------------------------

# 實際印出測試準確度
print(test_acc)
