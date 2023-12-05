# 採用python中scikit-learn最著名的iris鳶尾花資料集進行預測
# 參考網站: 
# 1. https://ithelp.ithome.com.tw/articles/10314722

# 引入scikit-learn中的datasets
from sklearn import datasets

import pandas as pd

# 將鳶尾花的資料給讀出來
iris = datasets.load_iris()

# 利用pandas套件將資料轉換成pandas中的dataframe資料型態
features = pd.DataFrame(data = iris.data, columns = iris.feature_names)
labels = pd.DataFrame(data = iris.target)

# pandas中的display函式來展示dataframe
display(features)
display(labels)

# ---------------------------------------------------

from sklearn.model_selection import train_test_split

# 將資料分為features以及labels
# 利用scikit-learn中的train_test_split函式將訓練集以及測試集給分開
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.33, random_state = 42)

# ---------------------------------------------------

from sklearn.neural_network import MLPClassifier

# 宣告一個MLPClassifier物件(目前這只代表一個空的模型)
clf = MLPClassifier()

# 給訓練集feature以及label，讓模型學習應該如何去做判斷
clf.fit(x_train, y_train)

# ----------------------------------【方法一】----------------------------------

# 定義一個函式accuracy
def accuracy(predict, real):
    n = 0
    for i in range(len(predict)):
        if (predict[i] == real[i]):
            n += 1
    
    return n / len(predict)


# 利用predict函式先得出對於x_test的預測結果
result = clf.predict(x_test)

# 將結果跟正確答案y_test丟進剛剛定義的accuracy函式
print(f"Accuracy: {accuracy(result, y_test[0].values) * 100} %")

# ----------------------------------【方法二】----------------------------------

# scikit-learn有提供函式來做計算正確率的功能
from sklearn.metrics import accuracy_score

result = clf.predict(x_test)

# 直接將預測結果和正確結果丟進accuracy_score函式中
print(f"Accuract: {accuracy_score(result, y_test) * 100} %")
