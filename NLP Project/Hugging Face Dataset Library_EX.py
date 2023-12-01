# 參考網站: 
# 1. https://ithelp.ithome.com.tw/articles/10295522(上一篇程式碼)
# 2. https://ithelp.ithome.com.tw/articles/10296418(衍伸說明)

# 沒有裝 Hugging Face Datasets 的 Library 的話
pip install datasets

# 使用下面的程式碼，可以先來看資料的資訊
# load_dataset_builder 不會把資料下載下來
from datasets import load_dataset_builder
ds_builder = load_dataset_builder("poem_sentiment")

# 使用下面兩欄程式碼，看資料的描述與feature
print(ds_builder.info.description)
print(ds_builder.info.features)

# 覺得資料 OK ，就可以下載下來
from datasets import load_dataset

# ------------------衍伸【載入本地端的 Dataset】說明------------------

# 〈用法一〉載入 Hugging Face Hub 上的 dataset
sentiment = load_dataset("poem_sentiment")

# 〈用法二〉載入 CSV 的 dataset
csv_dataset = load_dataset("csv", data_files="my_dataset.csv")

# 〈用法三〉載入 txt
txt_dataset = load_dataset("text", data_files="my_dataset.txt")

# 〈用法四〉載入 JSON
# 注意這裡用的是 JSON Lines 的格式
json_dataset = load_dataset("json", data_files="my_dataset.jsonl") 

# 〈用法五〉
pandas_dataset = load_dataset("pandas", data_files="my_dataset.pkl")


# ------------------衍伸【載入遠端的 Dataset】說明------------------

# Jupyter notebook 裡下載需加上驚嘆號，反之拿掉
#〈用法一〉
dataset_url = "https://your.dataset/url"
!wget {dataset_url}

#〈用法二〉
url = "https://your.dataset/url"
remote_dataset = load_dataset("csv", data_files=url)

# training dataset 和 test dataset 分開成不同的檔案
#〈用法三〉合併且載入 dataset
url = "https://your.dataset/url"
data_files = {
    "train": url + "train.json.gz",
    "test": url + "json.gz",
}

# 這裡可以省下解壓縮 gz 檔的動作，直接 load 成 dataset，非常的方便實用
remote_dataset = load_dataset("json", data_files=data_files)
