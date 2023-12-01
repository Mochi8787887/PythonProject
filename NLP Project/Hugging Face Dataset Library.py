# 參考網站: 
# 1. https://ithelp.ithome.com.tw/articles/10295522

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
sentiment = load_dataset("poem_sentiment")

# 查看資料長相
sentiment

# 切割資料
train_ds = sentiment["train"]
valid_ds = sentiment["validation"]
test_ds = sentiment["test"]

# 其他切割方式
# dataset_train = load_dataset("rotten_tomatoes", split="train")

# 把 dataset 轉成 Pandas
import pandas as pd

sentiment.set_format(type="pandas")

df = sentiment["train"][:]

df.head(10)

# 使用int2str 來把 label 轉成文字
def label_int2str(row):
	return sentiment["train"].features["label"].int2str(row)

df["label_name"] = df["label"].apply(label_int2str)
df.head(10)

# 查看 dataset 的 label 分佈
import matplotlib.pyplot as plt

df["label_name"].value_counts().plot.barh()
plt.title("Poem Classes")
plt.show()

# 不要pandas 格式
sentiment.reset_format()

# 可以把 pandas 處理過的轉成新的 dataset
from datasets import Dataset

label_name_dataset = Dataset.from_pandas(df)
label_name_dataset

# shuffle 資料
sentiment_train = sentiment["train"].shuffle(seed=5566).select(range(100))

# filter過濾資料
sentiment_filtered = sentiment.filter(lambda x: len(x["verse_text"]) > 30)
sentiment_filtered

new_dataset = sentiment.map(
    lambda x: {"verse_text": [ len(o) for o in x["verse_text"] ] }, batched=True
)
new_dataset['test'][:3]
