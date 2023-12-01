# 參考網站: 
# 1. https://ithelp.ithome.com.tw/articles/10292946

 # Jupyter Notebook 
!pip install transformers

from transformers import pipeline

#使用情感分析
classifier = pipeline("sentiment-analysis") 

# 內容可以使用英文 or 中文測試看看
classifier(
    [
        "寶寶覺得苦，但寶寶不說",
        "我愛寶寶"
    ]
)
