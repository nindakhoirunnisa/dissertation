import pandas as pd
import numpy as np
from newspaper import Article

class TitlePreprocessor:
    def get_full_title(self, df):
        try:
            article = Article(df['link'])
            article.download()
            article.html
            article.parse()
            title = article.title
            return title
        except Exception as e:
            return df['title']

    def remove_unnecessary_char(self, title):
        index = title.find(' ... -')
        return title[:index] if index != -1 else title


    
