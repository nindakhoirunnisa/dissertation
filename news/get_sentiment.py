import pandas as pd
from easynmt import EasyNMT
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class TranslatorAndSentimentAnalyzer:
    def __init__(self):
        self.translator = EasyNMT('opus-mt')
        self.analyzer = SentimentIntensityAnalyzer()

    def translate_and_analyze_sentiment(self, df):
        df['title_en'] = df.apply(lambda x: self.translator.translate(x['full_title'], source_lang='id', target_lang='en'), axis=1)
        df['sentiment'] = df['title_en'].apply(self.get_sentiment)
        return df

    def get_sentiment(self, text):
        vs = self.analyzer.polarity_scores(text)
        score = vs['compound']
        if (score >= 0.05):
            return 1
        elif (score > -0.05 and score < 0.05):
            return 0
        else:
            return -1
