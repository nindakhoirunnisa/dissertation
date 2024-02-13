import pandas as pd
from pygooglenews import GoogleNews
import datetime
from datetime import date, timedelta
import pymongo
from preprocess.preprocess_link import NewsPreprocessor
from dissertation.model.predict import PredictLabel

def get_news():
  gn = GoogleNews(lang='id', country='ID')
  data = []

  start_date = date(2020, 3, 1)
  end_date = date(2023, 7, 16)
  date_list = []

  delta = timedelta(days=1)
  current_date = start_date

  while current_date <= end_date:
      date_list.append((current_date.year, current_date.month, current_date.day))
      current_date += delta

  queries = ['vaksin', 'covid AND vaksin', 'sinovac', 'moderna', 'astrazeneca', 'pfizer', 'vaksinasi', 'sinopharm', 'novavax', 'sputnik-v', 'janssen', 'zifivax']

  data = []
  dataset = []

  for q in queries:
    for index in range(0, len(date_list), 2):
      if index+1 < len(date_list):
        start = datetime.date(*date_list[index])
        end = datetime.date(*date_list[index + 1])

        res = gn.search(query=q, from_=start.strftime('%Y-%m-%d'), to_=end.strftime('%Y-%m-%d'))
        entries = res['entries']

        for e in range(0, len(entries)):
          data.append(entries[e])
        
        for i in range(0, len(data)):
          dataset.append({'id': data[i]['id'],'title': data[i]['title_detail']['value'], 'link': data[i]['link'], 'published_at': datetime.datetime.fromtimestamp(time.mktime(data[i]['published_parsed'])), 'source': data[i]['source']['title'], 'source_link': data[i]['source']['href']})
        
  df = pd.DataFrame(dataset)

  keywords = ['covid', 'corona', 'vaksin covid', 'vaksinasi covid', 'vaksin covid-19', 'vaksin covid19', 'astrazeneca', 'vaksinasi', 'sinovac', 'moderna', 'pfizer', 'sinopharm', 'novavax', 'sputnik-v', 'janssen', 'zifivax']
  mask = df['title'].str.contains('|'.join(keywords), case=False)
  
  # Apply the mask to filter the DataFrame
  df = df[mask]

  #Remove prefix in link
  preprocessor = NewsPreprocessor(df)
  preprocessor.remove_prefix()

  client = pymongo.MongoClient('mongodb+srv://nindakhrnns:Mongodb0510@amazone.7o8cfqw.mongodb.net/?retryWrites=true&w=majority')
  db = client['news']
  medias = db['medias']
  all_medias = medias.find()
  data = list(all_medias)
  medias = pd.DataFrame(data)
  medias = medias.rename(columns={'Nama Media': 'name', 'Provinsi': 'province', 'Website': 'source_link', 'Status': 'status', 'User Status Certificate Date': 'date_certified'})
  medias['name'] = medias['source_link']
  medias['date_certified'] = pd.to_datetime(medias['date_certified'])
  df['published_at'] = pd.to_datetime(df['published_at'])
  mapping = {
    'Terverifikasi Administrasi': 1,
    'Terverifikasi Administratif': 1,
    'Terverifikasi Administratif dan Faktual': 2,
    'Terverifikasi Administrasi dan Faktual': 2
  }

  # Apply the value changes using replace()
  medias['status'] = medias['status'].replace(mapping)
  merged_df = pd.merge(medias, df, how='right', left_on='source_link', right_on='source_link')
  merged_df['matching_link'] = merged_df['source_link'].str.extract(f"({'|'.join(medias['source_link'])})")
  merged_df['credibility'] = merged_df['matching_link'].map(medias.set_index('source_link')['status'])
  merged_df['credibility'] = merged_df['credibility'].fillna(0)
  merged_df = merged_df[['id', 'title', 'link', 'published_at', 'province', 'source_link', 'credibility', 'name', 'source']]
  merged_df.loc[merged_df['source_link'].str.contains('.go.id', na=False), 'credibility'] = 2
  merged_df = merged_df[['id', 'title', 'link', 'published_at', 'province', 'source_link',
        'credibility', 'source']].rename(columns={'source': 'source_name'})
  
  #add province from model
  #add detected person
  
if __name__ == '__main__':
  get_news()


  
