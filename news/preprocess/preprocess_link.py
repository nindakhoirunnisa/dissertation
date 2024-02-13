import pandas as pd
import numpy as np

class MediaPreprocessor:
    def __init__(self, df):
        self.df = df

    def fill_blank_website(self, index):
        self.df.iloc[index]['Website'] = str(self.df.iloc[index]['Nama Media']).lower()

    def split_link(self):
        split_df = pd.DataFrame(columns=self.df.columns)  # Create an empty DataFrame with the same columns
        
        for _, row in self.df.iterrows():
            name = row['Nama Media']
            typee = row['Jenis Media']
            prov = row['Provinsi']
            website = row['Website']
            status = row['Status']
            date = row['User Status Certificate Date']
            
            # Check for multiple delimiters: ' atau ', ' / ', and '; '
            delimiters = [' atau ', ' / ', '; ']
            split_websites = [website]  # Default to original website if no delimiter found
            
            for delimiter in delimiters:
                if delimiter in website:
                    split_websites = website.split(delimiter)
                    break
            
            # Append each split value as a separate row in split_df
            for split_website in split_websites:
                split_df = split_df.append({'Nama Media': name, 'Jenis Media': typee, 'Provinsi': prov, 'Website': split_website, 'Status': status, 'User Status Certificate Date': date}, ignore_index=True)
        return split_df
    
    def remove_email(self):
        self.df['Website'] = self.df['Website'].str.split(' Email', n=1).str[0]
        # return df
    
    def change_email_to_website(self, index):
        self.df.iloc[index]['Website'] = str(self.df.iloc[index]['Nama Media']).lower()

    def change_address_to_website(self, index):
        self.df.iloc[index]['Website'] = str(self.df.iloc[index]['Nama Media']).lower()
    
    def replace_unnecessary(self):
        self.df['Website'] = self.df['Website'].str.replace('www.', '')
        self.df['Website'] = self.df['Website'].str.replace('http://', '')
        self.df['Website'] = self.df['Website'].str.replace('https://', '')
        self.df['Website'] = self.df['Website'].str.replace('https://www.', '')
        self.df['Website'] = self.df['Website'].str.replace('htpps://', '')
        self.df['Website'] = self.df['Website'].str.replace('Https://', '')
        self.df['Website'] = self.df['Website'].str.replace('http/', '')
        self.df['Website'] = self.df['Website'].str.replace('https//', '')
        self.df['Website'] = self.df['Website'].str.replace('http.//', '')
        self.df['Website'] = self.df['Website'].str.replace('Https/', '')
        self.df['Website'] = self.df['Website'].str.replace('Www//', '')
    
    def remove_postfix(self):
        self.df['Website'] = self.df['Website'].str.split('/', n=1).str[0]
        self.df['Website'] = self.df['Website'].str.lower()
    
    def remove_duplicate_websites(self):
        self.df['Date'] = pd.to_datetime(self.df['User Status Certificate Date'])  # Convert 'Date' column to datetime if not already
        self.df.sort_values(['Website', 'Date'], ascending=[True, False], inplace=True)  # Sort by 'Website' and then 'Date'
        self.df.drop_duplicates(subset='Website', keep='first', inplace=True)  # Keep the first occurrence (most recent date) for each 'Website'
        self.df.drop(columns=['Date'], inplace=True)

class NewsPreprocessor:
    def __init__(self, df):
        self.df = df
    
    def remove_prefix(self):
        self.df['source_link'] = self.df['source_link'].str.replace('https://www.', '')
        self.df['source_link'] = self.df['source_link'].str.replace('https://', '')
        self.df['source_link'] = self.df['source_link'].str.replace('http://', '')
        self.df['source_link'] = self.df['source_link'].str.replace('www.', '')
