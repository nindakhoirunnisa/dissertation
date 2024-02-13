from selenium import webdriver
import os
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from seleniumbase import Driver
import pandas as pd
from preprocess.preprocess_link import MediaPreprocessor
import pymongo
import warnings
warnings.filterwarnings("ignore")

driver = Driver(browser='chrome', headless=False)
driver.get('https://datapers.dewanpers.or.id/site/iframe-verified')

def get_credibility_score():
    table = driver.find_element(By.CSS_SELECTOR, "#w4-container > table")
    data = []

    while True:
        current_rows = table.find_elements(By.TAG_NAME, "tr")

        header_row = current_rows[0]
        header_cells = header_row.find_elements(By.TAG_NAME, "th")
        column_names = [cell.text for cell in header_cells]

        next_button = driver.find_element(By.CSS_SELECTOR, '#w4 > div > div.panel-footer > div.kv-panel-pager > ul > li.next')
        next_class = next_button.get_attribute("class")
        
        # Iterate through rows starting from the second row (excluding the header row)
        for row in current_rows[2:]:
            # Initialize an empty dictionary for the row data
            row_data = {}
        
            # Get all cells in the row
            cells = row.find_elements(By.TAG_NAME, "td")
        
            # Extract data from each cell and map it to the corresponding column name
            for i, cell in enumerate(cells):
                column_name = column_names[i]
                row_data[column_name] = cell.text
        
            # Append the row data dictionary to the list
            data.append(row_data)

        if next_class == 'next':
            nxt = driver.find_element(By.CSS_SELECTOR, '#w4 > div > div.panel-footer > div.kv-panel-pager > ul > li.next > a')
            nxt.click()
            WebDriverWait(driver, 10).until(EC.staleness_of(table))
            table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#w4-container table')))
        else:
            break    

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    df = df[['Nama Media', 'Jenis Media', 'Provinsi', 'Website', 'Status', 'User Status Certificate Date']]

    df = df[df['Jenis Media'] == 'Siber']
    df = df.reset_index(drop=True)

    #fill NaN website
    preprocessor = MediaPreprocessor(df)
    for i in df[pd.isna(df['Website'])].index:
        preprocessor.fill_blank_website(i)
    
    #split website
    df = preprocessor.split_link()
    
    #remove email from website
    preprocessor = MediaPreprocessor(df)
    
    for i in a.index:
        preprocessor.fill_blank_website(i)
    preprocessor.remove_email()

    #change email
    for i in df[df['Website'].str.contains('@')].index:
        preprocessor.change_email_to_website(i)

    #remove address
    for i in df[df['Website'].str.contains(' ')].index:
        preprocessor.change_address_to_website(i)
    preprocessor.replace_unnecessary()
    preprocessor.remove_postfix()

    #final pre-processing
    a = df[df['Website'].str.contains(' ')].index
    wnames = ['antaranews.com', 'metrotvnews.com']

    for i in range(len(a)):
        df.iloc[a[i]]['Website'] = wnames[i]
    
    preprocessor.remove_duplicate_websites()

    client = pymongo.MongoClient('mongodb+srv://nindakhrnns:Mongodb0510@amazone.7o8cfqw.mongodb.net/?retryWrites=true&w=majority')
    db = client['news']
    records = df.to_dict('records')

    collection = db['medias']
    collection.insert_many(records)

if __name__ == '__main__':
    get_credibility_score()
