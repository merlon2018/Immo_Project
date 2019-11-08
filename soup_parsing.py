from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import re
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import ElementClickInterceptedException
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import sqlite3
import os
import get_user_agent
from selenium.webdriver.chrome.options import Options


def initialize_browser(headless_state):
    opts = Options()
    opts.headless = headless_state           #set to True to navigate headless
    user_agent = get_user_agent.get_agent()
    opts.add_argument("user-agent=user_agent")           #set user agent imported from get_user_agent
    #setting the web new_browser with the cretaed Options
    browser = webdriver.Chrome("/Users/macbookpro_anas/Desktop/Machine-learning/drivers/chromedriver", options = opts)
    return browser


# Function that initialize dictionary with nan
def initialize_dict():
    data_list_title = ['Quartier ou lieu-dit','Année de construction', "Nombre d'étages", 
    'Etage du bien', 'Etat du bâtiment', 'Parkings intérieurs', 
    'Parkings extérieurs', 'Environnement', 'Surface habitable', 'Living', 
    'Surface living', 'Surface cuisine', 'Aménagement cuisine', 'Chambres', 
    'Surface chambre 1', 'Surface chambre 2', 'Surface chambre 3', 
    'Salles de bain', 'Salles de douche', 'Toilettes', 'Cave', 'Terrasse', 
    'Surface terrasse', 'Ascenseur', 'Accès handicapé', 'Alarme', 
    'Porte blindée', 'Prix de vente demandé', 'Revenu cadastral', 
    'CPEB - Consommation énergétique', 'Emission CO₂', 'Chauffage', 
    'Double vitrage', 'Attestation as-built', 
    "Attestation de conformité de l'installation électrique"]
    data_dict = dict(zip(data_list_title,[np.NAN]*len(data_list_title)))
    return data_dict


#Function that gets list of data from html
def get_data_form_html(driverr):
    WebDriverWait(driverr,60).until(EC.presence_of_element_located((By.ID, "iw-propertypage-verticals")))
    html = driverr.page_source
    soup = BeautifulSoup(html, 'lxml')
    tr_list= soup.find_all('tr')
    return tr_list


# Function that fills the previously intialized dictionary with data from webpage
def fill_dico(data_dict, tr_list):
    # Fill dico with data form WEBPAGE
    dico = {}
    for tr in tr_list:
        if (tr.find('td') is None) or (tr.find('th') is None):
            pass
        else:
            dico[tr.find('th').text.strip()] = tr.find('td').text.strip()
    # Fill my cretated data_dic with the info from WEBPAGE if it exists else fill with NaN
    for key in data_dict:
        data_dict[key] = dico.get(key, np.NaN)
    return data_dict


#Function that transforms dictionary of data to SQLite database
def dump_dict_to_db(data_dict):
    #Transform dictionary to dataframe
    index = [0]
    dict_to_df = pd.DataFrame(data_dict, index = index)
    #Transform dataframe to SQLite database
    df_into_sql = dict_to_df.to_sql('BruxellesDB',
    sqlite3.connect('/Users/macbookpro_anas/Desktop/Machine-learning/scrapper_immo/BruxellesDB.db'),
    if_exists='append',index=False)


#Function that calls all the previous functions and dump data into database
def scrap_one_ad(browserr):
    dict_of_nan = initialize_dict()
    data_list = get_data_form_html(browserr)
    filled_dict = fill_dico(dict_of_nan, data_list)
    dump_dict_to_db(filled_dict)


readDB = pd.read_sql('SELECT * FROM BruxellesDB',
sqlite3.connect('/Users/macbookpro_anas/Desktop/Machine-learning/scrapper_immo/BruxellesDB.db'))

# readDB1 = pd.read_sql('SELECT * FROM RealEstate',
# sqlite3.connect('/Users/macbookpro_anas/Desktop/Machine-learning/scrapper_immo/RealEstate.db'))


# if __name__ == '__main__':
#     browsr = initialize_browser(False)
#     browsr.get('https://www.immoweb.be/fr/annonce/appartement/a-vendre/uccle/1180/id7783221')
#     scrap_one_ad(browsr)

