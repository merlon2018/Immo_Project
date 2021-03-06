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
from selenium.webdriver.chrome.options import Options
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s:%(message)s')
file_handler = logging.FileHandler('immo_parsing.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# Function that initialize dictionary with nan
# data_list_title = ['Code_postal','Quartier ou lieu-dit','Année de construction', "Nombre d'étages",
# 'Etage du bien', 'Etat du bâtiment', 'Parkings intérieurs',
# 'Parkings extérieurs', 'Environnement', 'Surface habitable', 'Living',
# 'Surface living', 'Surface cuisine', 'Aménagement cuisine', 'Chambres',
# 'Surface chambre 1', 'Surface chambre 2', 'Surface chambre 3',
# 'Salles de bain', 'Salles de douche', 'Toilettes', 'Cave', 'Terrasse',
# 'Surface terrasse', 'Ascenseur', 'Accès handicapé', 'Alarme',
# 'Porte blindée', 'Prix de vente demandé', 'Revenu cadastral',
# 'CPEB - Consommation énergétique', 'Emission CO₂', 'Chauffage',
# 'Double vitrage', 'Attestation as-built',
# "Attestation de conformité de l'installation électrique"]
def initialize_dict():
    data_list_title = ['ID', 'Code_postal', 'Disponibilité', 'Quartier ou lieu-dit', 'Année de construction',
                       'Étage', "Nombre d'étages", 'État du bâtiment', 'Façades', 'Parkings intérieurs',
                       'Parkings extérieurs', 'Surface habitable', 'Surface du salon', 'Salon', 'Salle à manger',
                       'Type de cuisine', 'Chambres', 'Surface de la chambre 1', 'Surface de la chambre 2',
                       'Surface de la chambre 3', 'Surface de la chambre 4', 'Salles de bains',
                       'Salles de douche', 'Toilettes', 'Cave', 'Surface de la cave'
                       'Surface de la terrasse', 'Orientation de la terrasse', 'Ascenseur',
                       'Accès handicapé', "Niveau de consommation d'énergie primaire", "Classe énergétique",
                       'Numéro du rapport PEB', 'Emission CO²',
                       "Consommation théorique totale d'énergie primaire", 'Type de chauffage',
                       'Double vitrage', 'Revenu cadastral', 'Prix de vente']
    data_dict = dict(zip(data_list_title, [np.NAN]*len(data_list_title)))
    return data_dict


#Function that gets list of data from html
def get_data_form_html(driverr):
    xpaath = '//*[@id="main-content"]/div[2]'
    WebDriverWait(driverr, 60).until(EC.presence_of_element_located((By.XPATH, xpaath)))
    html = driverr.page_source
    soup = BeautifulSoup(html, 'lxml')
    # this gets all informations about the property except its postal code and ID
    tr_list = soup.find_all('tr')
    try:
        location = soup.find('div', attrs={"class": "classified__information--address"}).text
        location = " ".join(location.split())
        ID_ = soup.find('div', attrs={'class': 'classified__information--immoweb-code'}).text
        ID_ = " ".join(ID_.split())
        return tr_list, location, ID_
    except:
        logger.warning(f'Something went wrong with url : {driverr.current_url}')
    


# Function that fills the previously intialized dictionary with data from webpage
def fill_dico(driverr, data_dict, tr_list, localization, ID):
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
    data_dict['ID'] = ID
    data_dict['Code_postal'] = localization
    logger.info(f'Property with {ID} has been scraped successfully')
    return data_dict


#Function that transforms dictionary of data to SQLite database
def dump_dict_to_db(data_dict, DBname):
    #Transform dictionary to dataframe
    index = [0]
    dict_to_df = pd.DataFrame(data_dict, index=index)
    #Transform dataframe to SQLite database
    df_into_sql = dict_to_df.to_sql(f'{DBname}',
                                    sqlite3.connect(
                                        f'/Users/macbookpro_anas/Desktop/Machine-learning/scrapper_immo/{DBname}.db'),
                                    if_exists='append', index=False)


#Function that calls all the previous functions and dump data into database
def scrap_one_ad(browserr, DataBname):
    dict_of_nan = initialize_dict()
    data_list, loc, identif = get_data_form_html(browserr)
    filled_dict = fill_dico(browserr, dict_of_nan, data_list, loc, identif)
    dump_dict_to_db(filled_dict, DataBname)


# readDB1 = pd.read_sql('SELECT * FROM RealEstate',
# sqlite3.connect('/Users/macbookpro_anas/Desktop/Machine-learning/scrapper_immo/RealEstate.db'))


# if __name__ == '__main__':
#     browsr = initialize_browser(False)
#     browsr.get('https://www.immoweb.be/fr/annonce/appartement/a-vendre/uccle/1180/id7783221')
#     scrap_one_ad(browsr)
