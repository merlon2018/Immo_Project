from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import re
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import ElementClickInterceptedException
from bs4 import BeautifulSoup
import os
import requests
import urllib.request
import get_user_agent
import random
import soup_parsing
import pandas as pd
import numpy as np
import multiprocessing


#os.chdir(os.environ['scrapper_immo'])

#Setting python options

def initialize_browser(headless_state):
    opts = Options()
    opts.headless = headless_state           #set to True to navigate headless
    user_agent = get_user_agent.get_agent()
    opts.add_argument("user-agent=user_agent")           #set user agent imported from get_user_agent
    #setting the web new_browser with the cretaed Options
    browser = webdriver.Chrome("/Users/macbookpro_anas/Desktop/Machine-learning/drivers/chromedriver", options = opts)
    return browser


############################################################################################################
browser = initialize_browser(headless_state=False)
browser.get('https://www.immoweb.be/fr/immo/a-vendre')
#Select Apartment categorie
browser.switch_to.frame('IWEB_IFRAME_ID_SEARCH')
obj = Select(browser.find_element_by_name("xidcategorie"))
obj.select_by_index(6)
#type city and send key
browser.switch_to.frame('IWEB_IFRAME_ID_SEARCH')
elem = browser.find_element_by_xpath("//input[@id='localisation']")
elem.send_keys("bruxelles")
time.sleep(2)
elem.send_keys(Keys.TAB)
time.sleep(2)
# elem.send_keys("bruxelles")
# time.sleep(2)
# elem.send_keys(Keys.ARROW_DOWN)
# elem.send_keys(Keys.TAB)
# Wait for Element until it is clickable
wait = WebDriverWait(browser, 10)
search_bar = wait.until(EC.element_to_be_clickable((By.XPATH, "//*[@id='column-central']/div[2]/div[14]/p[3]/button")))
browser.execute_script("arguments[0].click();", search_bar)

url = browser.current_url+'?page='


# Get total number of pages to scrap
def get_total_number_of_pages():
    tag = browser.find_element_by_xpath(
        '//div[@class="top results-display-pagination"]//ul[@class="nav-nummer"]').get_attribute('innerHTML')
    soupe = BeautifulSoup(tag, 'lxml')
    find_li = soupe.find_all('li')
    get_page_num = []
    for elem in find_li:
        if elem.find('a') is not None:
            get_page_num.append(elem.find('a').text)
        else: 
            pass
    return int(get_page_num[-1])


total_number_of_pages = get_total_number_of_pages()

#browser.close()
#################################       We are in the list of flats WEBPAGE        ##################################### 

#list_of_aparts = browser.find_elements_by_class_name("result-xl-content")       #returns the list of all element

#browser.execute_script("arguments[0].click();", list_of_aparts[0])
#list_of_aparts[0].click()

# def loop_all_aparts(list_of_aparts,browsers):
#     try:
#         for i in range(len(list_of_aparts)):
#             browsers.execute_script("arguments[0].click();", list_of_aparts[i])
#             soup_parsing.scrap_one_ad(browsers)
#             t = random.uniform(2,3)
#             browsers.implicitly_wait(t)
#             browsers.back()
#             ignored_exceptions = (NoSuchElementException, StaleElementReferenceException)
#             wb_wait = WebDriverWait(browsers, 10, ignored_exceptions=ignored_exceptions)
#             wb_wait.until(EC.presence_of_element_located((By.CLASS_NAME, "result-xl-content")))
#             list_of_aparts = browsers.find_elements_by_class_name("result-xl-content")
#     except:
#         browsers.refresh()


def loop_all_aparts(list_of_aparts,browsers):
    while counter =< len(list_of_aparts):
        xpath = "//div[@class='actions']//iw-propertypage-topbar-search-actions//div//span[@class='icon--arrow']"
        t = random.uniform(1,2)
        browsers.implicitly_wait(t)
        try:       
            rightarrow = WebDriverWait(browser, 30).until(EC.visibility_of_element_located((By.XPATH, xpath)))
            rightarrow.click() 
        except Exception as e:
            browsers.refresh() 
            print(e)            
        counter+=1


def scrap_entire_webpage(pagenum):  
    new_browser = initialize_browser(headless_state=False)
    new_browser.get(url+str(pagenum))
    new_browser.implicitly_wait(10)
    list_of_flats = new_browser.find_elements_by_class_name("result-xl-content")
    loop_all_aparts(list_of_flats,new_browser)


time.sleep(2)
with multiprocessing.Pool() as pool:
    counter = 0
    pool.map(scrap_entire_webpage, range(1, total_number_of_pages+1))


# processes = []
# for page in range(1,total_number_of_pages+1):
#     p = multiprocessing.Process(target=scrap_entire_webpage, args=[page])
#     p.start()
#     processes.append(p)

# for process in processes:
#     process.join()


############        Select property, fetch data, get to previous page and so on ...     #####################

# for i in range(len(list_of_aparts)):
#     #print(i)        # FOR COUTING
#     browser.execute_script("arguments[0].click();", list_of_aparts[i])
#     #list_of_aparts[i].click()
#     time.sleep(5)
#     browser.back()
#     ignored_exceptions = (NoSuchElementException, StaleElementReferenceException)
#     wb_wait = WebDriverWait(browser, 10, ignored_exceptions=ignored_exceptions)
#     wb_wait.until(EC.presence_of_element_located((By.CLASS_NAME, "result-xl-content")))
#     list_of_aparts = browser.find_elements_by_class_name("result-xl-content")

##############################################################################################################


################################    Code to loop over all ads when we are in the first one  ################################ 

# count = True
# counter=1
# while count:
#     xpath = "//div[@class='actions']//iw-propertypage-topbar-search-actions//div//span[@class='icon--arrow']"
#     t = random.uniform(5,10)
#     browser.implicitly_wait(t)
#     try:       
#         rightarrow = WebDriverWait(browser, 30).until(EC.visibility_of_element_located((By.XPATH, xpath)))
#         rightarrow.click() 
#     except TimeoutException:
#         print("scrapping completed successfully")
#         count = False  
#     #time.sleep(5)           
#     counter+=1

################################ ################################ ################################ ################################ 


