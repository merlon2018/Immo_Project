""" change database name in scrap_one_ad() function in soup_parsing.py file 
in order to fill corresponding database """

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
#import get_user_agent
import random
import soup_parsing
import pandas as pd
import sqlite3
import numpy as np
import multiprocessing
#import scrap_proxies
import logging

logger_ = logging.getLogger(__name__)
logger_.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s:%(message)s')
file_handler = logging.FileHandler('immo_scaper.log')
file_handler.setFormatter(formatter)
logger_.addHandler(file_handler)


#os.chdir(os.environ['scrapper_immo'])

#Setting python options

def initialize_browser(headless_state):
	opts = Options()
	opts.headless = headless_state  # set to True to navigate headless
	#user_agent = get_user_agent.get_agent()
	#proxy = scrap_proxies.get_proxy()
	# set user agent imported from get_user_agent.py
	#opts.add_argument("user-agent=user_agent")
	# set different proxy from scrap_proxies.py
	#opts.add_argument('--proxy-server=%s' %proxy)
	#setting the web new_browser with the cretaed Options
	browser = webdriver.Chrome(
		"/Users/macbookpro_anas/Desktop/Machine-learning/drivers/chromedriver", options=opts)
	return browser


############################################################################################################
def get_page(url_):
	browser = initialize_browser(headless_state=False)
	browser.get(url_)
	try:
		WebDriverWait(browser, 60).until(EC.element_to_be_clickable(
			(By.XPATH, '//*[@id="uc-btn-accept-banner"]')))
		browser.find_element_by_xpath('//*[@id="uc-btn-accept-banner"]').click()
	except:
		pass
	return browser


init_url = 'https://www.immoweb.be/fr/recherche/appartement/a-vendre/bruxelles/arrondissement?countries=BE&page=1&orderBy=relevance'

#browser = get_page(init_url)


#Select Apartment categorie
# browser.switch_to.frame('IWEB_IFRAME_ID_SEARCH')
# obj = Select(browser.find_element_by_name("xidcategorie"))
# obj.select_by_index(6)
# #type city and send key
# browser.switch_to.frame('IWEB_IFRAME_ID_SEARCH')
# elem = browser.find_element_by_xpath("//input[@id='localisation']")
# elem.send_keys("bruxelles")
# time.sleep(2)
# elem.send_keys(Keys.TAB)
# time.sleep(2)
# elem.send_keys("bruxelles")
# time.sleep(2)
# elem.send_keys(Keys.ARROW_DOWN)
# elem.send_keys(Keys.TAB)
# Wait for Element until it is clickable
# wait = WebDriverWait(browser, 10)
# search_bar = wait.until(EC.element_to_be_clickable(
# 	(By.XPATH, "//*[@id='column-central']/div[2]/div[14]/p[3]/button")))
# browser.execute_script("arguments[0].click();", search_bar)

# url = browser.current_url+'?page='

# Get total number of pages to scrap


# def get_total_number_of_pages():
# 	tag = browser.find_element_by_xpath(
# 		'//div[@class="top results-display-pagination"]//ul[@class="nav-nummer"]').get_attribute('innerHTML')
# 	soupe = BeautifulSoup(tag, 'lxml')
# 	find_li = soupe.find_all('li')
# 	get_page_num = []
# 	for elem in find_li:
# 		if elem.find('a') is not None:
# 			get_page_num.append(elem.find('a').text)
# 		else:
# 			pass
# 	return int(get_page_num[-1])


def get_total_number_of_pages(browser):
	xpath_ = '/html/body/div[1]/div[2]/div/main/div/div/div[2]/div[1]/div/div/div[1]/div/div/div[1]/div/nav/ul/li[4]/a'
	WebDriverWait(browser, 10).until(
		EC.presence_of_element_located((By.XPATH, xpath_)))
	tag = browser.find_element_by_xpath(xpath_).get_attribute('innerHTML')
	soup = BeautifulSoup(tag, 'lxml')
	# print(soup.prettify())
	raw_text = soup.find('body').text.strip()
	num_of_pages = int(re.search('\d+', raw_text).group())
	return num_of_pages


# total_number_of_pages = get_total_number_of_pages(browser)


#browser.close()


#################################       We are in the list of flats WEBPAGE        #####################################

#list_of_aparts = browser.find_elements_by_class_name("result-xl-content")       #returns the list of all element

#browser.execute_script("arguments[0].click();", list_of_aparts[0])
#list_of_aparts[0].click()

# state is used to know if the parsing occured before the interuption
def loop_all_aparts(pagenum):
	url = 'https://www.immoweb.be/fr/recherche/appartement/a-vendre/bruxelles/arrondissement?countries=BE&page=' + \
		str(pagenum)+'&orderBy=relevance'
	browsers = get_page(url)
	# WebDriverWait(browsers, 20).until(
	# 	EC.presence_of_element_located((By.CLASS_NAME, 'card--result__body')))
	list_of_aparts = browsers.find_elements_by_class_name("card--result__body")  # find all elementS (flats)
	for i in range(len(list_of_aparts)):
		#browsers.execute_script("arguments[0].click();", list_of_aparts[i])
		list_of_aparts[i].click()
		soup_parsing.scrap_one_ad(browsers, "BruxellesDB_NEW")
		logger_.info(f'Flat {i+1} in page {pagenum} has been scraped successfully')
		t_1 = random.uniform(10, 15)
		time.sleep(t_1)
		browsers.back()
		t_2 = random.uniform(10, 15)
		time.sleep(t_2)
		ignored_exceptions = (NoSuchElementException,
								StaleElementReferenceException)
		wb_wait = WebDriverWait(browsers, 10, ignored_exceptions=ignored_exceptions)
		wb_wait.until(EC.presence_of_element_located(
			(By.CLASS_NAME, "search-results__list")))
		list_of_aparts = browsers.find_elements_by_class_name(
			"card--result__body")
		if (i+1) % 10 == 0:
			browsers.quit()
			browsers = get_page(url)
			list_of_aparts = browsers.find_elements_by_class_name("card--result__body")
	logger_.info(f'Page {pagenum} completed successfully')


# def loop_all_aparts(pagenum):
# 	url = 'https://www.immoweb.be/fr/recherche/appartement/a-vendre/bruxelles/arrondissement?countries=BE&page=' + \
#		str(pagenum)+'&orderBy=relevance'
#	browsers = get_page(url)
# 	WebDriverWait(browsers, 20).until(
# 		EC.presence_of_element_located((By.CLASS_NAME, 'search-results__list')))
# 	list_of_aparts = browsers.find_elements_by_class_name(
# 		"card--result__body")  # find all elementS (flats)
# 	for i in range(len(list_of_aparts)):
# 		state = False
# 		try:
# 			browsers.execute_script("arguments[0].click();", list_of_aparts[i])
# 			soup_parsing.scrap_one_ad(browsers, "BruxellesDB_NEW")
# 			state = True
# 			logger_.info(f'Flat {i+1} in page {pagenum} has been scraped successfully')
# 			t_1 = random.uniform(10, 15)
# 			browsers.implicitly_wait(t_1)
# 			browsers.back()
# 			t_2 = random.uniform(10, 15)
# 			browsers.implicitly_wait(t_2)
# 			ignored_exceptions = (NoSuchElementException,
# 			                      StaleElementReferenceException)
# 			wb_wait = WebDriverWait(browsers, 10, ignored_exceptions=ignored_exceptions)
# 			wb_wait.until(EC.presence_of_element_located(
# 				(By.CLASS_NAME, "search-results__list")))
# 			list_of_aparts = browsers.find_elements_by_class_name(
# 				"card--result__body")
# 			if (i+1) % 10 == 0:
# 				browsers.quit()
# 				browsers = initialize_browser(headless_state=False)
# 				browsers.get(url)
# 				list_of_aparts = browsers.find_elements_by_class_name("result-xl-content")
# 		except:
# 			browsers.refresh()
# 			if state == False:
# 				soup_parsing.scrap_one_ad(browsers)
# 				logger_.info(f'Flat {i+1} in page {pagenum} has been scraped successfully')
# 				browsers.back()
# 				list_of_aparts = browsers.find_elements_by_class_name("result-xl-content")
# 	logger_.info(f'Page {pagenum} completed successfully')


def scrap_entire_webpage(pagenum):
	loop_all_aparts(pagenum)


# time.sleep(5)
# with multiprocessing.Pool(processes=2) as pool:
# 	pool.map(scrap_entire_webpage, range(1, total_number_of_pages+1))


# readDB = pd.read_sql('SELECT * FROM BruxellesDBV2',
#                      sqlite3.connect('/Users/macbookpro_anas/Desktop/Machine-learning/scrapper_immo/BruxellesDBV2.db'))


# def loop_all_aparts(list_of_aparts, browsers, counter):
#     browsers.execute_script("arguments[0].click();", list_of_aparts[6])
#     xpath = "//div[@class='actions']//iw-propertypage-topbar-search-actions//div//span[@class='icon--arrow']"
#     t = random.uniform(1, 2)
#     while counter <= 24:
#         browsers.implicitly_wait(t)
#         soup_parsing.scrap_one_ad(browsers)
#         try:
#             rightarrow = WebDriverWait(browser, 30).until(
#                 EC.visibility_of_element_located((By.XPATH, xpath)))
#             rightarrow.click()
#         except Exception as e:
#             browsers.refresh()
#             print(e)
#         counter += 1


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
