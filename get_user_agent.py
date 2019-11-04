from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import random



opts = Options()
opts.headless = True           #set to True to navigate headless

# Set web driver
browser = webdriver.Chrome("/Users/macbookpro_anas/Desktop/Machine-learning/drivers/chromedriver", options = opts)

def get_agent_from_web():
    browser.get('https://developers.whatismybrowser.com/useragents/explore/software_name/chrome/')
    html_ua = browser.page_source
    sobba = BeautifulSoup(html_ua, 'lxml')
    user_links = sobba.find_all('td', class_= 'useragent')
    user_links = [link.find('a').text for link in user_links]
    #print(user_links)
    return random.choice(user_links)


def dump_user_agent():
    browser.get('https://developers.whatismybrowser.com/useragents/explore/software_name/chrome/')
    html_ua = browser.page_source
    sobba = BeautifulSoup(html_ua, 'lxml')
    user_links = sobba.find_all('td', class_= 'useragent')
    user_links = [link.find('a').text for link in user_links]
    with open('user_agents_list.txt', 'w') as f:
        for item in user_links:
            f.write("%s\n" % item)


def get_agent():
    lines = open('user_agents_list.txt', 'r').read().splitlines()
    return random.choice(lines)


