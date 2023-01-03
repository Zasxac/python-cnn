from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import urllib.request
import time
import sys
import os
site = 'https://www.google.com/search?tbm=isch&q='+"pineapple"
driver = webdriver.Chrome(executable_path = 'C:\\chrome.exe')
scroller = 0
driver.get(site)
i = 0
while scroller<7:  
    driver.execute_script("window.scrollBy(0,document.body.scrollHeight)")
    try:
        driver.find_element_by_xpath("/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[5]/input").click()
    except Exception as e:
        pass
    time.sleep(5)
    scroller+=1
soup = BeautifulSoup(driver.page_source, 'html.parser')
driver.close()
img_tags = soup.find_all("img", class_="rg_i")


image_no = 0
while image_no != 500:
    for i in img_tags:
        try:
            urllib.request.urlretrieve(i['src'], str(image_no)+".jpg")
            image_no+=1
            print("downloaded images = "+str(image_no),end='\r')
        except Exception as e:
            pass
