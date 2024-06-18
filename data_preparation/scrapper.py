import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

import os
import time
import base64


service = Service(executable_path=r'/usr/bin/chromedriver')
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(service=service, options=options)

def scrap(breed):
    url = ("https://www.google.com/search?q={s}&tbm=isch&tbs=sur%3Afc&hl=en&ved=0CAIQpwVqFwoTCKCa1c6s4-oCFQAAAAAdAAAAABAC&biw=1251&bih=568")

    driver.get(url.format(s=breed))
    scroll_height = 0
    tmp_scroll_height = 0
    src = []

    for d in range(0, 10):
        tmp_scroll_height = scroll_height
        driver.execute_script(f"window.scrollTo({scroll_height},document.body.scrollHeight);")
        time.sleep(2)
        scroll_height = driver.execute_script("return document.body.scrollHeight")
        print(f'Scroll height: {scroll_height}')
        if tmp_scroll_height == scroll_height:
            break

    imgResults = driver.find_elements(By.XPATH,"//img[contains(@class,'YQ4gaf')]")


    for img in imgResults:
        src.append(img.get_attribute('src'))    

    print(breed, len(src))
    count = 0
    for i in range(0, len(src)):
        if 'favicon' not in src[i]:
            if src[i].startswith('http'):
                img_data = requests.get(src[i]).content
                path = f'./dataset/{breed}/{count}.jpg'
                with open(path, 'wb') as handler:
                    handler.write(img_data)
                count+=1

breeds = ['Labrador+Retriever', 'Golden+Retriever', 'German+Shepherd', 'Beagle','French+Bulldog','Greyhound', 'Bulldog', 'Poodle', 'Rottweiler', 'Yorkshire+Terrier']

for i in breeds:
    os.makedirs(f'dataset/{i}', exist_ok=True)
    scrap(i)
