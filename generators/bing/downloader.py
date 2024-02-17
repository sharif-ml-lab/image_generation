import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from urllib import request

driver = webdriver.Firefox()
driver.get("https://www.bing.com/images/create")
input("Press Enter when you sign in\n")
driver.get("https://www.bing.com/images/create")
credit = int(driver.find_element(By.ID, "token_bal").text)
while credit > 0:
    print(f"Credits: {credit}")
    inp = driver.find_element(By.ID, "sb_form_q")
    inp.send_keys(
        "a man is washing dishes while a woman is coding with a laptop on the table"
    )
    inp.submit()
    time.sleep(30)
    images = driver.find_elements(By.XPATH, '//img[@class="mimg"]')
    for img in images:
        src = img.get_attribute("src")
        img_id = src.split("/")[-1].split("?")[0]
        full_size = src.split("?")[0] + "?pid=ImgGn"
        request.urlretrieve(full_size, f"utils/data/bing/{img_id}.jpg")
        print(f"Downloaded {img_id}")
    credit -= 1
driver.quit()
