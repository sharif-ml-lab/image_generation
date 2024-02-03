import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from fake_useragent import UserAgent
from selenium.webdriver.support import expected_conditions as EC
import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

op = uc.ChromeOptions()
browser = uc.Chrome(options=op)

username = 'EMAIL'
password = 'PASSWORD'

browser.get("https://accounts.google.com/signin")


WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.ID, "identifierId"))).send_keys(username)
browser.find_element(By.ID, "identifierId").send_keys(Keys.RETURN)

sleep(5)
browser.execute_script("""
    var passwordField = document.querySelector("input[type='password']");
    passwordField.value = 'PASSWORD';
    passwordField.dispatchEvent(new Event('change'));
    var enterKeyEvent = new KeyboardEvent("keydown", {
        bubbles: true, cancelable: true, keyCode: 13
    });
    passwordField.dispatchEvent(enterKeyEvent);
    """)

sleep(5)
browser.get("http://bard.google.com")

richText = WebDriverWait(browser, 15).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "rich-textarea .ql-editor.textarea"))
)
richText.click()

text_to_type = "Create an Image of Real Dog"
for char in text_to_type:
    browser.execute_script("""
        var evt = new KeyboardEvent('keypress', {
            key: arguments[0],
            bubbles: true
        });
        document.querySelector("rich-textarea .ql-editor.textarea").dispatchEvent(evt);
    """, char)

browser.execute_script("""
    var evt = new KeyboardEvent('keydown', {
        key: 'Enter',
        keyCode: 13,
        bubbles: true,
        cancelable: true
    });
    document.querySelector("rich-textarea").dispatchEvent(evt);
""")


images = WebDriverWait(browser, 15).until(
    EC.presence_of_all_elements_located((By.CLASS_NAME, "image"))
)

image_folder = "downloaded_images"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

count = 0
for index, image in enumerate(images):
    src = image.get_attribute("src")
    alt = image.get_attribute("alt")
    response = requests.get(src)
    if response.status_code == 200:
        with open(os.path.join(image_folder, f"image_{index}.jpg"), 'wb') as file:
            file.write(response.content)
            count += 1
            print(count, alt)
