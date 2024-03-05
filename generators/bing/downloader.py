import os
import json
import logging
import datetime
import pandas as pd
import shortuuid
import time
from generators.bing.proxy import get_proxy
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from urllib import request


logging.basicConfig(level=logging.INFO)


LOG_FILE_PATH = "utils/data/bing/update.log"
USERS_FILE_PATH = "utils/data/bing/users.json"
headless = True
global_driver = None


def get_or_create_driver(url, should_reset=False):
    global global_driver, headless
    if (
        should_reset
        or global_driver is None
        or not global_driver.service.is_connectable()
    ):
        options = Options()
        options.add_argument(f'--proxy-server="{get_proxy()}"')
        if headless:
            options.add_argument("--headless")
        global_driver = webdriver.Firefox(options=options)
        global_driver.get(url)
    return global_driver


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def log_credit_update():
    with open(LOG_FILE_PATH, "w") as log_file:
        log_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def should_update_credits():
    try:
        with open(LOG_FILE_PATH, "r") as log_file:
            date = log_file.read().strip()
            last_update = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
            return (datetime.datetime.now() - last_update).total_seconds() > 12 * 3600
    except FileNotFoundError:
        return True


def get_users_with_credits():
    with open(USERS_FILE_PATH, "r") as file:
        return json.load(file)


def set_users_with_credits(users):
    with open(USERS_FILE_PATH, "w") as file:
        json.dump(users, file, indent=4)


def process_prompt(prompt, path, output, caption_output_path):
    users = get_users_with_credits()
    for user in users:
        if user["credits"] > 0:
            logined = login_to_bing(user["email"], user["password"])
            downloaded = download_images(
                user, prompt, path, output, caption_output_path
            )
            if logined and downloaded:
                break
            else:
                continue
        else:
            logging.info(f"Skipping user {user['email']} due to insufficient credits.")


def login_to_bing(username, password):
    driver = get_or_create_driver(
        "https://www.bing.com/images/create", should_reset=True
    )
    try:
        WebDriverWait(driver, 6).until(
            EC.element_to_be_clickable((By.ID, "bnp_btn_accept"))
        ).click()
    except TimeoutException:
        pass

    try:
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "create_btn_c"))
        ).click()
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, ".signin_content > a:nth-child(1)")
            )
        ).click()
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "i0116"))
        ).send_keys(username)
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "idSIButton9"))
        ).click()
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "i0118"))
        ).send_keys(password)
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "idSIButton9"))
        ).click()
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "acceptButton"))
        ).click()
        time.sleep(10)
        logging.info(f"Logined {username}")
        return True
    except:
        logging.warning(f"Can Not Login: {username}")
        return False


def get_credits(email):
    driver = get_or_create_driver("https://www.bing.com/images/create")
    try:
        credit = int(
            WebDriverWait(driver, 10)
            .until(EC.element_to_be_clickable((By.ID, "token_bal")))
            .text
        )
        return credit
    except TimeoutException:
        logging.warning("Failed to retrieve credits for user: " + email)
        return 0


def update_user_credit_in_file(email, new_credits):
    with open(USERS_FILE_PATH, "r") as file:
        users = json.load(file)

    for user in users:
        if user["email"] == email:
            user["credits"] = new_credits
            break

    with open(USERS_FILE_PATH, "w") as file:
        json.dump(users, file, indent=4)


def update_user_credits():
    with open(USERS_FILE_PATH) as io:
        users = json.load(io)
    try:
        for user in users:
            driver = get_or_create_driver("https://www.bing.com/images/create")
            logined = login_to_bing(user["email"], user["password"])
            if logined == True:
                user["credits"] = get_credits(user["email"])
                logging.info(f"Updated credits for {user['email']}: {user['credits']}")
            else:
                logging.info(f"Updated credits for {user['email']}: {user['credits']}")
                user["credits"] = 0
            set_users_with_credits(users)
            driver.quit()
    except Exception as e:
        print(e)


def download_images(user, prompt, path, output, caption_output_path, max_images=4):
    ensure_directory_exists(path)
    driver = get_or_create_driver("https://www.bing.com/images/create")
    try:
        credit = int(
            WebDriverWait(driver, 10)
            .until(EC.element_to_be_clickable((By.ID, "token_bal")))
            .text
        )
        logging.warning(f"Credits: {credit}")

        inp = driver.find_element(By.ID, "sb_form_q")
        inp.send_keys(prompt + Keys.RETURN)
        new_credits = user["credits"] - 1
        update_user_credit_in_file(user["email"], new_credits)

        WebDriverWait(driver, 80).until(
            EC.presence_of_all_elements_located((By.XPATH, '//img[@class="mimg"]'))
        )
        images = driver.find_elements(By.XPATH, '//img[@class="mimg"]')
        random_id = shortuuid.ShortUUID().random(length=8)
        for idx, img in enumerate(images[:max_images], start=1):
            output["image_name"].append(f"{random_id}-{idx}.jpg")
            output["caption"].append(prompt)
            src = img.get_attribute("src")
            img_id = src.split("/")[-1].split("?")[0]
            full_size = src.split("?")[0] + "?pid=ImgGn"

            filename = os.path.join(path, f"{random_id}-{idx}.jpg")
            if not os.path.exists(filename):
                request.urlretrieve(full_size, filename)
                logging.info(f"Downloaded {filename}")
                pd.DataFrame(output).to_csv(caption_output_path, sep="|")
        return True

    except TimeoutException:
        logging.error("Failed to load page or find elements")
        return False


def process(prompts, opath, caption_output_path=None):
    if should_update_credits():
        update_user_credits()
        log_credit_update()

    output = {"image_name": [], "caption": []}
    driver = get_or_create_driver("https://www.bing.com/images/create")
    try:
        for prompt in prompts:
            process_prompt(prompt, opath, output, caption_output_path)
    finally:
        if caption_output_path is None:
            caption_output_path = opath + "/caption.csv"
        pd.DataFrame(output).to_csv(caption_output_path, sep="|")


if __name__ == "__main__":
    update_user_credits()
    log_credit_update()
