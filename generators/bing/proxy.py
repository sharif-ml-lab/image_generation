import requests
from bs4 import BeautifulSoup
import random


def get_proxy():
    url = "https://free-proxy-list.net/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    proxies = []
    for row in soup.find("table", {"class": "table"}).find_all("tr")[1:10]:
        tds = row.find_all("td")
        try:
            ip = tds[0].text.strip()
            port = tds[1].text.strip()
            https = tds[6].text.strip()

            if https == "yes":
                proxy = f"{ip}:{port}"
            else:
                proxy = f"{ip}:{port}"

            proxies.append(proxy)
        except IndexError:
            continue

    return random.choice(proxies) if proxies else None
