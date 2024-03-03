import requests
from bs4 import BeautifulSoup
import random
import urllib.request, socket


def is_bad_proxy(pip):
    try:
        proxy_handler = urllib.request.ProxyHandler({"http": pip})
        opener = urllib.request.build_opener(proxy_handler)
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)
        urllib.request.urlopen("https://www.bing.com/images/create")
    except urllib.error.HTTPError as e:
        print("Error code: ", e.code)
        return e.code
    except Exception as detail:
        print("ERROR:", detail)
        return 1
    return 0


def get_proxy():
    url = "https://free-proxy-list.net/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    proxies = []
    for row in soup.find("table", {"class": "table"}).find_all("tr")[1:15]:
        tds = row.find_all("td")
        try:
            ip = tds[0].text.strip()
            port = tds[1].text.strip()
            https = tds[6].text.strip()
            if not is_bad_proxy(f"{ip}:{port}"):
                if https == "yes":
                    proxy = f"https://{ip}:{port}"
                else:
                    proxy = f"http://{ip}:{port}"

                proxies.append(proxy)
        except IndexError:
            continue
    return random.choice(proxies) if proxies else None


if __name__ == "__main__":
    get_proxy()
