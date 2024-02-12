import requests
import json
from io import BytesIO
import base64
from PIL import Image


base_url = "http://localhost:8886/"
url = base_url + "v1/generation/text-to-image"
with open("utils/data/juggernaut/body.json") as io:
    body = json.load(io)


def generate(prompt):
    body["prompt"] = prompt
    req = requests.post(url=url, json=body)
    url_image = req.json()[0]["url"]
    url_image = base_url + "/".join(url_image.split("/")[3:])
    raw_image = requests.get(url_image, stream=True).raw
    image = Image.open(raw_image)
    return image
