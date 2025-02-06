import requests

url = "http://0.0.0.0:8082/predict"
image_path = "1cac244e2a73d3d396c2204b958ec6b0192d24ee.jpg"

with open(image_path, "rb") as img:
    files = {"file": img}
    response = requests.post(url, files=files)

print(response.json())