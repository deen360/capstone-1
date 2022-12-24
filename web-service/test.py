import requests

car = {
    "Cylinders": 4,
    "horsepower": 68.0,
    "weight": 2065,
    "acceleration": 14,

}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=car)
print(response.json())
