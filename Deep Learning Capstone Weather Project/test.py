import requests

url = "http://localhost:8080/2015-03-31/functions/function/invocations"
data = {'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Clouds_over_the_Atlantic_Ocean.jpg/800px-Clouds_over_the_Atlantic_Ocean.jpg'}

result = requests.post(url, json=data).json()
print(result)

