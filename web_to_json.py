import requests
from bs4 import BeautifulSoup
import json

url = "https://google.com" # Replace with your target URL
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# Example: Extracting all paragraph texts
paragraphs = [p.get_text() for p in soup.find_all('p')]

# Structure the data into a dictionary
data = {
    "title": soup.title.string,
    "paragraphs": paragraphs
}

# Convert to JSON string
json_output = json.dumps(data, indent=4)
print(json_output)