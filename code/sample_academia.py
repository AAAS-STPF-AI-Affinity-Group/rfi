import requests

# Set base URL
base_url = "https://www.nitrd.gov"

# Academia only
respondent_type = "Academia"

# Fake headers like a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://www.nitrd.gov/coordination-areas/ai/90-fr-9088-responses/"
}

# Fetch the API endpoint
response = requests.get(f"{base_url}/api/response-list?type={respondent_type}", headers=headers)

# Check if successful
if response.status_code == 200:
    data = response.json()
    print(f"Found {len(data)} responses for {respondent_type}:\n")
    for entry in data:
        respondent = entry['respondent']
        pdf_link = base_url + entry['file']
        print(f"- {respondent}: {pdf_link}")
else:
    print(f"Failed to fetch data, status code {response.status_code}")
