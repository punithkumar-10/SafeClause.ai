import requests
from requests_aws4auth import AWS4Auth
import os

# Storj credentials
access_key = 'jxv52ooceheejwc2njollbeo7gea'
secret_key = 'jyfrizf7g7vekzje7f4wwxa5frdplzbqkt3dao5nabvsbvaxqm46w'
endpoint_url = 'https://gateway.storjshare.io'

bucket_name = "safeclause-ai"
local_file_path = "/Users/punith/Downloads/lawsimpl-document.pdf"
object_key = "demo1-user/lawsimpl-document.pdf"

# Read file
with open(local_file_path, 'rb') as f:
    file_data = f.read()

# Create signed request and upload
auth = AWS4Auth(access_key, secret_key, 'us-east-1', 's3')
url = f'{endpoint_url}/{bucket_name}/{object_key}'

headers = {
    'Content-Length': str(len(file_data)),
    'Content-Type': 'application/octet-stream'
}

response = requests.put(url, data=file_data, auth=auth, headers=headers)

if response.status_code == 200:
    print(f"Uploaded successfully")
    print(f"File URL: {url}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)