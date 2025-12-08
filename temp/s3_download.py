import requests
from requests_aws4auth import AWS4Auth

# Storj credentials
access_key = 'jxv52ooceheejwc2njollbeo7gea'
secret_key = 'jyfrizf7g7vekzje7f4wwxa5frdplzbqkt3dao5nabvsbvaxqm46w'

# File URL
file_url = 'https://gateway.storjshare.io/safeclause-ai/demo1-user/lawsimpl-document.pdf'

# Create signed request
auth = AWS4Auth(access_key, secret_key, 'us-east-1', 's3')

# Download file
response = requests.get(file_url, auth=auth)

if response.status_code == 200:
    # Save to local file
    output_path = "/Users/punith/Downloads/lawsimpl-document-downloaded.pdf"
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded successfully to {output_path}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)