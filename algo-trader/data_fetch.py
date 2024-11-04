import json

def get_paper_creds():
    with open("/Users/alejandrovillasmil/Documents/GitHub/algo-trader/credentials.json", 'r') as file:
        creds = json.load(file)
    key = creds["ALPACA_PAPER_KEY"]
    secret = creds["ALPACA_PAPER_SECRET"]
    return key, secret

