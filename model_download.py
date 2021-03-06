# 参考 : https://qiita.com/jun40vn/items/66fff06abe48e01e23e3

import requests

def download_file_from_google_drive(id, destination):

       # ダウンロード画面のURL
       URL = "https://drive.google.com/uc?export=download"

       session = requests.Session()

       response = session.get(URL, params = { 'id' : id }, stream = True)
       token = get_confirm_token(response)

       if token:
           params = { 'id' : id, 'confirm' : token }
           response = session.get(URL, params = params, stream = True)

       save_response_content(response, destination)

def get_confirm_token(response):
       for key, value in response.cookies.items():
           if key.startswith('download_warning'):
               return value

       return None

def save_response_content(response, destination):
       CHUNK_SIZE = 32768

       with open(destination, "wb") as f:
           for chunk in response.iter_content(CHUNK_SIZE):
               if chunk: # filter out keep-alive new chunks
                   f.write(chunk)

download_file_from_google_drive("14m9thCxzSZYAOnSMuv9QOuIbcWvU09vl", "bert_output2.zip")