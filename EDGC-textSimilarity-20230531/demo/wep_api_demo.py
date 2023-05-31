# -- encoding:utf-8 --

import base64
import requests
import json
import cv2 as cv

url = "http://127.0.0.1:5000/convert"
url = "http://127.0.0.1:5000/similarity"
text = "如何更换花呗绑定银行卡"
text2 = "人工智能如何学习？"
data = {
    'text': text
}
data = {
    'text1': text,
    'text2': text2
}
result = requests.post(url=url, data=data)
if result.status_code == 200:
    print(result.text)
print(result.status_code)
