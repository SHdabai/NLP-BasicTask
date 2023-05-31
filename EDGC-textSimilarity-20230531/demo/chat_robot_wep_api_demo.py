# -- encoding:utf-8 --

import base64
import requests
import json
import cv2 as cv

url = "http://127.0.0.1:5000/chat_robot2"
data = {
    "question": "如何绑定花呗的支付方式",
    "threshold": 0.99
}
result = requests.post(url=url, data=data)
if result.status_code == 200:
    print(result.text)
print(result.status_code)
