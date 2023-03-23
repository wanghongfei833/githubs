# -*- coding: utf-8 -*-
# @Time    : 2023/1/4 14:30
# @Author  : HongFei Wang
import base64
import binascii
import importlib
import sys

importlib.reload(sys)
f1 = open("txt1.txt", 'r').read()
f2 = open("txt2.txt", 'r').read()

image_data = base64.b64decode(f2)
print("十六进制:", image_data[:20])
image_data = binascii.b2a_hex(image_data[::-1]).decode("utf-8")
print("字符串", image_data[:20])

# with open('./img.jpg', 'wb') as f:
# 	f.write(image_data)