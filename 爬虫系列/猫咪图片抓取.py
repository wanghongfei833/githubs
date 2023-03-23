# -*- coding: utf-8 -*-
# @Time    : 2023/1/4 14:00
# @Author  : HongFei Wang

import requests
from lxml import etree

headers = {
		"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
}
# https://www.315e7124945a.com/tp/siyuefulitu/zipai/2582/01.jpg.txt
urls = "https://www.315e7124945a.com"
response = requests.get("https://www.315e7124945a.com/index/home.html")
tree = etree.HTML(response.text)
tree = tree.xpath('//*[@id="section-menu"]/div/div[4]/ul[1]//li')
for imageUrl in tree:
	Urls = imageUrl.xpath("./a[1]/@href")[0]  # '/tupian/list-自拍偷拍-1.html'
	Urls = urls + Urls  # 拿到每个分类链接
	responseImage = requests.get(Urls)
	treeImage = etree.HTML(responseImage.text)
	treeImage = treeImage.xpath('//*[@id="main-container"]/div[2]/div[2]/ul//li')  # 拿到对应类别的每个图像类url
	for li in treeImage:
		li = li.xpath("./a[1]/@href")  # /tupian/detail-261677.html
		liUrl = urls + li[0]
		print(liUrl)
		responseImageOnes = requests.get(liUrl, headers=headers)
		print(responseImageOnes.text)
		break
	break
# print(tree)