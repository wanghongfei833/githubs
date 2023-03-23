# -*- coding: utf-8 -*-
# @Time    : 2023/1/4 15:53
# @Author  : HongFei Wang


urls = "magnet:?xt=urn:btih:8F638EB4795BB46E101B19FE31AC7775EAABFAF6 "

import os
import sys
import urllib

import pymysql
import wget


# 连接数据库 获取对应数据
def getDb():
	db = pymysql.connect(host='localhost', user='***', password='***', database='****')
	cursor = db.cursor()
	sql = "select filed1, filed2, filed3 from table_name"
	cursor.execute(sql)
	result = cursor.fetchall()
	for row in result:
		title = row[0]
		download = row[1]
		image = row[2]
		path = makeDir("D:\\resources\\" + title + "\\")
		getZip(download, path)
		getImage(image, path)
	db.close()


# 创建文件夹
def makeDir(path):
	path = path.strip()
	path = path.rstrip("\\")
	isExists = os.path.exists(path)
	if not isExists:
		# 如果不存在则创建目录
		# 创建目录操作函数
		os.makedirs(path)
		return path
	else:
		return path


# 下载资源压缩包
def getZip(url, path):
	filename = getFileName(url)
	wget.download(url, path + '\\' + filename, bar=bar_progress)
	print(filename + "文件下载完成")


# 下载进度条
def bar_progress(current, total, width=100):
	progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
	sys.stdout.write("\r" + progress_message)
	sys.stdout.flush()


# 下载资源图片
def getImage(url, path):
	filename = getFileName(url)
	urllib.request.urlretrieve(url, filename=path + '\\' + filename)
	print(filename + "图片下载完成")


# 获取资源名称
def getFileName(url):
	return url.split('/')[-1]


if __name__ == '__main__':
	getDb()