# -*- coding: utf-8 -*-
# @Time    : 2022/12/23 17:17
# @Author  : HongFei Wang
import configparser
import os
import time

import pyautogui as pg
import xlrd


def find_image(path, index=0):
	findImagePos = pg.locateCenterOnScreen(path)
	if findImagePos is not None:
		return findImagePos
	else:
		index += 1
		time.sleep(0.5)
		print(f"重调{index}次")
		findImagePos = find_image(path, index)
		if findImagePos is not None:
			return findImagePos


def findWindow(windowns_name):
	windows = pg.getWindowsWithTitle(windowns_name)
	if len(windows) == 0:
		raise Exception("微信窗口未找到")
	return windows[0]


pg.FAILSAFE = True
cfp = configparser.ConfigParser()
cfp.read("config.ini", encoding="utf-8")

'''获取所有的selections'''
selections = cfp.sections()
find_pos = eval(cfp.get("pos", "find_pos"))
particulars = eval(cfp.get("pos", "particulars"))
leftTop = eval(cfp.get("pos", "leftTop"))
imagesize = eval(cfp.get("pos", "imagesize"))
field_work = eval(cfp.get("pos", "field_work"))
image1 = eval(cfp.get("pos", "image1"))
image2 = eval(cfp.get("pos", "image2"))
getbackpos = eval(cfp.get("pos", "getbackpos"))
getbacklist = eval(cfp.get("pos", "getbacklist"))

wxWindow = findWindow("chrome")
wxWindow.activate()  # 激活窗口,将窗口最前化
time.sleep(0.5)
excel = xlrd.open_workbook("补充图斑台账(2)(1).xlsx")
sheets = excel.sheet_by_name("2013年之前修建")
values = [int(i) for i in sheets.col_values(3)[1:]]
if not os.path.exists("./dirs"): os.makedirs("./dirs")
for i in values:
	pathDir = os.path.join("./dirs", str(i))
	if not os.path.exists(pathDir):
		os.makedirs(pathDir)
	pg.click(find_pos[0], find_pos[1], 2)  # 点击查找
	pg.typewrite(str(i))
	pg.hotkey("enter")
	time.sleep(0.5)
	# 点击详情 详情位置也作为全局变量
	pg.click(particulars[0], particulars[1])
	# ----------------------------------------------------------
	# ----------------- 网络慢 择增加等待时间 ---------------
	time.sleep(3.)  # 点击详情后 等待几秒 加载地图
	# 获取截图左上角坐标
	image_particulars = pg.screenshot(region=(leftTop[0], leftTop[1], imagesize[0], imagesize[1]))  # -60 减掉任务栏
	image_particulars.save(os.path.join(pathDir, "详情情况.png"))  # 截图详细信息
	# 点击 外业图像
	time.sleep(0.5)
	pg.click(field_work[0], field_work[1])
	time.sleep(0.5)
	pg.click(image1[0], image1[1])
	image_particulars = pg.screenshot(region=(leftTop[0], leftTop[1], imagesize[0], imagesize[1]))  # -60 减掉任务栏
	image_particulars.save(os.path.join(pathDir, "外业1.png"))  # 截图详细信息
	# 点击返回切换第二张图像
	pg.click(getbackpos[0], getbackpos[1])
	time.sleep(0.5)
	# 点击 第二张图像
	pg.click(image2[0], image2[1])
	image_particulars = pg.screenshot(region=(leftTop[0], leftTop[1], imagesize[0], imagesize[1]))  # -60 减掉任务栏
	image_particulars.save(os.path.join(pathDir, "外业2.png"))  # 截图详细信息
	time.sleep(0.5)
	# 点击返回列表
	pg.click(getbacklist[0], getbacklist[1])
	time.sleep(0.5)
pg.alert(text='完成截图', title='恭喜', button='OK')