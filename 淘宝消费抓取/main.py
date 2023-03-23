# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 22:21
# @Author  : HongFei Wang
import time

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains  # 鼠标事件
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys  # 键盘事件

# wook = xlwt.Workbook()
# wook.add_sheet('王洪飞.xls')
f = open('save.txt', 'w')
urls = 'https://buyertrade.taobao.com/trade/itemlist/list_bought_items.htm?spm=a1z02.1.a2109.d1000368.t9rq8I&nekot=1470211439694'
chrome_option = webdriver.ChromeOptions()
chrome_option.add_argument('--window-size=1300,1000')  # 设置窗口界面大小
# chrome_option.add_argument(r'--user-data-dir=C:\Users\10095\AppData\Local\Google\Chrome\User Data')
# chrome_option.add_experimental_option('excludeSwitches', ['enable-automation'])
# chrome_option.add_argument('--headless')  # 启用无头模式
chrome_option.add_argument('--disable-gpu')
browser = webdriver.Chrome(options=chrome_option)
browser.set_page_load_timeout(5)
browser.set_script_timeout(5)
try:
	print("打开网页中")
	browser.get(urls)
except:
	print("超过10秒开始下拉")
	browser.execute_script("window.stop()")
time.sleep(50)
print('扫描完成')
temp_height = 0
down_number = 0
packgs = 38
index = 0
dissmiss = []
while True:
	index += 1
	print(f'第{index}页')
	titles = 4
	while True:
		try:
			names = browser.find_element(By.XPATH, '//*[@id="tp-bought-root"]/div[%s]' % titles)
			titles += 1
			texts = names.text
			f.write(texts.replace('\n', '\t'))
			print(texts.split('\n'))
			f.write('\n')
		except:
			print(f'第{index}页{titles - 4}条数据')
			break
	if index + 1 > packgs: break
	# browser.refresh()
	# time.sleep(5)
	browser.find_element(By.XPATH, '//div[@title="Quick jump to page"]/input').send_keys(Keys.CONTROL + 'a')  # 全选输入框
	time.sleep(0.5)
	browser.find_element(By.XPATH, '//div[@title="Quick jump to page"]/input').send_keys(Keys.BACKSPACE)  # 删除
	time.sleep(0.5)
	browser.find_element(By.XPATH, '//div[@title="Quick jump to page"]/input').send_keys('%s' % (index + 1))  # 输入页码
	time.sleep(0.5)
	searchButtonElement = browser.find_element(By.XPATH, '//div[@title="Quick jump to page"]/span[3]').click()  # 点击跳转
	ActionChainsDriver = ActionChains(browser).click(searchButtonElement)
	# ActionChainsDriver.perform()
	time.sleep(3)
browser.close()