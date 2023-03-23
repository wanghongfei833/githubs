# -*- coding: utf-8 -*-
# @Time    : 2022/11/17 11:07
# @Author  : HongFei Wang

import tqdm
import xlwt
from selenium import webdriver
from selenium.webdriver.common.by import By


def down_load_Data(browser: webdriver.Chrome, sheet, classes: str):
	ranks = browser.find_elements(By.XPATH,
	                              '//div[@class="view-sota-table show-overflow-x"]/table[1]/tbody[1]'
	                              '//tr/td[1]')
	names = browser.find_elements(By.XPATH,
	                              '//div[@class="view-sota-table show-overflow-x"]/table[1]/tbody[1]'
	                              '//tr/td[2]/div[1]')
	top1 = browser.find_elements(By.XPATH,
	                             '//div[@class="view-sota-table show-overflow-x"]/table[1]/tbody[1]'
	                             '//tr/td[3]')
	top5 = browser.find_elements(By.XPATH,
	                             '//div[@class="view-sota-table show-overflow-x"]/table[1]/tbody[1]'
	                             '//tr/td[4]')
	params = browser.find_elements(By.XPATH,
	                               '//div[@class="view-sota-table show-overflow-x"]/table[1]/tbody[1]'
	                               '//tr/td[5]')
	
	floaps = browser.find_elements(By.XPATH,
	                               '//div[@class="view-sota-table show-overflow-x"]/table[1]/tbody[1]'
	                               '//tr/td[6]')
	paper_data = browser.find_elements(By.XPATH,
	                                   '//div[@class="view-sota-table show-overflow-x"]/table[1]/tbody[1]'
	                                   '//tr/td[8]/div[1]/a[1]')  # 获取contesxt href
	# paper_data_paper_name = [i.text for i in paper_data]
	# paper_data_paper_href = [i.get_attribute('href') for i in paper_data]
	# title = ['rank','name','top1','top5','params','floaps','paper_name','paper_href']
	print('开始写入')
	for index in tqdm.tqdm(range(len(names)), total=min(200, len(names))):
		sheet.write(index + 1, 0, 'rank')
		sheet.write(index + 1, 1, ranks[index].text)
		sheet.write(index + 1, 2, names[index].text)
		sheet.write(index + 1, 3, 'top1')
		sheet.write(index + 1, 4, top1[index].text)
		sheet.write(index + 1, 5, "top5")
		sheet.write(index + 1, 6, top5[index].text)
		sheet.write(index + 1, 7, 'parmars')
		sheet.write(index + 1, 8, params[index].text)
		sheet.write(index + 1, 9, 'floaps')
		sheet.write(index + 1, 10, floaps[index].text)
		sheet.write(index + 1, 11, 'paper_name')
		sheet.write(index + 1, 12, paper_data[index].text)
		sheet.write(index + 1, 13, "paper_href")
		sheet.write(index + 1, 14, xlwt.Formula(paper_data[index].get_attribute('href')))
		if index > 200: break
	return None


headers = {
		'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36'
}

urls = 'https://paperswithcode.com/sota/image-classification-on-imagenet'

chrome_option = webdriver.ChromeOptions()
chrome_option.add_argument('--window-size=400,1000')  # 设置窗口界面大小
# chrome_option.add_argument('--headless')  # 启用无头模式
chrome_option.add_argument('--disable-gpu')
browser = webdriver.Chrome(options=chrome_option)
browser.set_page_load_timeout(30)
browser.set_script_timeout(30)
try:
	print("打开网页中")
	browser.get(urls)
except:
	print("超过10秒开始下拉")
	browser.execute_script("window.stop()")
# 滑动到底部
js_bottom = "window.scrollTo(0,document.body.scrollHeight)"
browser.execute_script(js_bottom)
returns = {}
# down_load_Data(browser,returns)
wook = xlwt.Workbook()
sheet_top1 = wook.add_sheet('top1')
sheet_top5 = wook.add_sheet('top5')

dis1 = down_load_Data(browser, sheet_top1, 1)
if dis1:
	print('出错top1', "\t".join(dis1))
searchButtonElement = browser.find_element(By.XPATH, '//*[@id="leaderboard"]/div[3]/table/thead/tr/th[4]/span')  # 点击跳转
browser.execute_script("$(arguments[0]).click()", searchButtonElement)
# ActionChainsDriver = ActionChains(browser).click(searchButtonElement)
dis2 = down_load_Data(browser, sheet_top5, 2)
print('出错top5', dis2)
wook.save('Ranks.xls')
# //*[@id="leaderboard"]/div[3]/table/thead/tr/th[4]/span

browser.close()