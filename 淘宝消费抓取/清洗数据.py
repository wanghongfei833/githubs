# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 14:15
# @Author  : HongFei Wang
import re

import xlwt

name = '王洪飞'
wook = xlwt.Workbook()
sheet_wang = wook.add_sheet(f'{name}')
paths = fr'{name}.txt'
find_money = re.compile('(\d*)元')
f = open(paths, 'r')
ress = []
sheet_wang.write(0, 0, '订单号')
sheet_wang.write(0, 1, '产品')
sheet_wang.write(0, 2, '价格')

rows = 0
for lines in f.readlines():
	
	lines = lines.strip('\n')
	if '跳转' in lines: continue
	if not lines: continue
	lines = lines.split('\t')
	if '交易关闭' in lines:
		continue
	temp_list = []
	temp_list.append(lines[0].split(' ')[0])  # 订单号
	temp_list.append(lines[1].strip(' [交易快照]'))  # 产品
	for col in range(len(lines)):
		if '含运费' in lines[col]:
			temp_list.append(lines[col - 1].strip('￥'))
			break
	if len(temp_list) == 2:
		res = re.findall(find_money, ''.join(temp_list))
		try:
			temp_list.append(res[0])
		except:
			temp_list.append('0')
	if temp_list[0]:
		rows += 1
		sheet_wang.write(rows, 0, temp_list[0])
		sheet_wang.write(rows, 1, temp_list[1])
		sheet_wang.write(rows, 2, float(temp_list[2]))

wook.save(f'{name}淘宝消费信息.xls')