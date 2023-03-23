# -*- coding: utf-8 -*-
# @Time    : 2022/10/16 8:38
# @Author  : HongFei Wang

# # 地图大小
# map_size = 64
# # 满意度
# linjie = 0.4
# # 空格位置
# space = int(map_size**2 * 0.1)
# people_ones = int(map_size**2 * 0.4)
#
# # 第一步随机生成位置
# people_ones_list, people_twos_list, space_list = setpoint(map_size, space,people_ones)
# maps = np.ones((map_size, map_size)) * 2
# maps = updateMap(map_size, people_ones_list, people_twos_list, maps)
# color = ['white', 'green', 'yellow']
# plt.figure(figsize=(8, 8), dpi=100)
# for i in range(1000):
#     Ones, Twos, Space, Maps, manyidu = randomMove(people_ones_list, people_twos_list, space_list, linjie, maps)
#
#     if manyidu is None:
#         break
#     plt.clf()
#     plt.title(f'{i + 1}-{manyidu * 100:.2f}%')
#     maps = updateMap(map_size, people_ones_list, people_twos_list, maps)
#     plt.imshow(maps, cmap=matplotlib.colors.ListedColormap(color))
#     plt.pause(1)
# plt.imshow(maps)
# plt.show()


"""
aaa
"""
import os
import sys
import time
from configparser import ConfigParser

import pymysql

# 读取ini配置文件的过程
dbpath = os.getcwd() + '\\dbs.ini'
# dbcity=os.getcwd()+'\\db100config\\citydbconfig.ini'
target = ConfigParser()
target.read(dbpath, encoding='utf-8')

getip = target.get('dbconnection', 'ip')
getname = target.get('dbconnection', 'name')
getpassword = target.get('dbconnection', 'password')
dbname = target.get('dbconnection', 'dbname')

# 数据库连接检查
try:
	db = pymysql.connect(host=getip, user=getname, passwd=getpassword, db=dbname)
	print('连接成功')
except:
	print("连接数据库失败，请检查网络！")
	time.sleep(2)
	sys.exit()