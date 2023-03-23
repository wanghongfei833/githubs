# -*- coding: utf-8 -*-
# @Time    : 2023/2/6 11:12
# @Author  : HongFei Wang
import pymysql
from pymysql.err import *


class MySQLs(object):
	"""
	mysql 链接封装函数
	"""
	
	def __init__(self, root="root", password="717496"):
		super().__init__()
		self.root = root
		self.password = password
		# pymysql.err.OperationalError: (1045, "Access denied for user 'root'@'localhost' (using password: YES)")
		# pymysql.err.OperationalError: (1049, "Unknown database '123'")
		try:
			self.con = pymysql.connect(host='localhost', user=root, password=password, database="make_money")
			self.cursor = self.con.cursor()  # 获取游标
			print("数据库链接成功")
		except OperationalError as e:
			if '1045' in str(e):
				print("密码错误")
			elif "1049" in str(e):
				self.initTbale()
	
	#
	# 	messagebox.showerror("错误", "数据库链接失败")
	
	def initTbale(self):
		"""
		初始化数据库 完成建库 建表
		:returns None
		"""
		self.con = pymysql.connect(host='localhost', user=self.root, password=self.password)
		self.cursor = self.con.cursor()  # 获取游标
		self.cursor.execute("create database IF NOT EXISTS make_money")
		self.con = pymysql.connect(host='localhost', user=self.root, password=self.password, database="make_money")
		self.cursor = self.con.cursor()  # 获取游标
		self.cursor.execute("create TABLE IF NOT EXISTS  pending("
		                    "ID INT AUTO_INCREMENT PRIMARY KEY,"
		                    "YEAR INT(4),"
		                    "MONTH INT,"
		                    "DAY INT,"
		                    "NAME VARCHAR(3000),"
		                    "DESCRIBLE VARCHAR(10000),"
		                    "ORIGINAL INT(6),"
		                    "DOING INT(1))")
		self.con.commit()
	
	def addData(self, inputs):
		"""

		:param args: 顺序 id 年 月 日  描述 价格
		:return:
		"""
		sql = f"insert into pending(YEAR,MONTH,DAY,NAME,DESCRIBLE,ORIGINAL,DOING) VALUES {inputs}"
		# sql = "insert into pending(YEAR,MONTH,DAY,NAME,DESCRIBLE,ORIGINAL,DOING) VALUES (2022, 2, 6,'小姐','完成一个单子',500,1)"
		self.cursor.execute(sql)
		self.con.commit()
	
	def update(self, k1, v1, k2, v2):
		"""

		:param k1:
		:param v1:
		:param k2:
		:param v2:
		:return:
		"""
		sql = f"update  pending set {k1}={v1} where {k2} = {v2}"
		self.cursor.execute(sql)
		self.con.commit()
	
	def removes(self, id):
		sql = "delete from empdb.employee where eid={}".format(id)
		self.cursor.execute(sql)
		self.con.commit()
	
	def closeConn(self):
		self.con.close()
		self.cursor.close()
	
	def findData(self, sql):
		# if classes=="all":
		# 	# 查询所有信息
		# 	sql = 'select * from user WHERE gongzi>100'
		# elif classes=="compline"
		# 	sql = 'select * from pending WHERE DOING==1'
		# else:
		# 	sql = 'select * from pending WHERE DOING==0'
		self.cursor.execute(sql)
		return self.cursor.fetchall()
# mysqls = MySQLs()
# mysqls.addData(2022, 2, 6, 'jjj', '完成一个单子', 500, 1)