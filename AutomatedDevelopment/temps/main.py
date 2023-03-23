# -*- coding: utf-8 -*-
# @Time    : 2022/12/29 17:31
# @Author  : HongFei Wang
import tkinter as tk


class Main(object):
	def __init__(self, width=600, height=400, widthadd=200, heightadd=200):
		self.root = tk.Tk()
		self.root.title("自动化")
		self.root.geometry(f"{width}x{height}+{widthadd}+{heightadd}")
		# 定义文件
		self.pythonTxt = None
		self.FarmsList = [
				f"# -*- coding: utf-8 -*-\n"
				f"# @Author  : HongFei Wang\n"
				f"import pyautogui as pag"
		]
		self.blocks = []
		self.posX = []
		self.posY = []
		self.button = []
		self.keyboard = []
		self.index = -1
	
	# self.posx = tk.IntVar()
	# self.posy = tk.IntVar()
	# self.button = tk.StringVar()
	
	def addMouseClick(self):
		self.posX.append(tk.IntVar())
		self.posY.append(tk.IntVar())
		self.button.append(tk.StringVar())
		self.index += 1
		self.blocks.append(
				{"name": "click",
				 "posx": self.posX[self.index].get(),
				 "posy": self.posY[self.index].get(),
				 "button": self.button[self.index].get()}
		)
	
	def writeMouseClick(self):
		self.FarmsList.append(f"pag.click({self.posX[self.index].get()},"
		                      f"{self.posY[self.index].get()},"
		                      f"button='{self.button[self.index].get()}')")
	
	def setBlcok(self):
		for rows, data in enumerate(self.blocks):
			if data["name"] == "click":  # 点击
				tk.Label(self.root, text="点击").place(x=20, y=rows * 20)
				tk.Label(self.root, text="x").place(x=20, y=rows * 20)
				tk.Entry(self.root, textvariable=self.)
				tk.Label(self.root, text="y").place(x=20, y=rows * 20)
				tk.Label(self.root, text="鼠标键位").place(x=20, y=rows * 20)
			
			elif data["name"] == "drat":  # 拖拽
				pass
			elif data["name"] == "keyboard":  # 按键输入
				pass
			elif data["name"] == "group":  # 按键组合
				pass