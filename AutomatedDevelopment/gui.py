# -*- coding: utf-8 -*-
# @Time    : 2022/12/27 21:40
# @Author  : HongFei Wang
import tkinter as tk
from tkinter import *
from tkinter.filedialog import *

# from gets import gets
import PyHook3 as pyHook

datasets = {
		"Lmenu": "altleft",
		"Rmenu": "altright",
		"Lcontrol": "ctrlleft",
		"Rcontrol": "ctrlright",
		"Return": "enter",
		"Back": "backspace",
}


class UI(object):
	def __init__(self, width=600, height=400, widthadd=200, heightadd=200):
		self.win = tk.Tk()
		self.win.title("自动化")
		self.win.geometry(f"{width}x{height}+{widthadd}+{heightadd}")
		self.pythonTxt = None
		self.posTemp = None
		self.saves = tk.StringVar()
		self.hm = pyHook.HookManager()  # 监听事件
		self.starPos = None  # 判断 拖拽是否发生
		self.moves = False  # 当前是否为拖拽事件
		
		self.MouseMess = None  # 鼠标事件
		self.KyeBordMess = None  # 键盘事件
		self.MouseDrat = False  # 鼠标拖拽
		self.MouseClick = False  # 鼠标点击判断
		self.KeyBordPred = False  # 键盘输入
		
		"""
		整体思路为：
		1、 一直监听事件
		2、 当出现增加点击事件时 ADDS为T 且点击事件函数开始监听
		3、 当出现增加拖拽事件时 ADDS为T 且拖拽事件函数监听
		4、 点击F12时 ADDS为F
		"""
	
	def complate(self):
		# 初始化信息
		self.MouseDrat = False  # 鼠标拖拽
		self.MouseClick = False  # 鼠标点击判断
		self.KeyBordPred = False  # 键盘输入
		self.MouseMess = None  # 鼠标事件
		self.KyeBordMess = None  # 键盘事件
		print("完成操作")
	
	def changeMouseClick(self):
		self.MouseClick = True
	
	def changeMouseDrat(self):
		self.MouseDrat = True
	
	def changeKeybord(self):
		self.KeyBordPred = True
	
	def openFile(self):
		paths = asksaveasfilename(defaultextension=".py")
		self.saves.set(paths)
		self.pythonTxt = open(paths, "w")
		self.pythonTxt.write(f"# -*- coding: utf-8 -*-\n"
		                     f"# @Author  : HongFei Wang\n"
		                     f"import pyautogui as pag\n"
		                     f"pag.sleep(2)\n")
	
	def setFarm(self):
		"""
		组件
		:return:
		"""
		Button(self.win, text="添加保存路径", command=self.openFile).place(x=0, y=0)
		Entry(self.win, textvariable=self.saves, width=60).place(x=100, y=5)
		
		Button(self.win, text="添加鼠标点击", command=self.changeMouseClick).place(x=0, y=50)
		Button(self.win, text="添加鼠标拖拽", command=self.changeMouseDrat).place(x=0, y=100)
		Button(self.win, text="添加键盘输入", command=self.changeKeybord).place(x=0, y=150)
		Button(self.win, text="添加键盘组合", ).place(x=0, y=200)
		Button(self.win, text=" 退出并保存 ", command=self.exit).place(x=0, y=300)
	
	# 键盘输入事件
	def keybordAll(self, event):
		keys = event.Key  # 拿到Key值
		if keys == "F12":
			self.complate()
		elif self.KeyBordPred:  # 是否进行按键事件
			if keys != "F12":  # 不是退出按键
				message = event.MessageName
				if keys in datasets.keys():
					keys = datasets[keys]
				if "Numpad" in keys:
					keys = keys.replace("Numpad", "")
				if "down" in message:
					self.pythonTxt.write(f"pag.keyDown('{keys}')\n")
				else:
					print(f"pag.keyUp('{keys}')\n")
					self.pythonTxt.write(f"pag.keyUp('{keys}')\n")
		
		return True
	
	# 鼠标点击事件
	def mouseClick(self, event):
		"""
		进行 单次 点击 函数
		:param event:
		:return:
		"""
		message = event.MessageName
		if self.MouseClick:
			if "move" not in message and "wheel" not in message and "down" in message:
				print("点击")
				_, lr, ud = message.split(" ")  # 获取鼠标点击左右键
				pose = event.Position  # (x,y)
				print(f"pag.click({pose[0]},{pose[1]},button='{lr}')")
				self.pythonTxt.write(f"pag.click({pose[0]},{pose[1]},button='{lr}')\n")
		return True
	
	# 鼠标拖拽
	def mouseDrat(self, event):
		"""
		鼠标拖拽函数
		:return:
		"""
		if self.MouseDrat:
			message = event.MessageName
			if "move" not in message and "wheel" not in message:
				pose = event.Position  # (x,y)
				_, lr, ud = message.split(" ")  # 获取鼠标点击左右键 获取初始坐标 点击
				if "down" in message:
					self.moves = True
					self.starPos = pose
					self.pythonTxt.write(f"pag.mouseDown({pose[0]},{pose[1]},button='{lr}')\n")
				elif self.moves and "up" in message:
					distance = (self.starPos[0] - pose[0]) ** 2 + (self.starPos[1] - pose[1]) ** 2
					if distance > 10:  # 移动一定距离才开始计算
						self.moves = False
						print("拖拽")
						self.pythonTxt.write(f"pag.moveTo({pose[0]},{pose[1]})\n")
						self.pythonTxt.write(f"pag.mouseUp(button='{lr}')\n")
		return True
	
	def mouseListen(self, event):
		self.mouseClick(event)
		self.mouseDrat(event)
		return True
	
	# 鼠标事件监听
	
	def listens(self):
		self.hm.KeyAll = self.keybordAll
		self.hm.MouseAll = self.mouseListen
		self.hm.HookKeyboard()
		self.hm.HookMouse()  # 监听
	
	def exit(self):
		self.pythonTxt.close()
		self.win.destroy()
	
	def main(self) -> object:
		"""
		主循环
		:return:
		"""
		self.listens()
		self.setFarm()
		self.win.mainloop()


mains = UI()
if __name__ == '__main__':
	mains.main()
# pag.mouseDown(424, 340,button="left")
# pag.moveTo(850, 453,2)
# pag.mouseUp(button="left")