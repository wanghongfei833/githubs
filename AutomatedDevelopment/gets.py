# -*- coding: utf-8 -*-
# @Time    : 2022/12/27 12:46
# @Author  : HongFei Wang
import sys

import PyHook3 as pyHook
import pythoncom  # 没这个库的直接pip install pywin32安装

from libSefl.config import temps


def key(event, f):
	close(event, f)
	temps.key_down_up(event, f)
	return True


def mouse(event, f):
	temps.mouse_down(event, f)
	temps.mous_moves(event, f)
	return True


def close(event, f):
	if event.Key == "F12":
		f.close()
		sys.exit()


def gets(f):
	hm = pyHook.HookManager()
	hm.KeyAll = key  # 将OnKeyboardEvent函数绑定到KeyDown事件上
	hm.MouseAll = mouse
	hm.HookMouse()
	hm.HookKeyboard()
	# 循环监听
	pythoncom.PumpMessages()


if __name__ == '__main__':
	with open("try.py", "w") as f:
		f.write("""
# -*- coding: utf-8 -*-
# @Time    : 2022/12/27 12:46
# @Author  : HongFei Wang
import pyautogui as pag
pag.sleep(1)
""")
		gets()