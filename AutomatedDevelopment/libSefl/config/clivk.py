# -*- coding: utf-8 -*-
# @Time    : 2022/12/27 10:56
# @Author  : HongFei Wang

import PyHook3 as pyHook
import pythoncom  # 没这个库的直接pip install pywin32安装


# 鼠标事件处理函数
def OnMouseEvent(event):
	print('MessageName:', event.MessageName)  # 事件名称
	# print('Message:', event.Message)  # windows消息常量
	# print('Time:', event.Time)  # 事件发生的时间戳
	print('Window:', event.Window)  # 窗口句柄
	print('WindowName:', event.WindowName)  # 窗口标题
	print('Position:', event.Position)  # 事件发生时相对于整个屏幕的坐标
	print(type(event.Position), event.Position[0], event.Position[1])
	print('Wheel:', event.Wheel)  # 鼠标滚轮
	# print('Injected:', event.Injected)  # 判断这个事件是否由程序方式生成，而不是正常的人为触发。
	print('---')
	
	# 返回True代表将事件继续传给其他句柄，为False则停止传递，即被拦截
	return True


# 键盘事件处理函数
def OnKeyboardEvent(event):
	# 同上
	return True


# 1
def main():
	#  创建管理器
	hm = pyHook.HookManager()
	# 监听鼠标
	# hm.MouseLeftDown 是将“鼠标左键按下”这一事件和func这个函数绑定，即每次鼠标左键按下都会执行func
	# 如果希望监测鼠标print('MessageName:', event.MessageName)  # 同上，共同属性不再赘述
	# 	# print('Message:', event.Message)
	# 	# print('Time:', event.Time)
	# 	print('Window:', event.Window)
	# 	print('WindowName:', event.WindowName)
	# 	print('Ascii:', event.Ascii, chr(event.Ascii))  # 按键的ASCII码
	# 	print('Key:', event.Key)  # 按键的名称
	# 	# print('KeyID:', event.KeyID)  # 按键的虚拟键值
	# 	# print('ScanCode:', event.ScanCode)  # 按键扫描码
	# 	# print('Extended:', event.Extended)  # 判断是否为增强键盘的扩展键
	# 	# print('Injected:', event.Injected)
	# 	# print('Alt', event.Alt)  # 是某同时按下Alt
	# 	# print('Transition', event.Transition)  # 判断转换状态
	# 	print('---')中键按下则：hm.MouseMiddleDown，鼠标右键按下则：hm.MouseRightDown
	# hm.MouseAllButtonsDown = OnMouseEvent  # 将OnMouseEvent函数绑定到MouseAllButtonsDown事件上
	# hm.MouseAllButtonsUp = OnMouseEvent  # 将OnMouseEvent函数绑定到MouseAllButtonsDown事件上
	# hm.MouseLeftDown = OnMouseEvent
	hm.KeyDown = OnKeyboardEvent  # 将OnKeyboardEvent函数绑定到KeyDown事件上
	hm.HookMouse()
	hm.HookKeyboard()
	
	# 循环监听
	pythoncom.PumpMessages()


if __name__ == "__main__":
	main()