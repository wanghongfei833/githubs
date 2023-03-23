# -*- coding: utf-8 -*-
# @Time    : 2022/12/27 10:56
# @Author  : HongFei Wang

datasets = {
		"Lmenu": "altleft",
		"Rmenu": "altright",
		"Lcontrol": "ctrlleft",
		"Rcontrol": "ctrlright",
		"Return": "enter",
		"Back": "backspace",
}
# time.sleep(1)
# pag.mouseDown(button="right")
# print("鼠标按下")
# time.sleep(3)
# pag.mouseUp(button="right")
mouseChange = False
posTemp = ()


# pag.moveTo()


def mouse_down(event, f: open):
	global mouseChange, posTemp  # 把鼠标移动改为true
	message = event.MessageName
	if "move" not in message and "wheel" not in message:
		print(message)
		_, lr, ud = message.split(" ")
		if "up" in message:  # 松开鼠标
			mouseChange = False
			if posTemp:
				f.write(f"pag.moveTo({posTemp[0]},{posTemp[1]},0.3)\n")
				posTemp = ()
			f.write(f"pag.mouseUp(button='{lr}')\n")
		if "down" in message:  # 按下鼠标
			print("按下鼠标")
			mouseChange = True
			pose = event.Position  # (x,y)
			f.write(f"pag.mouseDown({pose[0]},{pose[1]},'{lr}')\n")
	if "move" not in message and "wheel" in message:
		wheel = event.Wheel
		f.write(f"pag.scroll({wheel})\n")


# pag.scroll(x)
def mous_moves(event, f):
	if "move" in event.MessageName and mouseChange:
		global posTemp
		posTemp = event.Position  # (x,y)


def key_down_up(event, f):
	keys = event.Key  # 拿到Key值
	message = event.MessageName
	if keys in datasets.keys():
		keys = datasets[keys]
	if "Numpad" in keys:
		keys = keys.replace("Numpad", "")
	if "down" in message:
		f.write(f"pag.keyDown('{keys}')\n")
	else:
		f.write(f"pag.keyUp('{keys}')\n")