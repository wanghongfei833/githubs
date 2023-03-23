# -*- coding: utf-8 -*-
# @Time    : 2023/2/6 19:56
# @Author  : HongFei Wang
import tkinter
from tkinter import *

master = Tk()
strs = tkinter.StringVar()
strs.set("this is a message")
w = Message(master, textvariable=strs)

w.pack()

mainloop()