import math
import os
import time
import tkinter as tk
from threading import Thread
from tkinter import messagebox
from tkinter.filedialog import *
from tkinter.ttk import Combobox

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torch.nn import functional as F
from torchvision.transforms import transforms as tf

from model import conform, effic

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 下面老是报错 shape 不一致


class GuiModel(object):
	def __init__(self):
		super(GuiModel, self).__init__()
		
		self.modelclass2 = None
		self.modelclass1 = None
		self.modelclass0 = None
		self.root = tk.Tk()
		self.root.geometry("1000x800+100+100")
		self.res = {}
		self.imageFile = tk.StringVar()  # image path
		self.imageSave = tk.StringVar()  # word save
		self.batchSize = tk.IntVar()  #
		self.batchSize.set(4)
		self.device = tk.StringVar()
		self.deviceList = ("cuda", "cpu", "mps")
		
		self.fig2 = plt.figure(dpi=150)
		self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.root)
		self.canvas2.get_tk_widget().place(x=10, y=60)
		self.imageShowList = []
		self.imageShowLists = []
		self.imageShowIndex = -1
		self.drawShow = True
		# 定义核伤类型
		self.dictAll = {
				"00": " 轨头内侧核伤", "01": "轨头中央核伤", "02": "轨头外侧核伤",
				"10": "螺孔水平裂纹", "11": " 螺孔斜上裂纹", "12": " 螺孔斜下裂纹",
				"20": "轨底伤损", "21": "其他裂纹"
		}
	
	# 显示图像
	def drawImage(self):
		self.fig2.clf()
		row = math.ceil(math.sqrt(self.batchSize.get()))
		# cols =max( self.batchSize.get()//row,1)
		if self.imageShowIndex >= len(self.imageShowLists):
			self.imageShowIndex = 0
		if self.imageShowLists and self.drawShow:
			for i in range(len(self.imageShowLists)):
				fig = self.fig2.add_subplot(row, row, i + 1)
				fig.axis("off")
				fig.imshow(self.imageShowLists[i])
				self.imageShowIndex += 1
		
		if self.drawShow: self.root.after(1000, self.drawImage)
		self.canvas2.draw()
	
	def tkFram(self):
		file = tk.Button(self.root, text='打开文件', borderwidth=2, font=("", 15),
		                 command=lambda: self.choiseDir(self.imageFile))
		file.place(x=50, y=20)
		save = tk.Button(self.root, text='保存路径', borderwidth=2, font=("", 15),
		                 command=self.addModel)
		pred = tk.Button(self.root, text='推理计算', borderwidth=2, font=("", 15), command=lambda: self.thred(self.preds()))
		deviceLabel = tk.Button(self.root, borderwidth=2, font=("", 15), text='设备选择:')
		deviceChoise = Combobox(self.root, values=self.deviceList, textvariable=self.device, width=5, font=("", 15))
		deviceChoise.current(0)
		batchSizeLable = tk.Button(self.root, text='BatchSize:', font=("", 15))
		batchChoise = tk.Entry(self.root, textvariable=self.batchSize, font=("", 15))
		self.lable = tk.Label(self.root, text='加载模型中...', font=("宋体", 15))
		self.lable.place(x=200, y=300)
		file.place(x=50, y=20)
		save.place(x=200, y=20)
		pred.place(x=400, y=20)
		deviceLabel.place(x=550, y=20)
		deviceChoise.place(x=670, y=25)
		batchSizeLable.place(x=770, y=20)
		batchChoise.place(x=900, y=25)
	
	def on_closing(self):
		
		self.root.quit()
	
	def choiseDir(self, var):
		var_ = askdirectory()
		var.set(var_)
		self.imageShowList = [plt.imread(os.path.join(var_, i)) for i in os.listdir(var_)]
	
	def resultsData(self, pred, classTitle):
		P = F.softmax(pred, dim=1) * 100
		P = torch.round(P) / 100  # 取得概率
		classes = torch.argmax(P, dim=1)  # 取得预测类别
		reTemp = {}
		try:
			for index in range(len(classTitle)):
				p = P[index, classes[index].item()].item()
				# if p < 0.7:
				# 	reTemp[classTitle[index]] = {classes[index].item(): f"概率为{p}小于0.7，疑似无故障"}
				# else:
				reTemp[classTitle[index]] = {classes[index].item(): p}
		
		except:
			reTemp[classTitle[0]] = {classes.item(): P[classes.item()].item()}
		return reTemp
	
	def oneImage(self, imagedata, image_paths: list, model1, **kwargs):
		# 定义返回信息
		result = {}
		# 数据预处理
		images = []
		self.fig2.clf()
		for index, image in enumerate(imagedata):
			# image = plt.imread(image_path)
			image = cv2.resize(image, (kwargs['size']))  # 重采样
			if image.dtype == np.uint8:
				image = (image / 255).astype(np.float32)
			image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # totensor
			norm = tf.Normalize(kwargs['mean'], kwargs['std'])  # 归一化
			image = norm(image[:, :3])
			images.append(image)
		if len(images) == 1:
			images.append(image)
			image_paths += image_paths
		images = torch.cat(images, dim=0)
		images = images.to(kwargs['device'])
		# 网络运算
		out = model1(images)  # 三大类
		out = out.cpu()
		model1.cpu()
		out = torch.argmax(out, dim=1)
		
		# 三大类 非三大类则是没有错误
		class0 = images[out[out == 0]]
		class1 = images[out[out == 1]]
		class2 = images[out[out == 2]]
		# 获取正确类
		classTrue = out[out != 0]
		classTrue = classTrue[classTrue != 1]
		classTrue = classTrue[classTrue != 2]
		for i in classTrue:
			result[images[i]] = "图像无故障问题"
		class0Tilte = []
		class1Tilte = []
		class2Tilte = []
		for i in range(len(out)):
			if out[i].item() == 0:  # 第 0 类
				class0Tilte.append("0-" + os.path.basename(image_paths[i].replace('.png', '')))
			elif out[i].item() == 1:
				class1Tilte.append("1-" + os.path.basename(image_paths[i].replace('.png', '')))
			elif out[i].item() == 2:
				class2Tilte.append("2-" + os.path.basename(image_paths[i].replace('.png', '')))
		
		# 第一个小类
		if len(class0):
			self.modelclass0.to(kwargs['device'])
			self.modelclass0.eval()
			outclass0 = self.modelclass0(class0)
			res0 = self.resultsData(outclass0, class0Tilte)  # 拿到概率和类别
			result = {**result, **res0}
			self.modelclass0.cpu()
		if len(class1):
			self.modelclass1.to(kwargs['device'])
			self.modelclass1.eval()
			outclass1 = self.modelclass1(class1)
			res1 = self.resultsData(outclass1, class1Tilte)  # 拿到概率和类别
			result = {**result, **res1}
			self.modelclass1.cpu()
		if len(class2):
			self.modelclass2.to(kwargs['device'])
			outclass2 = self.modelclass2(class2)
			res2 = self.resultsData(outclass2, class2Tilte)  # 拿到概率和类别
			result = {**result, **res2}
			self.modelclass2.cpu()
		return result
	
	def preds(self):
		assert self.modelclass0 and self.modelclass1 and self.modelclass2, messagebox.showerror('erro', 'place add model')
		device = torch.device(self.device.get())
		self.drawShow = True
		model1 = conform.conform(1000)
		# load_dict = torch.load('./checkpoint/comform_threes.pth')
		# model1.load_state_dict(load_dict)
		ret = {}
		# imagedir = r'D:\temps\裂隙识别\image\三大类\核伤'
		star = time.time()
		imagedir = self.imageFile.get()
		imagedirList = os.listdir(imagedir)  # 进行batch_size 定义 每次2个
		with torch.no_grad():
			for index in range(0, len(imagedirList), self.batchSize.get()):
				try:
					imageBatchSize = imagedirList[index:index + self.batchSize.get()]
				except:
					imageBatchSize = imagedirList[index:]
				model1.to(device)
				self.imageShowLists = [plt.imread(os.path.join(self.imageFile.get(), i))[:, :, :3] for i in imageBatchSize]
				temp = self.oneImage(self.imageShowLists,
				                     imageBatchSize,
				                     model1,
				                     size=(256, 256),
				                     device=device,
				                     mean=[0.911, 0.907, 0.782], std=[0.151, 0.154, 0.144])
				ret = {**ret, **temp}
			for k, v in ret.items():
				maxp = k.split('-')[0]
				minp, p = list(v.keys())[0], list(v.values())[0]
				self.res[k.split('-')[1]] = {"损伤类型": self.dictAll[f"{maxp}{minp}"], "概率": f'{p:.3f}'}
			self.res['使用时间'] = f'{time.time() - star:.3f}S'
			self.res['推理速度'] = f'{(index + self.batchSize.get()) / (time.time() - star):.3f}'
		print(self.res)
	
	def addModel(self):
		
		# 读取权重未写
		loadPath = ["./checkpoint/effic_rail_head.pth",
		            "./checkpoint/effic_rail_web.pth",
		            "./checkpoint/effic_other.pth"]
		self.modelclass0 = effic.effic(3)
		self.modelclass1 = effic.effic(3)
		self.modelclass2 = effic.effic(2)
		# self.modelclass0.load_state_dict(torch.load(loadPath[0]))
		# self.modelclass1.load_state_dict(torch.load(loadPath[1]))
		# self.modelclass2.load_state_dict(torch.load(loadPath[2]))
		self.lable.destroy()
		messagebox.showinfo('提示..', '模型加载完毕')
	
	def thred(self, func):
		t = Thread(target=func)
		t.daemon = True
		t.start()
	
	# print(json.dumps(ret))
	# sys.stdout.flush()
	
	def run(self):
		self.tkFram()
		self.drawImage()
		self.thred(self.addModel())
		self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
		self.root.update()
		self.root.mainloop()


if __name__ == '__main__':
	# loads = torch.load('./checkpoint/effic_rail web.pth')
	# model2 = effic.effic(3)
	#
	# model2.load_state_dict(loads)
	# for (k1,v1),(k2,v2) in zip(loads.items(),model2.state_dict().items()):
	# 	print(k1,v1.size(),k2,v2.size())
	main = GuiModel()
	main.run()