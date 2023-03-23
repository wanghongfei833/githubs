# -*- coding: utf-8 -*-
# @Time    : 2023/2/20 11:47
# @Author  : HongFei Wang
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# 深入第一层
def DNet_instance(config):
	# 定义基本参数
	"""
	net_type, \ 网络类型
		dataset, \数据格式
		IMG_size, \图像大小
		lr_base, \ 学习率
		batch_size, \ 批次
		nClass, \ 多少类 10class
		nLayer = \ 多少层 结构10
		config.net_type, \ 基础数据
			config.data_set, \
			config.IMG_size, \
			config.lr_base, \
			config.batch_size, \
			config.nClass, \
			config.nLayer
	:param config:
	:return:
	"""
	net_type, \
		dataset, \
		IMG_size, \
		lr_base, \
		batch_size, \
		nClass, \
		nLayer = \
		config.net_type, \
			config.data_set, \
			config.IMG_size, \
			config.lr_base, \
			config.batch_size, \
			config.nClass, \
			config.nLayer
	if net_type == "BiDNet":
		lr_base = 0.01
	if dataset == "emnist":
		lr_base = 0.01
	# 第二层网络结构 返回基础信息的集合 {} 字典类型
	config_base = DNET_config(batch=batch_size, lr_base=lr_base)
	# if 拿来判断网络
	if hasattr(config, 'feat_extractor'):
		config_base.feat_extractor = config.feat_extractor
	env_title = f"{net_type}_{dataset}_{IMG_size}_{lr_base}_{config_base.env_title()}"
	if net_type == "MF_DNet":
		freq_list = [0.3e12, 0.35e12, 0.4e12, 0.42e12]
		env_title = env_title + f"_C{len(freq_list)}"
	if net_type == "BiDNet":
		config_base = DNET_config(batch=batch_size, lr_base=lr_base, chunk="binary")
	
	if net_type == "cnn":
		model = Mnist_Net(config=config_base)
		return env_title, model
	# --------------调用此处网络-------------------
	# 第三层解释
	if net_type == "DNet":
		model = D2NNet(IMG_size, nClass, nLayer, config_base)
	
	# model.double()
	
	return env_title, model


# 第二层 返回基础配置信息
class DNET_config:
	def __init__(self, batch, lr_base, modulation="phase", init_value="random", random_seed=42,
	             support=SuppLayer.SUPP.exp, isFC=False):
		"""
		
		:param batch: 批次
		:param lr_base:
		:param modulation:
		:param init_value:
		:param random_seed:
		:param support: 'exp'
		:param isFC:
		:returns : 返回 字典
		"""
		self.custom_legend = "Express Wavenet"  # "Express_OFF"  "Express Wavenet","Pan_OFF Express_OFF"    #for paper and debug
		self.seed = random_seed
		# 随机种子
		seed_everything(self.seed)
		
		self.init_value = init_value  # "random"  "zero" # 初始化方法
		self.rDrop = 0
		self.support = support  # None
		self.modulation = modulation  # ["phase","phase_amp"]
		self.output_chunk = "2D"  # ["1D","2D"]
		self.output_pooling = "max"
		self.batch = batch
		self.learning_rate = lr_base
		self.isFC = isFC
		self.input_scale = 1
		self.wavelet = None  # dict paramter for wavelet
		# if self.isFC == True:            self.learning_rate = lr_base/10
		self.input_plane = ""  # "fourier"
	
	#  以下函数皆是 命名类 就是修改名字
	
	def env_title(self):
		title = f"{self.support.value}"
		if self.isFC:       title += "[FC]"
		if self.custom_legend is not None:
			title += f"_{self.custom_legend}"
		return title
	
	def __repr__(self):
		main_str = f"lr={self.learning_rate}_ mod={self.modulation} input={self.input_scale} detector={self.output_chunk} " \
		           f"support={self.support}"
		if self.isFC:       main_str += " [FC]"
		if self.custom_legend is not None:
			main_str = main_str + f"_{self.custom_legend}"
		return main_str


# 第三层 创建网络 并定义forward（正向传播)
class D2NNet(nn.Module):
	# 多分类损失函数
	@staticmethod
	def binary_loss(output, target, reduction='mean'):
		nSamp = target.shape[0]
		nGate = output.shape[1] // 2
		loss = 0
		for i in range(nGate):
			target_i = target % 2
			val_2 = torch.stack([output[:, 2 * i], output[:, 2 * i + 1]], 1)
			
			loss_i = F.cross_entropy(val_2, target_i, reduction=reduction)
			loss += loss_i
			target = (target - target_i) / 2
		
		# loss = F.nll_loss(output, target, reduction=reduction)
		return loss
	
	# 二分类损失函数
	@staticmethod
	def logit_loss(output, target,
	               reduction='mean'):  # https://stackoverflow.com/questions/53628622/loss-function-its-inputs-for-binary-classification-pytorch
		nSamp = target.shape[0]
		nGate = output.shape[1]
		loss = 0
		loss_BCE = nn.BCEWithLogitsLoss()
		for i in range(nGate):
			target_i = target % 2
			out_i = output[:, i]
			loss_i = loss_BCE(out_i, target_i.double())
			loss += loss_i
			target = (target - target_i) / 2
		return loss
	
	# 预测函数（二分类、多分类、不同计算）
	def predict(self, output):
		if self.config.support == "binary":
			nGate = output.shape[1] // 2
			pred = 0
			for i in range(nGate):
				no = 2 * (nGate - 1 - i)
				val_2 = torch.stack([output[:, no], output[:, no + 1]], 1)
				pred_i = val_2.max(1, keepdim=True)[1]  # get the index of the max log-probability
				pred = pred * 2 + pred_i
		elif self.config.support == "logit":
			nGate = output.shape[1]
			# assert nGate == self.n
			pred = 0
			for i in range(nGate):
				no = nGate - 1 - i
				val_2 = F.sigmoid(output[:, no])
				pred_i = (val_2 + 0.5).long()
				pred = pred * 2 + pred_i
		else:
			pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
		# pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
		return pred
	
	def GetLayer_(self):
		# layer = DiffractiveAMP
		if self.config.wavelet is None:
			# -------------------结构层-- 第四层介绍--------------------------
			layer = DiffractiveLayer
		else:
			layer = DiffractiveWavelet
		return layer
	
	# 初始化函数 定义网络基本参数
	def __init__(self, IMG_size, nCls, nDifrac, config):
		"""
		
		:param IMG_size: 图像大小
		:param nCls: 种类 10类
		:param nDifrac: 结构层数
		:param config: 配置信息
		"""
		super(D2NNet, self).__init__()
		self.M, self.N = IMG_size
		# 这里的Z 是自写的一系列封装函数
		# 比如 sigmod函数之类的
		
		self.z_modulus = Z.modulus
		self.nDifrac = nDifrac
		self.nClass = nCls
		self.config = config
		self.title = f"DNNet"
		self.highWay = 1  # 1,2,3
		if self.config.input_plane == "fourier":
			self.highWay = 0
		# 做一些基础的判断 确定属性
		if hasattr(self.config, 'feat_extractor'):
			if self.config.feat_extractor != "last_layer":
				self.feat_extractor = []
		# 判断信息是否错误
		if self.config.output_chunk == "2D":
			assert (self.M * self.N >= self.nClass)
		else:
			assert (self.M >= self.nClass and self.N >= self.nClass)
		print(f"D2NNet nClass={nCls} shape={self.M, self.N}")
		
		layer = self.GetLayer_()  # 定义网络变量
		#  添加网络层
		self.DD = nn.ModuleList([
				layer(self.M, self.N, config) for i in range(self.nDifrac)
		])
		if self.config.input_plane == "fourier":  # 默认不插入
			#  添加 傅里叶变换层 FFT——layer对数据进行傅里叶变换
			#  最开始插入
			#  a = [1,2,3,4] -->[fft,1,2,3,4]
			self.DD.insert(0, FFT_Layer(self.M, self.N, config, isInv=False))
			
			# 结尾插入
			#  a = [1,2,3,4] -->[fft,1,2,3,4,fft]
			self.DD.append(FFT_Layer(self.M, self.N, config, isInv=True))
		self.nD = len(self.DD)
		self.laySupp = None
		
		if self.highWay > 0:
			# 定义一个可训练参数 并初始化（正态分布）
			
			self.wLayer = torch.nn.Parameter(torch.ones(len(self.DD)))
			if self.highWay == 2:
				self.wLayer.data.uniform_(-1, 1)
			elif self.highWay == 1:
				self.wLayer = torch.nn.Parameter(torch.ones(len(self.DD)))
		
		# self.DD.append(DropOutLayer(self.M, self.N,drop=0.9999))
		# 判断最后一层输出是什么 （全连接 还是其他方式,默认全连接 下面不管）
		if self.config.isFC:
			# 拿到 最后一层 以及损失函数
			self.fc1 = nn.Linear(self.M * self.N, self.nClass)
			self.loss = UserLoss.cys_loss
			self.title = f"DNNet_FC"
		elif self.config.support != None:
			self.laySupp = SuppLayer(config, self.nClass)
			self.last_chunk = ChunkPool(self.laySupp.nChunk, config, pooling=config.output_pooling)
			self.loss = UserLoss.cys_loss
			a = self.config.support.value
			self.title = f"DNNet_{self.config.support.value}"
		else:
			self.last_chunk = ChunkPool(self.nClass, config, pooling=config.output_pooling)
			self.loss = UserLoss.cys_loss
		# 修改名字
		if self.config.wavelet is not None:
			self.title = self.title + f"_W"
		if self.highWay > 0:
			self.title = self.title + f"_H"
		if self.config.custom_legend is not None:
			self.title = self.title + f"_{self.config.custom_legend}"
		
		'''
        BinaryChunk is pool
        elif self.config.support=="binary":
            self.last_chunk = BinaryChunk(self.nClass, pooling="max")
            self.loss = D2NNet.binary_loss
            self.title = f"DNNet_binary"
        elif self.config.support == "logit":
            self.last_chunk = BinaryChunk(self.nClass, isLogit=True, pooling="max")
            self.loss = D2NNet.logit_loss
        '''
	
	# 绘图函数
	def visualize(self, visual, suffix):
		no = 0
		for plot in visual.plots:
			images, path = [], ""
			if plot['object'] == 'layer pattern':
				path = f"{visual.img_dir}/{suffix}.jpg"
				for no, layer in enumerate(self.DD):
					info = f"{suffix},{no}]"
					title = f"layer_{no + 1}"
					if self.highWay == 2:
						a = self.wLayer[no]
						a = torch.sigmoid(a)
						info = info + f"_{a:.2g}"
					elif self.highWay == 1:
						a = self.wLayer[no]
						info = info + f"_{a:.2g}"
						title = title + f" w={a:.2g}"
					image = layer.visualize(visual, info, {'save': False, 'title': title})
					images.append(image)
					no = no + 1
			if len(images) > 0:
				image_all = np.concatenate(images, axis=1)
				# cv2.imshow("", image_all);    cv2.waitKey(0)
				cv2.imwrite(path, image_all)
	
	def legend(self):
		if self.config.custom_legend is not None:
			leg_ = self.config.custom_legend
		else:
			leg_ = self.title
		return leg_
	
	def __repr__(self):
		main_str = super(D2NNet, self).__repr__()
		main_str += f"\n========init={self.config.init_value}"
		return main_str
	
	def input_trans(self, x):  # square-rooted and normalized
		# x = x.double()*self.config.input_scale
		if True:
			x = x * self.config.input_scale
			x_0, x_1 = torch.min(x).item(), torch.max(x).item()
			assert x_0 >= 0
			x = torch.sqrt(x)
		else:  # 为何不行，莫名其妙
			x = Z.exp_euler(x * 2 * math.pi).float()
			x_0, x_1 = torch.min(x).item(), torch.max(x).item()
		return x
	
	def do_classify(self, x):
		if self.config.isFC:
			x = torch.flatten(x, 1)
			x = self.fc1(x)
			return x
		
		x = self.last_chunk(x)
		if self.laySupp != None:
			x = self.laySupp(x)
		# output = F.log_softmax(x, dim=1)
		return x
	
	def OnLayerFeats(self):
		pass
	
	def forward(self, x):
		if hasattr(self, 'feat_extractor'):
			self.feat_extractor.clear()
		nSamp, nChannel = x.shape[0], x.shape[1]
		assert (nChannel == 1)
		if nChannel > 1:
			no = random.randint(0, nChannel - 1)
			x = x[:, 0:1, ...]
		x = self.input_trans(x)
		if hasattr(self, 'visual'):            self.visual.onX(x.cpu(), f"X@input")
		summary = 0
		for no, layD in enumerate(self.DD):
			info = layD.__repr__()
			x = layD(x)
			if hasattr(self, 'feat_extractor'):
				self.feat_extractor.append((self.z_modulus(x), self.wLayer[no]))
			if hasattr(self, 'visual'):         self.visual.onX(x, f"X@{no + 1}")
			if self.highWay == 2:
				s = torch.sigmoid(self.wLayer[no])
				summary += x * s
				x = x * (1 - s)
			elif self.highWay == 1:
				summary += x * self.wLayer[no]
			elif self.highWay == 3:
				summary += self.z_modulus(x) * self.wLayer[no]
		if self.highWay == 2:
			x = x + summary
			x = self.z_modulus(x)
		elif self.highWay == 1:
			x = summary
			x = self.z_modulus(x)
		elif self.highWay == 3:
			x = summary
		elif self.highWay == 0:
			x = self.z_modulus(x)
		if hasattr(self, 'visual'):            self.visual.onX(x, f"X@output")
		
		if hasattr(self, 'feat_extractor'):
			return
		elif hasattr(self.config, 'feat_extractor') and self.config.feat_extractor == "last_layer":
			return x
		else:
			output = self.do_classify(x)
			return output


# 第四层 基础的网络结构 --> 网络就是基础的网络结构堆叠而成的
class DiffractiveLayer(torch.nn.Module):
	def SomeInit(self, M_in, N_in, HZ=0.4e12):
		assert (M_in == N_in)
		self.M = M_in
		self.N = N_in
		self.z_modulus = Z.modulus
		self.size = M_in
		self.delta = 0.03
		self.dL = 0.02
		self.c = 3e8
		self.Hz = HZ  # 0.4e12
		
		self.H_z = self.Init_H()
	
	# 定义名字之类的
	def __repr__(self):
		# main_str = super(DiffractiveLayer, self).__repr__()
		main_str = f"DiffractiveLayer_[{(int)(self.Hz / 1.0e9)}G]_[{self.M},{self.N}]"
		return main_str
	
	def __init__(self, M_in, N_in, config, HZ=0.4e12):
		super(DiffractiveLayer, self).__init__()
		self.SomeInit(M_in, N_in, HZ)
		assert config is not None
		self.config = config
		# self.init_value = init_value
		# self.rDrop = rDrop
		if not hasattr(self.config, 'wavelet') or self.config.wavelet is None:
			if self.config.modulation == "phase":  # 定义 trans 参数（可训练参数）
				self.transmission = torch.nn.Parameter(data=torch.Tensor(self.size, self.size), requires_grad=True)
			else:
				self.transmission = torch.nn.Parameter(data=torch.Tensor(self.size, self.size, 2), requires_grad=True)
			
			init_param = self.transmission.data  # 拿到刚刚定义参数的数据
			# 查看是哪种初始化 对应执行哪种函数 默认未 random
			if self.config.init_value == "reverse":  #
				half = self.transmission.data.shape[-2] // 2
				init_param[..., :half, :] = 0
				init_param[..., half:, :] = np.pi
			elif self.config.init_value == "random":
				init_param.uniform_(0, np.pi * 2)  # 初始化
			elif self.config.init_value == "random_reverse":
				init_param = torch.randint_like(init_param, 0, 2) * np.pi
			elif self.config.init_value == "chunk":
				sections = split__sections()
				for xx in init_param.split(sections, -1):
					xx = random.random(0, np.pi * 2)
	
	# self.rDrop = config.rDrop
	
	# self.bias = torch.nn.Parameter(data=torch.Tensor(1, 1), requires_grad=True)
	
	def visualize(self, visual, suffix, params):
		param = self.transmission.data
		name = f"{suffix}_{self.config.modulation}_"
		return visual.image(name, param, params)
	
	def share_weight(self, layer_1):
		tp = type(self)
		assert (type(layer_1) == tp)
	
	# del self.transmission
	# self.transmission = layer_1.transmission
	# 将数据进行傅里叶变换 随后返回傅里叶变换的虚部实部 [real,imag]
	def Init_H(self):
		# Parameter
		N = self.size
		df = 1.0 / self.dL
		d = self.delta
		lmb = self.c / self.Hz
		k = np.pi * 2.0 / lmb
		D = self.dL * self.dL / (N * lmb)
		
		# phase
		def phase(i, j):
			i -= N // 2
			j -= N // 2
			return (i * df) * (i * df) + (j * df) * (j * df)
		
		ph = np.fromfunction(phase, shape=(N, N), dtype=np.float32)
		# H
		H = np.exp(1.0j * k * d) * np.exp(-1.0j * lmb * np.pi * d * ph)
		H_f = np.fft.fftshift(H) * self.dL * self.dL / (N * N)
		# print(H_f);    print(H)
		H_z = np.zeros(H_f.shape + (2,))
		H_z[..., 0] = H_f.real
		H_z[..., 1] = H_f.imag
		# H_z = torch.from_numpy(H_z).cuda()
		return H_z
	
	def Diffractive_(self, u0, theta=0.0):
		# 查询最后维度是否为2  IninH 后 最后维度是2
		if Z.isComplex(u0):
			z0 = u0
		else:
			z0 = u0.new_zeros(u0.shape + (2,))
			z0[..., 0] = u0
		
		N = self.size
		df = 1.0 / self.dL
		# 傅里叶变换
		z0 = Z.fft(z0)
		# print(z0)
		# print(self.H_z)
		# 变换后数据点乘
		u1 = Z.Hadamard(z0, self.H_z)
		u2 = Z.fft(u1, "C2C", inverse=True)
		return u2 * N * N * df * df
	
	# 把实部和虚部进行sin cos 操作
	def GetTransCoefficient(self):
		'''
            eps = 1e-5; momentum = 0.1; affine = True

            mean = torch.mean(self.transmission, 1)
            vari = torch.var(self.transmission, 1)
            amp_bn = torch.batch_norm(self.transmission,mean,vari)
        :return:
        '''
		amp_s = Z.exp_euler(self.transmission)
		
		return amp_s
	
	def forward(self, x):
		# 傅里叶变换后 点乘
		diffrac = self.Diffractive_(x)
		# 把实部和虚部进行sin cos 操作
		amp_s = self.GetTransCoefficient()
		# 批处理)张量A和张量B之间的复点乘。 由于阿达玛乘积是可交换的，所以Hadamard(A, B)=Hadamard(B, A)
		x = Z.Hadamard(diffrac, amp_s.float())
		if (self.config.rDrop > 0):  # Drop 效果
			drop = Z.rDrop2D(1 - self.rDrop, (self.M, self.N), isComlex=True)
			x = Z.Hadamard(x, drop)
		# x = x+self.bias
		return x