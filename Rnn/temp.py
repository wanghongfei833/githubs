# -*- coding: utf-8 -*-
# @Time    : 2022/8/17 11:40
# @Author  : HongFei Wang
import torch
from torch import nn


class LSTM_CELL(nn.Module):
	def __init__(self, inputs, handsize):
		super(LSTM_CELL, self).__init__()
		self.ins = inputs
		self.ous = handsize
		self.rnn = nn.LSTMCell(inputs, handsize)
	
	def forward(self, x):
		x, y = x
		s, b, d = x.shape
		hx = torch.randn(b, self.ous)  # batch_size=3, hidden_size=20
		cx = torch.randn(b, self.ous)
		h = []
		c = []
		for i in range(x.size()[0]):  # 逐个单词进行计算
			hx, cx = self.rnn(x[i], (hx, cx))  # input[i]是序列中的第i个单词
			h.append(hx)  # 记录中间层的h
			c.append(cx)
		h = torch.stack(h, dim=0)  # 沿着dim=0进行concat,matrix的列表变3d tensor
		c = torch.stack(c, dim=0)
		return [h, c]


class Decoder_Lstm_one(nn.Module):
	def __init__(self):
		super(Decoder_Lstm_one, self).__init__()
		self.bais = nn.LSTM(input_size=64, hidden_size=64, num_layers=1)
		self.layer1 = LSTM_CELL(64, 128)
		self.layer2 = LSTM_CELL(128, 256)
		self.layer3 = LSTM_CELL(256, 200)
		self.layer4 = LSTM_CELL(256, 200)
		self.layer5 = LSTM_CELL(256, 200)
	
	def forward(self, x):
		y, (h, c) = self.bais(x)
		y1, c1 = self.layer1([y, c])
		y2, c2 = self.layer2([y1, c1])
		y3, c3 = self.layer3([y2, c2])
		y4, c4 = self.layer4([y3, c3])
		y5, c5 = self.layer5([y4, c4])
		return y5, c1, c2, c3, c4, c5


class Decoder_Lstm_other(nn.Module):
	def __init__(self):
		super(Decoder_Lstm_other, self).__init__()
		self.layer1 = LSTM_CELL(64, 128)
		self.layer2 = LSTM_CELL(128, 256)
		self.layer3 = LSTM_CELL(256, 200)
		self.layer4 = LSTM_CELL(256, 200)
		self.layer5 = LSTM_CELL(256, 200)
	
	def forward(self, y, c):
		y1, c1 = self.layer1([y, c[0]])
		y2, c2 = self.layer2([y1, c1])
		y3, c3 = self.layer3([y2, c2])
		y4, c4 = self.layer4([y3, c3])
		y5, c5 = self.layer5([y4, c4])
		return y5, c1, c2, c3, c4, c5


a = torch.randn(100, 16, 64)
net1 = Decoder_Lstm_one()
net2 = Decoder_Lstm_other()
net3 = Decoder_Lstm_other()
b, c1, c2, c3, c4, c5 = net1(a)