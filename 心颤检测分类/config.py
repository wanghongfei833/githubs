# -*- coding: utf-8 -*-
# @Time    : 2023/2/16 22:45
# @Author  : HongFei Wang
import argparse


# tf = transforms.ToTensor()
#
# a = np.random.random((125, 200, 3))
# b = torch.from_numpy(a)
# c = tf(a)
# d = cv2.resize(a,(256,256))
# print(a.shape, b.size(),c.size(),d.shape)


def config():
	parser = argparse.ArgumentParser(description="model_training")
	parser.add_argument("--data_path", default="/data/", help="数据集根目录")
	parser.add_argument("--num-classes", default=20, type=int)
	parser.add_argument("--device", default="cuda", help="training device")
	parser.add_argument("-b", "--batch-size", default=4, type=int)
	parser.add_argument("--epochs", default=30, type=int, metavar="N", help="训练次数")
	# optim
	parser.add_argument('--lr', default=0.0001, type=float, help='学习率')
	parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
	                    metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
	# train
	parser.add_argument('--resume', default='', help='resume from checkpoint')
	#
	parser.add_argument('--mean', default=[0.5, 0.5, 0.5], help='数据集均值 ')
	parser.add_argument('--std', default=[0.5, 0.5, 0.5], help='数据集方差')
	
	args = parser.parse_args()
	
	return args