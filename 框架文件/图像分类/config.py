# -*- coding: utf-8 -*-
# @Time    : 2023/2/16 22:45
# @Author  : HongFei Wang
import argparse


def config():
	parser = argparse.ArgumentParser(description="model_training")
	parser.add_argument("--data_path", default="/data/", help="VOCdevkit root")
	parser.add_argument("--num-classes", default=20, type=int)
	parser.add_argument("--device", default="cuda", help="training device")
	parser.add_argument("-b", "--batch-size", default=4, type=int)
	parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to train")
	# optim
	parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
	parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
	                    metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
	# train
	parser.add_argument('--resume', default='', help='resume from checkpoint')
	#
	parser.add_argument('--mean', default=[0.5, 0.5, 0.5], help='dataloader mean ')
	parser.add_argument('--std', default=[0.5, 0.5, 0.5], help='dataloader std')
	
	args = parser.parse_args()
	
	return args