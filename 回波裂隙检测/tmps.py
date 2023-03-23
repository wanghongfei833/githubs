# -*- coding: utf-8 -*-
# @Time    : 2023/2/17 15:54
# @Author  : HongFei Wang
import sys


def mian():
	print(f'参数1为{sys.argv[1]}')
	print(f'参数2为{sys.argv[2]}')
	print(f'参数3为{sys.argv[3]}')
	sys.stdout.flush()


if __name__ == '__main__':
	mian()