# -*- coding: utf-8 -*-
# @Time    : 2022/12/28 16:10
# @Author  : HongFei Wang
import sys

import pygame

pygame.init()

screen = pygame.display.set_mode((300, 300))
while True:
	event = pygame.event.poll()
	if event.type == pygame.QUIT:
		sys.exit()
	if event.type == pygame.MOUSEBUTTONDOWN:
		print(pygame.mouse.get_pos())