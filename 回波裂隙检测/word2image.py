# -*- coding: utf-8 -*-
# @Time    : 2023/2/5 15:42
# @Author  : HongFei Wang
import os
import re

import docx


def word2img2(word_path, result_path):
	doc = docx.Document(word_path)
	dict_rel = doc.part._rels
	for rel in dict_rel:
		rel = dict_rel[rel]
		if "image" in rel.target_ref:
			if not os.path.exists(result_path):
				os.makedirs(result_path)
			img_name = re.findall("/(.*)", rel.target_ref)[0]
			word_name = os.path.splitext(word_path)[0]
			if os.sep in word_name:
				new_name = word_name.split('\\')[-1]
			else:
				new_name = word_name.split('/')[-1]
			img_name = f'{new_name}_{img_name}'
			with open(f'{result_path}/{img_name}', "wb") as f:
				f.write(rel.target_part.blob)


for dirs in os.listdir('./data'):
	word2img2(f"./data/{dirs}", "./image/" + dirs.strip("docx"))