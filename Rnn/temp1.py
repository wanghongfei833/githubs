# -*- coding: utf-8 -*-
# @Time    : 2022/8/17 23:28
# @Author  : HongFei Wang

import re
import time

import requests

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf8")  ## encoding 的参数写成网站的编码格式即可
s = requests.session()


class TaobaoLogin:
	
	def __init__(self, ua, account, TPL_password2):
		self.account = account  # 淘宝用户名
		self.ua = ua  # 淘宝关键参数，包含用户浏览器等一些信息，很多地方会使用，从浏览器或抓包工具中复制，可重复使用
		self.TPL_password2 = TPL_password2  # 加密后的密码，从浏览器或抓包工具中复制，可重复使用
	
	def user_check(self):
		print("1.调用账户是否需要滑动验证码接口(True:需要/False:不需要)")
		user_check_url = "https://login.taobao.com/member/request_nick_check.do?_input_charset=utf-8"
		data = {
				"username": self.account,
				"ua": self.ua
		}
		try:
			user_check_response = s.post(user_check_url, data=data)
		except Exception as e:
			print("用户验证接口请求失败，msg:")
			raise e
		user_check_result = user_check_response.json()["needcode"]
		print("2.返回结果为:%s" % user_check_result)
		if not user_check_result:
			pass
		else:
			print("3.需要滑动验证，搞不定！休息10秒再来一次试试")
			time.sleep(10)
			self.user_check()
		
		return user_check_result
	
	def get_token(self):
		if not self.user_check():
			print("3.调用验证密码获取token的接口")
			verify_password_url = "https://login.taobao.com/member/login.jhtml"
			verify_password_headers = {
					"cookie": "__wpkreporterwid_=5260f939-d553-4d4a-0578-e48989700390; mt=ci=-1_0; lego2_cna=5X48M5KUUW5HW5UH5YYCCPC4; t=d8d16c182d88933007f0bccbcfe3b577; xlly_s=1; thw=cn; ctoken=9QoUdTXF5NRbnj27uEu23caK; _tb_token_=37da96eeeb338; _m_h5_tk=cee880ca29d8fbe48ba3d4b3c9b9dc0f_1660806619487; _m_h5_tk_enc=3f1d7260613f05ad14c9dc0d10d44cca; cookie2=1cff084d77436bcad05ead780559793f; _samesite_flag_=true; mt=ci=0_0; cna=LNX6Gsb+uFICAbfcSQv7uTJw; l=eBaTzIRrL3y0dEmBBO5BFurza77TaIRb8rVzaNbMiInca6Ch9FN-xNCHRDlkWdtjgt5UaexPUcfRbdFvSJ4U-x1Hrt7APlUOrFJ68e1..; isg=BEFBvZ9hx22cGCs8DqSolJG9UI1bbrVgNF8q_aOW5sinimBc6b36MdbAbP7MhE2Y; tfstk=ctlCBNNGVMjI5r9PT2TwU5eoScF5ZO-_SpZKRhmGf5M2s7uCi914chVZEGVzyP1..; sgcookie=E1008xcza8vO+JkTtAsfiuikHpEO0LdugDM8m5oblCYXaxXtTh3eyFa5gsdDGcFHuHbpy63VFr7zKbY5k0xoUVHsHUraZ+DdYkETPN82dr0rhjdrFUQWx3rOitCZe657ItUe; unb=1582950425; uc1=cookie14=UoeyDtfAipcc4g==&pas=0&cookie15=UIHiLt3xD8xYTw==&existShop=false&cookie16=V32FPkk/xXMk5UvIbNtImtMfJQ==&cookie21=WqG3DMC9Fb5mPLIQo9kR; uc3=vt3=F8dCv4CJu/ni20sMhhM=&id2=UoTUOLr0OKomJg==&nk2=GgoQFIZYEz3C87FTeA==&lg2=U+GCWk/75gdr5Q==; csg=c797aabb; lgc=yao1192113894; cancelledSubSites=empty; cookie17=UoTUOLr0OKomJg==; dnk=yao1192113894; skt=c68baa13ace68c67; existShop=MTY2MDgwMTkzMA==; uc4=nk4=0@GIavOvVtBOb77ynaTG58+5Cgwp1PYjVB&id4=0@UOx+xVbOy2+zu78MKax/Jga66XdI; tracknick=yao1192113894; _cc_=VT5L2FSpdA==; _l_g_=Ug==; sg=45d; _nk_=yao1192113894; cookie1=UNRmOonbqzUHyuN+//NjYILoj3wrEGOJec49QzLjtAs=",
					"Connection": "keep-alive",
					"cache-control": "no-cache",
					"content-type": "application/x-www-form-urlencoded",
					"origin": "https://login.taobao.com",
					"referer": "https://login.taobao.com/member/login.jhtml?redirectURL=https%3A%2F%2Fwww.taobao.com%2F",
					"upgrade-insecure-requests": "1",
					"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36"
			}
			verify_password_data = {
					"TPL_username": self.account,
					"TPL_password": "",
					"TPL_password_2": self.TPL_password2,
					"ncoSig": "",
					"ncoSessionid": "",
					"ncoToken": "6617f9d84e25f8f774ad1b5d6a7fa5336c5fe3b2",
					"slideCodeShow": "false",
					"useMobile": "false",
					"lang": "zh_CN",
					"loginsite": "0",
					"newlogin": "0",
					"TPL_redirect_url": "https://www.taobao.com/",
					"from": "tb",
					"fc": "default",
					"style": "default",
					"css_style": "",
					"keyLogin": "false",
					"qrLogin": "true",
					"newMini": "false",
					"newMini2": "false",
					"tid": "",
					"loginType": "3",
					"minititle": "",
					"minipara": "",
					"pstrong": "",
					"sign": "",
					"need_sign": "",
					"isIgnore": "",
					"full_redirect": "",
					"sub_jump": "",
					"popid": "",
					"callback": "",
					"guf": "",
					"not_duplite_str": "",
					"need_user_id": "",
					"poy": "",
					"gvfdcname": "10",
					"gvfdcre": "8747470733A2F2F6C6F67696E2E74616F62616F2E636F6D2F6D656D6265722F6C6F676F75742E6A68746D6C3F73706D3D613231626F2E323031372E3735343839343433372E372E356166393131643959427031513326663D746F70266F75743D7472756526726564697265637455524C3D68747470732533412532462532467777772E74616F62616F2E636F6D253246",
					"from_encoding": "",
					"sub": "",
					"loginASR": "1",
					"loginASRSuc": "1",
					"allp": "",
					"oslanguage": "zh-CN",
					"sr": " 1920*1080",
					"osVer": "",
					"naviVer": "chrome|74.03729131",
					"osACN": "Mozilla",
					"osAV": "5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36",
					"osPF": "Win32",
					"miserHardInfo": "",
					"appkey": "00000000",
					"nickLoginLink": "",
					"mobileLoginLink": "https://login.taobao.com/member/login.jhtml?redirectURL=https://www.taobao.com/&useMobile=true",
					"showAssistantLink": "",
					"um_token": " TF4565FB51ACF66D3CFA13717D3CAC2903A6C266FC2AF9490AEC8FDD715"
			}
			
			verify_password_res = s.post(verify_password_url, headers=verify_password_headers, data=verify_password_data)
			# print(res.text)
			st_token_url = re.search(r'<script src="(.*?)"></script>', verify_password_res.text).group(1)
			print("4.获取到st_token_url:" + st_token_url)
			print("5.get请求st_token_url，以获取token")
			st_response = s.get('http:' + st_token_url)
			st_token = re.search(r'data":{"st":"(.*?)"}}', st_response.text).group(1)
			print("6.st_token:" + st_token)
			return st_token
	
	def get_nick_name(self):
		my_taobao_url = 'https://login.taobao.com/member/vst.htm?st=' + self.get_token()
		my_taobao_headers = {
				'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
		}
		taobao_response = s.get(my_taobao_url, headers=my_taobao_headers)
		# print(taobao_response.text)
		my_taobao_match = re.search(r'top.location.href = "(.*?)";', taobao_response.text).group(1)
		# print("taobao_location:"+str(my_taobao_match))
		print("7.跳转到我的淘宝链接:" + my_taobao_match)
		
		res = s.get(my_taobao_match, headers=my_taobao_headers)
		# print(res.text)
		nick_name_match = re.search(r'<input id="mtb-nickname" type="hidden" value="(.*?)"/>', res.text).group(1)
		print("8.我的淘宝昵称是:" + nick_name_match)


if __name__ == '__main__':
	import setting
	
	ua = setting.ua
	TPL_password2 = setting.passward2
	account = setting.cell
	login = TaobaoLogin(ua=ua, account=account, TPL_password2=TPL_password2)
	login.get_nick_name()