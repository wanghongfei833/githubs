# -*- coding: utf-8 -*-
# @Time    : 2022/8/18 17:48
# @Author  : HongFei Wang
import io
import json
import sys
import time

import requests

from setting import headers

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gb18030')

url = 'https://m.ctrip.com/restapi/soa2/13444/json/getCommentCollapseList?'

times = str(int(1000 * time.time()))
params = {
		# '_fxpcqlniredt': '09031030118777961081',
		# 'x-traceID': '0903103011877961081-166081' + times[:7] + '-' + times[7:],
		'arg': {'channelType': 2,
		        'collapseType': 0,
		        'commentTagId': 0,
		        'pageIndex': 3,
		        'pageSize': 10,
		        'poiId': 89906,
		        "sortType": 3,
		        "sourceType": 1,
		        "starType": 0},
		"fc": {"auth": "",
		       "cid": "09031030118777961081",
		       "ctok": "",
		       "cver": "1.0",
		       "lang": "01",
		       "sid": "8888",
		       "syscode": "09",
		       "xsid": "",
		       "extension": []}
}
url = url + '_fxpcqlniredt=09031030118777961081' + '&x-traceID0903103011877961081-166081' + times[:7] + '-' + times[7:]
# url = 'https://m.ctrip.com/restapi/soa2/13444/json/getCommentCollapseList?_fxpcqlniredt=09031030118777961081&x-traceID=09031030118777961081-1660817510738-8442903'
respose = requests.post(url, headers=headers, data=json.dumps(params)).text
htmls = json.loads(respose)
print(htmls)

a = "https://m.ctrip.com/restapi/soa2/13444/json/getCommentCollapseList?_fxpcqlniredt=09031030118777961081&x-traceID=09031030118777961081-1660819659757-7024481"
b = 'https://m.ctrip.com/restapi/soa2/13444/json/getCommentCollapseList?_fxpcqlniredt=09031030118777961081&x-traceID=09031030118777961081-1660819673930-8705238'
print(a == b)