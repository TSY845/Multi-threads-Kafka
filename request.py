import requests

# url = "139.159.179.226:9092"
datas = {"reportEventId": "44f7306390944ea48953a13a6326f85a", "robotNumber": "ROB23040224", "typeName": "门窗检测", "event_pos": "2.5224,-1.5382", "event_type": "2", "disVoice": "0", "file_type": "2", "floor": "25", "map": "01", "posName": "门窗图像测试", "identify": "15", "robotId": "3157e730186d46b385b4cfb38570ffbd", "url": "https://file.test.concoai.com/image/3.jpg", "typeCode": "E0015", "disShowFlg": "0", "imgUrl": "https://file.test.concoai.com/image/3.jpg", "eventTypeId": "81b5cf4bee494187b3bd82db66638077", "disMsg": "0", "disPlay": "1", "rName": "室内1号", "tenantId": "2c0ba80be39b46a1a6368c0938deb196", "priorityCode": "LOW", "operationType": "15", "cmd": "202", "vehicleid": "ROB23040224", "time": "20210806150102", "result": "True", "points": "[]"}
r = requests.post("https://resttest.concoai.com/inDoorServer/identify/",
                  json=datas)
print(r.content)
print(r.status_code)