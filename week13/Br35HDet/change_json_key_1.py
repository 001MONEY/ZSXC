
import json
import os



json_path = r"D:\project\step1\week13\Br35HDet\Br35HDet\annotations_all.json"
data = json.load(open(json_path,"r",encoding="utf-8"))
new_data = {}
for keys,items in data.items():
    key_ = keys.split(".jpg")[0]+".jpg"
    items_ = items
    new_data[key_] = items_

new_json_path = r"D:\project\step1\week13\Br35HDet\Br35HDet\annotations_all_new.json"

json.dump(new_data,open(new_json_path,"w",encoding="utf-8"), ensure_ascii=False)
