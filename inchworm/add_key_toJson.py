import os
import json

def func_1():
    for dir_name in os.listdir("collection"):
        if os.path.exists("collection/"+dir_name+"/"+"agent_data/0.json"):
            with open("collection/"+dir_name+"/"+"agent_data/0.json", "r") as f:
                dict = json.load(f)
            if "firstStep" not in dict:
                with open("collection/"+dir_name+"/"+"agent_data/0.json", "w") as f:
                    dict.update({"firstStep": False})
                    json.dump(dict, f)
        else:
            print(dir_name)

def func_2():
    for name in os.listdir("elite"):
        with open("elite/"+name, "r") as f:
            dict = json.load(f)
        if "head_friction" in dict:
            dict.pop("head_friction")
            dict["friction_list"] = [0.3 for _ in range(dict["parts_num"])]
            dict["friction_list"][-1] = 1.5
            with open("elite/"+name, "w") as f:
                # dict["parts_size"] = (6/dict["parts_num"], 0.25)
                json.dump(dict, f)
        else:
            pass

func_2()
