import h5py
import os
import json

def update_metadata(file_path,metadata):
    with h5py.File(file_path, "a") as file:
        #file.attrs.update(metadata)
        for key, value in metadata.items():
            file.attrs[key]=value

def get_metadata(filepath):
    metadata={}
    with h5py.File(filepath, "r") as file:
        metadata.update(file.attrs)
    return(metadata)

def str_to_dict(str):
    str=str.replace("false","False").replace("true","True").replace("null","None")
    dict=eval(str)
    return(dict)

def dict_to_str(dict):
    str=json.dumps(dict)
    str=str.replace("False","false").replace("True","true").replace("None","null")
    return(str)




def v2_to_v5(v2_metadata,v5_metadata):

    env_spec_new=str_to_dict(v2_metadata["env_spec"])
    v5_env_spec=str_to_dict(v5_metadata["env_spec"])

    env_spec_new.pop("autoreset")
    del env_spec_new["apply_api_compatibility"]
    env_spec_new["additional_wrappers"]=[] # no estoy seguro, ver si funciona sin...


    env_spec_new["id"]=v5_env_spec["id"]
    env_spec_new["entry_point"]=v5_env_spec["entry_point"]

    v2_metadata["env_spec"]=dict_to_str(env_spec_new)
    return(v2_metadata)



if __name__=="__main__":

    dir_path=os.path.expanduser("~")
    minari_path=os.path.join(dir_path,".minari/datasets/") #halfcheetah-expert-v0/data/main_data.hdf5"

    data_path="data/main_data.hdf5"
    # Iterate over files in the directory
    for filename in os.listdir(minari_path):
        file_path = os.path.join(minari_path, filename)
        if filename.endswith('v2'):
            v2_path=os.path.join(file_path,data_path)
            v2_metadata=get_metadata(v2_path)

            try:
                v5_filepath=file_path[:-2]+"v0"
                v5_path=os.path.join(v5_filepath,data_path)
                v5_metadata=get_metadata(v5_path)
            except FileNotFoundError:

                print("file v5 not founded")
            new_metadata=v2_to_v5(v2_metadata,v5_metadata)
            update_metadata(v2_path,new_metadata)