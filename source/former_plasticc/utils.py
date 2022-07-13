import pandas as pd
import random

def load_data(data_file, metadata_file):
    data = pd.read_csv(data_file, sep=",", names=["obj_id","mjd","passband","flux","flux_err","detected"], skiprows=1)
    ids = data['obj_id']
    ids = ids.drop_duplicates().values
    metadata = pd.read_csv(metadata_file, sep=",", names=["object_id","ra","decl","gal_l","gal_b","ddf",
    "hostgal_specz","hostgal_photoz","hostgal_photoz_err","distmod","mwebv","target"], skiprows=1)
    return [data, metadata, ids]


def get_classes(metadata):
    targets = metadata["target"]
    classes = targets.drop_duplicates().values
    return classes

def random_objs_per_class(n,classes,metadata):
    objs = []
    for cl in classes:
        objs_in_class = metadata[metadata["target"]==cl]["object_id"].values
        n_objs_in_class = len(objs_in_class)
        for i in range(n):
            rand = random.randint(0,n_objs_in_class-1)
            obj = [objs_in_class[rand],cl]
            objs.append(obj)
    return objs


