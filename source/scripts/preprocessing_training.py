import pandas as pd
import numpy as np
from utils import load_data, get_classes

def retag_classes(objs, filename):
    [data, metadata,ids]=objs
    classes=get_classes(metadata)
    retagged = metadata

    for i,c in enumerate(classes):
        d = np.where(classes==c)
        retagged.loc[retagged["target"]==c,"target"] = i
        #  = metadata["target"].apply(lambda x: classes.index(x))

    retagged.to_csv(filename, sep =",")


if __name__ == "__main__":
    # data, metadata, ids= 

    retag_classes(load_data("training_set.csv", "training_set_metadata.csv"), "retagged_training_set_metadata.csv")
