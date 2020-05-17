import numpy as np
import pandas as pd
import os

def extract_notes(path, naming_conv, notes, ds_type="train", label="pos"):
    compiled_path = path + ds_type + "/"
    # Check if base path exists
    if(not os.path.exists(compiled_path)):
        os.mkdir(compiled_path)
    
    # add ds_type folder
    # Check if labeled directory exists
    if(label != "None"):
        compiled_path = path + ds_type + "/" + label + "/"
        if(not os.path.exists(compiled_path)):
            os.mkdir(compiled_path)
    
    print("Extracting", len(notes), "notes to", compiled_path)
    for idx in notes.index:
        n = notes.loc[idx].TEXT
        with open("{base}{fname}_{id}.txt".format(base=compiled_path, fname=naming_conv, id=idx), "w") as f:
            f.write(n)
    
    assert len(os.listdir(compiled_path)) == len(notes), 'Not all the notes were successfully extracted.'