from Config import Config
from TransE import TransE
import sys
# import os


def get_ckpt(p):
    ckpt = None
    with open(p + "checkpoint", 'r') as f:
        first_line = f.readline()
        ckpt = first_line.split(':')[1].strip().replace('"', '').split('/')
        ckpt = ckpt[len(ckpt) - 1]
    return ckpt


#/home/luigi/IdeaProjects/OpenKE_new_Spark/benchmarks/DBpedia
dataset_path = '/home/luigi/files/stuff/superuser/9/1/'
path = dataset_path + 'model/'
ckpt = get_ckpt(path)


con = Config()
con.set_in_path(dataset_path)
con.set_test_link_prediction(True)
# con.set_test_triple_classification(True)
con.set_dimension(int(100))
con.init()
con.set_model_and_session(TransE)
con.set_import_files(path+ckpt)
con.set_test_log_path(path)
con.test()

