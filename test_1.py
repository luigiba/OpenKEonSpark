from Config import Config
from TransE import TransE
from TransH import TransH
from TransD import TransD
from TransR import TransR
import sys



def get_ckpt(p):
    ckpt = None
    with open(p + "checkpoint", 'r') as f:
        first_line = f.readline()
        ckpt = first_line.split(':')[1].strip().replace('"', '').split('/')
        ckpt = ckpt[len(ckpt) - 1]
    return ckpt


#/home/luigi/IdeaProjects/OpenKE_new_Spark/benchmarks/DBpedia
# dataset_path = '/content/drive/My Drive/DBpedia/1/0/'
dataset_path = '/home/luigi/files/stuff/DBpedia/5/1/'
# dataset_path = '/home/luigi/files/stuff/superuser/9/1/'
path = dataset_path + 'model/'
# path = '/home/luigi/IdeaProjects/OpenKEonSpark/res_spark/'
print(path)
ckpt = get_ckpt(path)


# con = Config(cpp_lib_path='/content/OpenKEonSpark/release/Base.so')
con = Config()
con.set_in_path(dataset_path)
# con.set_test_link_prediction(True)
# con.set_test_triple_classification(True)
con.set_dimension(int(64))
con.init()
con.set_model_and_session(TransE)
con.set_import_files(path+ckpt)
# con.set_test_log_path(path)
# con.set_n_threads_LP(1)
# con.test()
con.plot_roc(5)


