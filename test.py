from Config import Config
from TransE import TransE
import sys

n = sys.argv[1]
max = sys.argv[2]
dim = sys.argv[3]

def get_ckpt(p):
    ckpt = None
    with open(p + "checkpoint", 'r') as f:
        first_line = f.readline()
        ckpt = first_line.split(':')[1].strip().replace('"', '').split('/')
        ckpt = ckpt[len(ckpt) - 1]
    return ckpt

dataset_path = '/content/drive/My Drive/DBpedia/{}/{}/'.format(max, n)
path = dataset_path + 'model/'
ckpt = get_ckpt(path)

con = Config(cpp_lib_path='/content/OpenKEonSpark/release/Base.so')
con.set_in_path(dataset_path)
con.set_test_link_prediction(True)
con.set_test_triple_classification(True)
con.set_dimension(int(dim))
con.init()
con.set_model_and_session(TransE)
con.set_import_files(path+ckpt)
con.set_test_log_path(path)
con.test()
print(con.acc)
