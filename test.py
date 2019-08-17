from Config import Config
from TransE import TransE
from TransH import TransH
from TransR import TransR
from TransD import TransD
import sys

for arg in sys.argv:
    print(type(arg), arg)
    print("\n")

n = sys.argv[1]
max = sys.argv[2]
dim = sys.argv[3]
model = sys.argv[4]
lp = sys.argv[5]



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
con.set_test_link_prediction(bool(int(lp)))
con.set_test_triple_classification(True)
con.set_dimension(int(dim))
con.init()

if model.lower() == "transe":
    con.set_model_and_session(TransE)
    if (int(dim) <= 32): con.set_n_threads_LP(10)
    if (int(dim) > 32 and int(dim) <= 64): con.set_n_threads_LP(10)
    if (int(dim) > 64 and int(dim) <= 128): con.set_n_threads_LP(5)
    if (int(dim) > 128 and int(dim) <= 256): con.set_n_threads_LP(5)
elif model.lower() == "transh":
    con.set_model_and_session(TransH)
    if (int(dim) <= 32): con.set_n_threads_LP(10)
    if (int(dim) > 32 and int(dim) <= 64): con.set_n_threads_LP(7)
    if (int(dim) > 64 and int(dim) <= 128): con.set_n_threads_LP(5)
    if (int(dim) > 128 and int(dim) <= 256): con.set_n_threads_LP(5)
elif model.lower() == "transr":
    con.set_model_and_session(TransR)
    if (int(dim) <= 32): con.set_n_threads_LP(7)
    if (int(dim) > 32 and int(dim) <= 64): con.set_n_threads_LP(5)
    if (int(dim) > 64 and int(dim) <= 128): con.set_n_threads_LP(4)
    if (int(dim) > 128 and int(dim) <= 256): con.set_n_threads_LP(3)
else:
    con.set_model_and_session(TransD)
    if (int(dim) <= 32): con.set_n_threads_LP(7)
    if (int(dim) > 32 and int(dim) <= 64): con.set_n_threads_LP(5)
    if (int(dim) > 64 and int(dim) <= 128): con.set_n_threads_LP(4)
    if (int(dim) > 128 and int(dim) <= 256): con.set_n_threads_LP(3)

con.set_import_files(path+ckpt)
con.set_test_log_path(path)
con.test()
print(con.acc)
