"""
Once learned the embedding model, this script could be used to:
    use the model learnt (predict_head_entity, predict_tail_entity, predict_relation, predict_triple)
    evaluate test triple classification task:
        - precision, recall, accuracy, f-measure
        - ROC curve

@:param
    sys.argv[1]; path where dataset is located
    sys.argv[2]; path where the model is located
    sys.argv[3]; path to OpenKEonSpark/release/Base.so
    sys.argv[4]: embedding dimensionality
    sys.argv[5]: model to use (transe/transh/transr/transd)
    sys.argv[6] (optional): target relation index (if you want to plot ROC curve)
"""

from Config import Config
from TransE import TransE
from TransH import TransH
from TransR import TransR
from TransD import TransD
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES']='-1'

for arg in sys.argv: print(arg, type(arg))
target_rel_index = None

dataset_path = sys.argv[1]
model_path = sys.argv[2]
cpp_path = sys.argv[3]
dim = sys.argv[4]
model = sys.argv[5]
if (len(sys.argv) >= 7): target_rel_index = sys.argv[6]



def get_ckpt(p):
    ckpt = None
    with open(os.path.join(p,"checkpoint"), 'r') as f:
        first_line = f.readline()
        ckpt = first_line.split(':')[1].strip().replace('"', '').split('/')
        ckpt = ckpt[len(ckpt) - 1]
    return ckpt


ckpt = get_ckpt(model_path)
con = Config(cpp_lib_path=cpp_path)
con.set_in_path(dataset_path)
con.set_test_triple_classification(True)
con.set_dimension(int(dim))
con.init()

if model.lower() == "transe": con.set_model_and_session(TransE)
elif model.lower() == "transh": con.set_model_and_session(TransH)
elif model.lower() == "transr": con.set_model_and_session(TransR)
else: con.set_model_and_session(TransD)

con.set_import_files(os.path.join(model_path, ckpt))

con.test()
if target_rel_index != None: con.plot_roc(rel_index=int(target_rel_index), fig_name='plot.png')
