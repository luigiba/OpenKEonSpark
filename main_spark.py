import argparse
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from tensorflowonspark import TFCluster
import distribute_training
import tensorflow as tf
import sys
import os
import traceback
import time
import numpy as np
from os import path

NEW_BATCH_TRIPLES_FILE_NAME = 'batch2id.txt'
NEW_BATCH_ENTITIES_FILE_NAME = 'batchEntity2id.txt'
NEW_BATCH_TEST_FILE_NAME = 'batchTest2id.txt'
NEW_BATCH_VALID_FILE_NAME = 'batchValid2id.txt'

ENTITIES_FILE_NAME = 'entity2id.txt'
TRIPLES_FILE_NAME = 'train2id.txt'
TEST_FILE_NAME = 'test2id.txt'
VALID_FILE_NAME = 'valid2id.txt'

#tensors which depends from entities dimension
ENTITY_EMBEDDING_TENSOR_NAME = 'ent_embeddings'     #from TransE, TransH, TransR, TransD
ENTITY_TRANSFER_TENSOR_NAME = 'ent_transfer'        #from TransD


def update_entities_and_model():
    '''
    Update the tensor variables if new entites are introduced in the new batch
    '''
    n_entities = 0
    n_new_entities = 0
    final_entity_size = 0
    batch_entities = []
    entity_lines = []

    ######### READ NEW ENTITIES #########
    with open(os.path.join(sys.argv.input_path, NEW_BATCH_ENTITIES_FILE_NAME), 'r') as f:
        n_new_entities = int(f.readline().strip())
        if n_new_entities > 0:
            for _ in range(n_new_entities):
                batch_entities.append(f.readline())


    ######### READ OLD ENTITIES #########
    with open(os.path.join(sys.argv.input_path, ENTITIES_FILE_NAME)) as f:
        entity_lines = f.readlines()
    n_entities = int(entity_lines[0])
    final_entity_size = n_entities + n_new_entities
    if sys.argv.debug: print("Number of new entities in batch: " + str(n_new_entities))


    ######### UPDATE THE MODEL #########
    if n_new_entities > 0:
        con, ckpt = distribute_training.get_conf_to_update_model(sys.argv.output_path)
        vars = []

        if sys.argv.debug: print("\nGLOBAL VARS FOUNDED IN CHECKPOINT:\n")
        with con.graph.as_default():
            with con.sess.as_default():
                for v in tf.global_variables():
                    if sys.argv.debug: print(str(v.name) + " " + str(v.shape))
                    vars.append(v)
        if sys.argv.debug: print('\n')


        if sys.argv.debug: print("NEW GLOBAL VARIABLES")
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            with sess.as_default():
                for v in vars:
                    current_name = v.name.split(':')[0]

                    if current_name == ENTITY_EMBEDDING_TENSOR_NAME or current_name == ENTITY_TRANSFER_TENSOR_NAME:
                        tmp = tf.get_variable(name=current_name, shape=[final_entity_size, v.shape[1]], initializer=tf.contrib.layers.xavier_initializer(uniform = False), dtype=v.dtype)
                        sess.run(tf.initialize_variables([tmp]))
                        tmp_value = con.sess.run(v)
                        sess.run(tf.scatter_update(tmp, [i for i in range(0, n_entities)], tmp_value))

                    elif current_name in [ENTITY_EMBEDDING_TENSOR_NAME+'/Adam', ENTITY_EMBEDDING_TENSOR_NAME+'/Adam_1', ENTITY_TRANSFER_TENSOR_NAME+'/Adam', ENTITY_TRANSFER_TENSOR_NAME+'/Adam_1']:
                        tmp = tf.get_variable(name=current_name, shape=[final_entity_size, v.shape[1]], initializer=tf.zeros_initializer(), dtype=v.dtype)
                        sess.run(tf.initialize_variables([tmp]))
                        tmp_value = con.sess.run(v)
                        sess.run(tf.scatter_update(tmp, [i for i in range(0, n_entities)], tmp_value))

                    else:
                        tmp = tf.get_variable(name=current_name, shape=v.shape, dtype=v.dtype)
                        tmp_value = con.sess.run(v)
                        sess.run(tf.assign(tmp, tmp_value))

                for v in tf.global_variables():
                    print(str(v.name) + " " + str(v.shape))

                saver = tf.train.Saver()
                saver.save(sess, ckpt, write_state=False)


    ######### UPDATE ENTITY FILE #########
    if n_new_entities > 0:
        #update number of entities
        entity_lines[0] = str(final_entity_size) + "\n"

        #update entities: append the new entities at the end of the file
        entity_lines = entity_lines + batch_entities

        #update entity2id.txt
        with open(os.path.join(sys.argv.input_path, ENTITIES_FILE_NAME), "w") as f:
            f.writelines(entity_lines)
    if sys.argv.debug: print("Entity file updated")

    return n_new_entities, final_entity_size


def update_triples(file_name_to_update, file_name_batch):
    '''
    Update file_name_to_update by appending the triples contained in file_name_batch
    The first line of file_name_to_update (i.e. the number of triples) is updated accordingly
    :param file_name_to_update:
    :param file_name_batch:
    '''
    batch_triples_size = 0
    batch_triples = []

    #open batch file
    with open(os.path.join(sys.argv.input_path, file_name_batch)) as f:
        batch_triples_size = int(f.readline().strip())
        if batch_triples_size > 0:
            for _ in range(batch_triples_size):
                batch_triples.append(f.readline())
    print("Number of new lines: " + str(batch_triples_size))

    if batch_triples_size > 0:
        #open file
        with open(os.path.join(sys.argv.input_path, file_name_to_update)) as f:
            lines = f.readlines()

        #update number of triples
        lines[0] = str(int(lines[0]) + batch_triples_size) + "\n"

        #update triples
        lines = lines + batch_triples       #append the new triples at the end of the file

        #update file
        with open(os.path.join(sys.argv.input_path, file_name_to_update), "w") as f:
            f.writelines(lines)
    if sys.argv.debug: print("File updated")


def feed_batch():
    '''
    Update files containing entities / relations / triples with data contained in new batch
    '''
    if sys.argv.debug: print("New batch file founded")
    try:
        if sys.argv.debug: print("Updating "+ENTITIES_FILE_NAME+" and model tensors...")
        update_entities_and_model()

        if sys.argv.debug: print("Updating "+TRIPLES_FILE_NAME+"...")
        update_triples(TRIPLES_FILE_NAME, NEW_BATCH_TRIPLES_FILE_NAME)

        if sys.argv.debug: print("Updating "+TEST_FILE_NAME+"...")
        update_triples(TEST_FILE_NAME, NEW_BATCH_TEST_FILE_NAME)

        if sys.argv.debug: print("Updating "+VALID_FILE_NAME+"...")
        update_triples(VALID_FILE_NAME, NEW_BATCH_VALID_FILE_NAME)


    except Exception as e:
        print("Error occured while feeding new batch:")
        traceback.print_exc()
        print(e)


def is_new_batch():
    '''
    Return True if there is a new batch to train
    '''
    return os.path.isfile(os.path.join(sys.argv.input_path,NEW_BATCH_TRIPLES_FILE_NAME)) and \
           os.path.isfile(os.path.join(sys.argv.input_path,NEW_BATCH_ENTITIES_FILE_NAME)) and \
            os.path.isfile(os.path.join(sys.argv.input_path,NEW_BATCH_TEST_FILE_NAME)) and \
           os.path.isfile(os.path.join(sys.argv.input_path,NEW_BATCH_VALID_FILE_NAME))


def remove_batch_files():
    '''
    Remove 4 batch files
    '''
    os.remove(os.path.join(sys.argv.input_path,NEW_BATCH_TRIPLES_FILE_NAME))
    os.remove(os.path.join(sys.argv.input_path,NEW_BATCH_ENTITIES_FILE_NAME))
    os.remove(os.path.join(sys.argv.input_path,NEW_BATCH_TEST_FILE_NAME))
    os.remove(os.path.join(sys.argv.input_path,NEW_BATCH_VALID_FILE_NAME))


def get_test_total():
    try:
        first_line = ''
        with open(os.path.join(sys.argv.input_path,TEST_FILE_NAME), 'r') as f:
            first_line = f.readline()
        return int(first_line.strip())

    except Exception as e:
        traceback.print_exc()
        return 0


def n_n():
    '''
    Generates type_constrain.txt file
    '''
    if sys.argv.debug: print("Generating constraints file...")

    lef = {}
    rig = {}
    rellef = {}
    relrig = {}

    triple = open(os.path.join(sys.argv.input_path,"train2id.txt"), "r")
    valid = open(os.path.join(sys.argv.input_path,"valid2id.txt"), "r")
    test = open(os.path.join(sys.argv.input_path,"test2id.txt"), "r")

    tot = (int)(triple.readline())
    for i in range(tot):
        content = triple.readline()
        h,t,r = content.strip().split()
        if not (h,r) in lef:
            lef[(h,r)] = []
        if not (r,t) in rig:
            rig[(r,t)] = []
        lef[(h,r)].append(t)
        rig[(r,t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

    tot = (int)(valid.readline())
    for i in range(tot):
        content = valid.readline()
        h,t,r = content.strip().split()
        if not (h,r) in lef:
            lef[(h,r)] = []
        if not (r,t) in rig:
            rig[(r,t)] = []
        lef[(h,r)].append(t)
        rig[(r,t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

    tot = (int)(test.readline())
    for i in range(tot):
        content = test.readline()
        h,t,r = content.strip().split()
        if not (h,r) in lef:
            lef[(h,r)] = []
        if not (r,t) in rig:
            rig[(r,t)] = []
        lef[(h,r)].append(t)
        rig[(r,t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

    test.close()
    valid.close()
    triple.close()

    f = open(os.path.join(sys.argv.input_path,"type_constrain.txt"), "w")
    f.write("%d\n"%(len(rellef)))
    for i in rellef:
        f.write("%s\t%d"%(i,len(rellef[i])))
        for j in rellef[i]:
            f.write("\t%s"%(j))
        f.write("\n")
        f.write("%s\t%d"%(i,len(relrig[i])))
        for j in relrig[i]:
            f.write("\t%s"%(j))
        f.write("\n")
    f.close()



if __name__ == '__main__':
    print("Creating Spark Context...")
    sc = SparkContext(conf=SparkConf().setAppName('OpenKEonSpark'))

    print("Parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=int(sc._conf.get("spark.executor.instances")))
    parser.add_argument("--num_ps", help="number of ps nodes", type=int, default=1)
    parser.add_argument("--num_gpus", help="number of gpus to use", type=int, default=0)
    parser.add_argument("--cpp_lib_path", help="cpp lib.so absolute path contained in ./release/Base.so", type=str, default=None)
    parser.add_argument("--input_path", help="dataset absolute path", type=str, default=None)
    parser.add_argument("--output_path", help="model output absolute path", type=str, default=None)
    parser.add_argument("--train_times", help="no. epochs", type=int, default=100)
    parser.add_argument("--n_mini_batches", help="# mini batches; if set to zero it will be automatically computed", type=int, default=0)
    parser.add_argument("--alpha", help="learning rate", type=float, default=0.00001)
    parser.add_argument("--margin", help="margin hyperparameter used during training", type=float, default=1.0)
    parser.add_argument("--bern_flag", help="whether to use or not bern method for sampling; ; 0=False, n=True", type=int, default=0)
    parser.add_argument("--embedding_dimension", help="embedding dimension (both entities and rel)", type=int, default=64)
    parser.add_argument("--ent_dimension", help="entities embedding dimension", type=int, default=0)
    parser.add_argument("--rel_dimension", help="relations embedding dimension", type=int, default=0)
    parser.add_argument("--ent_neg_rate", help="number of negative triples generated by corrupting the entity", type=int, default=1)
    parser.add_argument("--rel_neg_rate", help="number of negative triples generated by corrupting the relation", type=int, default=0)
    parser.add_argument("--optimizer", help="Optimization algorithm (SGD/Adam)", type=str, default="SGD")
    parser.add_argument("--early_stop_patience", help="no. epochs to wait for accuracy/loss improvement before early stop", type=int, default=5)
    parser.add_argument("--early_stop_stopping_step", help="perfrom early stop each stopping step", type=int, default=1)
    parser.add_argument("--early_stop_start_step", help="perfrom early stop from start step", type=int, default=1)
    parser.add_argument("--model", help="model to be used (TransE/TransH/TransR/TransD)", type=str, default="TransE")
    parser.add_argument("--debug", help="if Ture prints additional debug information", type=bool, default=True)
    parser.add_argument("--mode", help="whether to perform train or evaluation mode", type=str, default="train")
    parser.add_argument("--test_head", help="perform link prediction evaluation on missing head, too (only if mode != 'train'); 0=False, n=True", type=int, default=0)
    (args, remainder) = parser.parse_known_args()
    num_workers = args.cluster_size - args.num_ps
    print("===== num_executors={}, num_workers={}, num_ps={}".format(args.cluster_size, num_workers, args.num_ps))
    sys.argv = args

    #generate type_constrain.txt file
    n_n()
    if args.mode == 'train':
        if is_new_batch(): feed_batch()
        try: os.remove(os.path.join(args.output_path,"stop.txt"))
        except: pass



    if args.debug: print("Launching jobs...")
    elapsed_time = time.time()
    cluster = TFCluster.run(sc, distribute_training.main_fun, args, args.cluster_size, args.num_ps, True, TFCluster.InputMode.TENSORFLOW)
    cluster.shutdown(timeout=-1)
    elapsed_time = time.time() - elapsed_time
    with open(os.path.join(args.output_path,'time.txt'), 'w') as f:
        f.write("Elapsed time: " + str(elapsed_time) + "\n")


    if args.mode == 'train':
        if is_new_batch(): remove_batch_files()
        if args.debug: print("Restoring the best model founded during training...")
        if path.exists(os.path.join(args.output_path,"stop.txt")):
            step = None
            with open(os.path.join(args.output_path,"stop.txt"), "r") as f:
                step = int(f.readline().strip())

            if step != None:
                min_diff = sys.maxsize
                nearest = None
                with open(os.path.join(args.output_path,"checkpoint"), "r") as ckpt_file:
                    lines = ckpt_file.readlines()

                for line in lines:
                    l = line.replace('"','').split("/")
                    n = int(l[len(l) - 1].split("-")[1])
                    diff = abs(step - n)
                    if diff < min_diff:
                        nearest = n
                        min_diff = diff

                if nearest != None:
                    with open(os.path.join(args.output_path,"checkpoint"), "w") as ckpt_file:
                        ckpt_file.write('model_checkpoint_path: "'+os.path.join(args.output_path, 'model.ckpt-'+str(nearest))+'"\n')
                        ckpt_file.write('all_model_checkpoint_paths: "'+os.path.join(args.output_path, 'model.ckpt-'+str(nearest))+'"\n')

                for f in os.listdir(args.output_path):
                    if f.startswith("model") and len(f.split("model.ckpt-"+str(nearest))) == 1:
                        os.remove(os.path.join(args.output_path,f))
    else:
        #set link prediction on tail variables
        d = {
            'r_tot' : 0.0, 'r_filter_tot' : 0.0, 'r_tot_constrain' : 0.0, 'r_filter_tot_constrain' : 0.0,
            'r1_tot' : 0.0, 'r1_filter_tot' : 0.0, 'r1_tot_constrain' : 0.0, 'r1_filter_tot_constrain' : 0.0,
            'r3_tot' : 0.0, 'r3_filter_tot' : 0.0,  'r3_tot_constrain' : 0.0, 'r3_filter_tot_constrain' : 0.0,
            'r_rank' : 0.0, 'r_filter_rank' : 0.0, 'r_rank_constrain' : 0.0, 'r_filter_rank_constrain' : 0.0,
            'r_reci_rank' : 0.0,'r_filter_reci_rank' : 0.0, 'r_reci_rank_constrain' : 0.0, 'r_filter_reci_rank_constrain' : 0.0,
            'r_mis_err' : 0.0, 'r_spec_err' : 0.0, 'r_gen_err' : 0.0,
            'r_filter_mis_err' : 0.0, 'r_filter_spec_err' : 0.0, 'r_filter_gen_err' : 0.0,
            'r_mis_err_constrain' : 0.0, 'r_spec_err_constrain' : 0.0, 'r_gen_err_constrain' : 0.0,
            'r_filter_mis_err_constrain' : 0.0, 'r_filter_spec_err_constrain' : 0.0, 'r_filter_gen_err_constrain' : 0.0
        }

        if args.test_head != 0:
            #set link prediction on head variables
            d['l_tot'] = 0.0
            d['l_filter_tot'] = 0.0
            d['l_tot_constrain'] = 0.0
            d['l_filter_tot_constrain'] = 0.0
            d['l1_tot'] = 0.0
            d['l1_filter_tot'] = 0.0
            d['l1_tot_constrain'] = 0.0
            d['l1_filter_tot_constrain'] = 0.0
            d['l3_tot'] = 0.0
            d['l3_filter_tot'] = 0.0
            d['l3_tot_constrain'] = 0.0
            d['l3_filter_tot_constrain'] = 0.0
            d['l_rank'] = 0.0
            d['l_filter_rank'] = 0.0
            d['l_rank_constrain'] = 0.0
            d['l_filter_rank_constrain'] = 0.0
            d['l_reci_rank'] = 0.0
            d['l_filter_reci_rank'] = 0.0
            d['l_reci_rank_constrain'] = 0.0
            d['l_filter_reci_rank_constrain'] = 0.0
            d['l_mis_err'] = 0.0
            d['l_spec_err'] = 0.0
            d['l_gen_err'] = 0.0
            d['l_filter_mis_err'] = 0.0
            d['l_filter_spec_err'] = 0.0
            d['l_filter_gen_err'] = 0.0
            d['l_mis_err_constrain'] = 0.0
            d['l_spec_err_constrain'] = 0.0
            d['l_gen_err_constrain'] = 0.0
            d['l_filter_mis_err_constrain'] = 0.0
            d['l_filter_spec_err_constrain'] = 0.0
            d['l_filter_gen_err_constrain'] = 0.0

        testTotal = get_test_total()
        stop_evaluation = False

        #get results from workers
        for j in range(0, num_workers):
            with open(os.path.join(args.output_path,"lp_worker_"+str(j)), "r") as f:
                lines_w = f.readlines()

                if lines_w[0].strip() != 'done':
                    stop_evaluation = True
                    break

                for i_w in range(1, len(lines_w)):
                    key = lines_w[i_w].split(":")[0].strip()
                    value = float(lines_w[i_w].split(":")[1].strip())
                    d[key] += value

        if stop_evaluation:
            print("Workers didnt finish link prediction evaluation; Restart the job to finish it")
            sys.exit(0)

        for key in d.keys():
            d[key] = np.divide(d[key], testTotal)

        #print link prediction evaluation results
        print("\n ========== LINK PREDICTION RESULTS ==========\nNo type constraint results:")
        print("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format("metric", "MRR", "MR", "hit@10", "hit@3", "hit@1", "hit@1GenError", "hit@1SpecError", "hit@1MisError"))
        if args.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format("l(raw):",d['l_reci_rank'], d['l_rank'], d['l_tot'], d['l3_tot'], d['l1_tot'], d['l_gen_err'], d['l_spec_err'], d['l_mis_err']))
        print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format("r(raw):", d['r_reci_rank'], d['r_rank'], d['r_tot'], d['r3_tot'], d['r1_tot'], d['r_gen_err'], d['r_spec_err'], d['r_mis_err']))
        if args.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}\n".format("mean(raw):", np.divide((d['l_reci_rank']+d['r_reci_rank']),2), np.divide((d['l_rank']+d['r_rank']),2), np.divide((d['l_tot']+d['r_tot']),2), np.divide((d['l3_tot']+d['r3_tot']),2), np.divide((d['l1_tot']+d['r1_tot']),2), np.divide((d['l_gen_err']+d['r_gen_err']),2), np.divide((d['l_spec_err']+d['r_spec_err']),2), np.divide((d['l_mis_err']+d['r_mis_err']),2)))

        if args.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format("l(filter):", d['l_filter_reci_rank'], d['l_filter_rank'], d['l_filter_tot'], d['l3_filter_tot'], d['l1_filter_tot'], d['l_filter_gen_err'], d['l_filter_spec_err'], d['l_filter_mis_err']))
        print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format('r(filter):', d['r_filter_reci_rank'], d['r_filter_rank'], d['r_filter_tot'], d['r3_filter_tot'], d['r1_filter_tot'], d['r_filter_gen_err'], d['r_filter_spec_err'], d['r_filter_mis_err']))
        if args.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}\n".format('mean(filter):', np.divide((d['l_filter_reci_rank']+d['r_filter_reci_rank']),2), np.divide((d['l_filter_rank']+d['r_filter_rank']),2), np.divide((d['l_filter_tot']+d['r_filter_tot']),2), np.divide((d['l3_filter_tot']+d['r3_filter_tot']),2), np.divide((d['l1_filter_tot']+d['r1_filter_tot']),2), np.divide((d['l_filter_gen_err']+d['r_filter_gen_err']),2), np.divide((d['l_filter_spec_err']+d['r_filter_spec_err']),2), np.divide((d['l_filter_mis_err']+d['r_filter_mis_err']),2)))

        print("Type constraint results:")
        print("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format("metric", "MRR", "MR", "hit@10", "hit@3", "hit@1", "hit@1GenError", "hit@1SpecError", "hit@1MisError"))
        if args.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format('l(raw):', d['l_reci_rank_constrain'], d['l_rank_constrain'], d['l_tot_constrain'], d['l3_tot_constrain'], d['l1_tot_constrain'], d['l_gen_err_constrain'], d['l_spec_err_constrain'], d['l_mis_err_constrain']))
        print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format('r(raw):', d['r_reci_rank_constrain'], d['r_rank_constrain'], d['r_tot_constrain'], d['r3_tot_constrain'], d['r1_tot_constrain'], d['r_gen_err_constrain'], d['r_spec_err_constrain'], d['r_mis_err_constrain']))
        if args.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}\n".format('mean(raw):', np.divide((d['l_reci_rank_constrain']+d['r_reci_rank_constrain']),2), np.divide((d['l_rank_constrain']+d['r_rank_constrain']),2), np.divide((d['l_tot_constrain']+d['r_tot_constrain']),2), np.divide((d['l3_tot_constrain']+d['r3_tot_constrain']),2), np.divide((d['l1_tot_constrain']+d['r1_tot_constrain']),2), np.divide((d['l_gen_err_constrain']+d['r_gen_err_constrain']),2), np.divide((d['l_spec_err_constrain']+d['r_spec_err_constrain']),2), np.divide((d['l_mis_err_constrain']+d['r_mis_err_constrain']),2)))

        if args.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format('l(filter):', d['l_filter_reci_rank_constrain'], d['l_filter_rank_constrain'], d['l_filter_tot_constrain'], d['l3_filter_tot_constrain'], d['l1_filter_tot_constrain'], d['l_filter_gen_err_constrain'], d['l_filter_spec_err_constrain'], d['l_filter_mis_err_constrain']))
        print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format('r(filter):', d['r_filter_reci_rank_constrain'], d['r_filter_rank_constrain'], d['r_filter_tot_constrain'], d['r3_filter_tot_constrain'], d['r1_filter_tot_constrain'], d['r_filter_gen_err_constrain'], d['r_filter_spec_err_constrain'], d['r_filter_mis_err_constrain']))
        if args.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}\n".format('mean(filter):', np.divide((d['l_filter_reci_rank_constrain']+d['r_filter_reci_rank_constrain']),2), np.divide((d['l_filter_rank_constrain']+d['r_filter_rank_constrain']),2), np.divide((d['l_filter_tot_constrain']+d['r_filter_tot_constrain']),2), np.divide((d['l3_filter_tot_constrain']+d['r3_filter_tot_constrain']),2), np.divide((d['l1_filter_tot_constrain']+d['r1_filter_tot_constrain']),2), np.divide((d['l_filter_gen_err_constrain']+d['r_filter_gen_err_constrain']),2), np.divide((d['l_filter_spec_err_constrain']+d['r_filter_spec_err_constrain']),2), np.divide((d['l_filter_mis_err_constrain']+d['r_filter_mis_err_constrain']),2)))

        #remove checkpoint generated from threads
        print()
        for index in range(0, num_workers):
            try: os.remove(os.path.join(args.output_path,"lp_worker_"+str(index)))
            except: print(" LOG:\tFile " + os.path.join(args.output_path,"lp_worker_"+str(index)) + " not founded")
        print()


    if args.debug: print("OpenKEonSpark jobs finished")

