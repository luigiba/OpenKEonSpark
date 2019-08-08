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
from os import path


DEBUG = distribute_training.DEBUG

NEW_BATCH_TRIPLES_FILE_NAME = 'batch2id.txt'
NEW_BATCH_ENTITIES_FILE_NAME = 'batchEntity2id.txt'
NEW_BATCH_TEST_FILE_NAME = 'batchTest2id.txt'
NEW_BATCH_VALID_FILE_NAME = 'batchValid2id.txt'

ENTITIES_FILE_NAME = 'entity2id.txt'
TRIPLES_FILE_NAME = 'train2id.txt'
TEST_FILE_NAME = 'test2id.txt'
VALID_FILE_NAME = 'valid2id.txt'

ENTITY_EMBEDDING_TENSOR_NAME = 'ent_embeddings'


def update_entities_and_model():
    n_entities = 0
    n_new_entities = 0
    final_entity_size = 0
    batch_entities = []
    entity_lines = []

    ######### READ NEW ENTITIES #########
    with open(sys.argv.input_path+NEW_BATCH_ENTITIES_FILE_NAME, 'r') as f:
        n_new_entities = int(f.readline().strip())
        if n_new_entities > 0:
            for _ in range(n_new_entities):
                batch_entities.append(f.readline())


    ######### READ OLD ENTITIES #########
    with open(sys.argv.input_path+ENTITIES_FILE_NAME) as f:
        entity_lines = f.readlines()
    n_entities = int(entity_lines[0])
    final_entity_size = n_entities + n_new_entities
    if DEBUG: print("Number of new entities in batch: " + str(n_new_entities))


    ######### UPDATE THE MODEL #########
    if n_new_entities > 0:
        con, ckpt = distribute_training.get_conf_to_update_model(sys.argv.output_path)
        vars = []

        if DEBUG: print("\nGLOBAL VARS FOUNDED IN CHECKPOINT:\n")
        with con.graph.as_default():
            with con.sess.as_default():
                for v in tf.global_variables():
                    if DEBUG: print(str(v.name) + " " + str(v.shape))
                    vars.append(v)
        if DEBUG: print('\n')


        if DEBUG: print("NEW GLOBAL VARIABLES")
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            with sess.as_default():
                for v in vars:
                    current_name = v.name.split(':')[0]

                    if current_name == ENTITY_EMBEDDING_TENSOR_NAME:
                        tmp = tf.get_variable(name=current_name, shape=[final_entity_size, v.shape[1]], initializer=tf.contrib.layers.xavier_initializer(uniform = False), dtype=v.dtype)
                        sess.run(tf.initialize_variables([tmp]))
                        tmp_value = con.sess.run(v)
                        sess.run(tf.scatter_update(tmp, [i for i in range(0, n_entities)], tmp_value))

                    elif current_name in [ENTITY_EMBEDDING_TENSOR_NAME+'/Adam', ENTITY_EMBEDDING_TENSOR_NAME+'/Adam_1']:
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

        #update entity2id
        with open(sys.argv.input_path+ENTITIES_FILE_NAME, "w") as f:
            f.writelines(entity_lines)
    if DEBUG: print("Entity file updated")

    return n_new_entities, final_entity_size


def update_triples(file_name_to_update, file_name_batch):
    batch_triples_size = 0
    batch_triples = []

    #open batch file
    with open(sys.argv.input_path + file_name_batch) as f:
        batch_triples_size = int(f.readline().strip())
        if batch_triples_size > 0:
            for _ in range(batch_triples_size):
                batch_triples.append(f.readline())
    print("Number of new lines: " + str(batch_triples_size))

    if batch_triples_size > 0:
        #open file
        with open(sys.argv.input_path + file_name_to_update) as f:
            lines = f.readlines()

        #update number of triples
        lines[0] = str(int(lines[0]) + batch_triples_size) + "\n"

        #update triples
        lines = lines + batch_triples       #append the new triples at the end of the file

        #update file
        with open(sys.argv.input_path + file_name_to_update, "w") as f:
            f.writelines(lines)
    if DEBUG: print("File updated")


def feed_batch():
    current_time = time.time()
    if DEBUG: print("New batch file founded")
    try:
        if DEBUG: print("Updating "+ENTITIES_FILE_NAME+" and model tensors...")
        update_entities_and_model()

        if DEBUG: print("Updating "+TRIPLES_FILE_NAME+"...")
        update_triples(TRIPLES_FILE_NAME, NEW_BATCH_TRIPLES_FILE_NAME)

        if DEBUG: print("Updating "+TEST_FILE_NAME+"...")
        update_triples(TEST_FILE_NAME, NEW_BATCH_TEST_FILE_NAME)

        if DEBUG: print("Updating "+VALID_FILE_NAME+"...")
        update_triples(VALID_FILE_NAME, NEW_BATCH_VALID_FILE_NAME)

        elapsed_time = time.time() - current_time
        print("Time elapsed for new entities initialization: {}".format(elapsed_time))

        return elapsed_time

    except Exception as e:
        print("Error occured while feeding new batch:")
        traceback.print_exc()
        print(e)


def is_new_batch():
    return os.path.isfile(sys.argv.input_path+NEW_BATCH_TRIPLES_FILE_NAME) and \
           os.path.isfile(sys.argv.input_path+NEW_BATCH_ENTITIES_FILE_NAME) and \
            os.path.isfile(sys.argv.input_path+NEW_BATCH_TEST_FILE_NAME) and \
           os.path.isfile(sys.argv.input_path+NEW_BATCH_VALID_FILE_NAME)


def remove_batch_files():
    os.remove(sys.argv.input_path+NEW_BATCH_TRIPLES_FILE_NAME)
    os.remove(sys.argv.input_path+NEW_BATCH_ENTITIES_FILE_NAME)
    os.remove(sys.argv.input_path+NEW_BATCH_TEST_FILE_NAME)
    os.remove(sys.argv.input_path+NEW_BATCH_VALID_FILE_NAME)


def n_n():
    lef = {}
    rig = {}
    rellef = {}
    relrig = {}

    triple = open(sys.argv.input_path+"train2id.txt", "r")
    valid = open(sys.argv.input_path+"valid2id.txt", "r")
    test = open(sys.argv.input_path+"test2id.txt", "r")

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

    f = open(sys.argv.input_path+"type_constrain.txt", "w")
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
    if DEBUG: print("Creating Spark Context...")
    sc = SparkContext(conf=SparkConf().setAppName('OpenKEonSpark'))


    if DEBUG: print("Parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=int(sc._conf.get("spark.executor.instances")))
    parser.add_argument("--num_ps", help="number of ps nodes", type=int, default=1)
    parser.add_argument("--num_gpus", help="number of gpus to use", type=int, default=0)
    parser.add_argument("--cpp_lib_path", help="cpp lib.so absolute path", type=str, default=None)
    parser.add_argument("--input_path", help="dataset absolute path", type=str, default="drive/My\ Drive/DBpedia/5/0/")
    parser.add_argument("--output_path", help="model output absolute path", type=str, default="OpenKE_new_Spark/res_spark")
    parser.add_argument("--train_times", help="no. epochs", type=int, default=1000)
    parser.add_argument("--n_batches", help="no. batches", type=int, default=0)
    parser.add_argument("--alpha", help="learning rate", type=float, default=0.001)
    parser.add_argument("--margin", help="margin hyperparameter used during training", type=float, default=1.0)
    parser.add_argument("--bern_flag", help="use bern method for sampling", type=bool, default=False)
    parser.add_argument("--embedding_dimension", help="embedding dimension (both entities and rel)", type=int, default=64)
    parser.add_argument("--ent_dimension", help="entities embedding dimension", type=int, default=0)
    parser.add_argument("--rel_dimension", help="relations embedding dimension", type=int, default=0)
    parser.add_argument("--ent_neg_rate", help="number of negative triples generated by corrupting the entity", type=int, default=1)
    parser.add_argument("--rel_neg_rate", help="number of negative triples generated by corrupting the realtion", type=int, default=0)
    parser.add_argument("--optimizer", help="Optimization algorithm", type=str, default="SGD")
    parser.add_argument("--early_stop_patience", help="no. epochs to wait for loss improvement before early stop", type=int, default=5)
    parser.add_argument("--early_stop_stopping_step", help="perfrom early stop each stopping step", type=int, default=1)
    parser.add_argument("--early_stop_start_step", help="perfrom early stop from start step", type=int, default=1)
    parser.add_argument("--model", help="model to be used", type=str, default="TransE")


    (args, remainder) = parser.parse_known_args()
    num_workers = args.cluster_size - args.num_ps
    print("===== num_executors={}, num_workers={}, num_ps={}".format(args.cluster_size, num_workers, args.num_ps))


    if DEBUG: print("Setting batch files if present...")
    sys.argv = args
    if is_new_batch(): feed_batch()


    if DEBUG: print("Generating type files...")
    n_n()


    if DEBUG: print("Removing stop file...")
    try:
        os.remove(args.output_path+"/stop.txt")
    except:
        pass


    if DEBUG: print("Creating cluster...")
    training_time = time.time()
    cluster = TFCluster.run(sc, distribute_training.main_fun, args, args.cluster_size, args.num_ps, True, TFCluster.InputMode.TENSORFLOW)



    if DEBUG: print("Shutdowning cluster...")
    cluster.shutdown()
    training_time = time.time() - training_time



    if DEBUG: print("Removing batch files if present...")
    if is_new_batch(): remove_batch_files()



    if DEBUG: print("Printing time information on files...")
    with open(args.output_path+'/time.txt', 'w') as f:
        f.write("Training time: " + str(training_time) + "\n")



    if DEBUG: print("Restoring the best model founded during training...")
    if path.exists(args.output_path+"/stop.txt"):
        step = None
        with open(args.output_path+"/stop.txt", "r") as f:
            step = int(f.readline().strip())

        if step != None:
            min_diff = sys.maxsize
            nearest = None
            with open(args.output_path+"/checkpoint", "r") as ckpt_file:
                lines = ckpt_file.readlines()

            for line in lines:
                l = line.replace('"','').split("/")
                n = int(l[len(l) - 1].split("-")[1])
                diff = abs(step - n)
                if diff < min_diff:
                    nearest = n
                    min_diff = diff

            if nearest != None:
                with open(args.output_path+"/checkpoint", "w") as ckpt_file:
                    ckpt_file.write('model_checkpoint_path: "'+args.output_path+'/model.ckpt-'+str(nearest)+'"\n')
                    ckpt_file.write('all_model_checkpoint_paths: "'+args.output_path+'/model.ckpt-'+str(nearest)+'"\n')

            for f in os.listdir(args.output_path):
                if f.startswith("model") and len(f.split("model.ckpt-"+str(nearest))) == 1:
                    os.remove(args.output_path+"/"+f)


    if DEBUG: print("Training finished")

