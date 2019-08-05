#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import ctypes
import json
import sys
import threading
import time

class Config(object):
    '''
    use ctypes to call C functions from python and set essential parameters.
    '''
    #EDIT
    def __init__(self, cpp_lib_path=None, init_new_entities=False):
        self.init_new_entities = init_new_entities

        if init_new_entities == False:
            if cpp_lib_path == None:
                cpp_lib_path = '/home/luigi/IdeaProjects/OpenKEonSpark/release/Base.so'
            base_file = os.path.abspath(cpp_lib_path)
            self.lib = ctypes.cdll.LoadLibrary(base_file)
            self.lib.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

            #link prediction
            self.lib.getTailBatch.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.testTail.argtypes = [ctypes.c_int64, ctypes.c_void_p]
            self.lib.testTail.restype = ctypes.POINTER(ctypes.c_int64 * 4)
            self.lib.getHeadBatch.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.testHead.argtypes = [ctypes.c_int64, ctypes.c_void_p]
            self.lib.testHead.restype = ctypes.POINTER(ctypes.c_int64 * 4)
            self.test_head = 0

            #triple classification
            self.lib.getTestBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.getValidBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.getBestThreshold.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.test_triple_classification.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

            self.in_path = None
            self.test_log_path = None
            self.out_path = None
            self.bern = 0
            self.hidden_size = 64
            self.ent_size = self.hidden_size
            self.rel_size = self.hidden_size
            self.train_times = 0
            self.margin = 1.0
            self.nbatches = 100
            self.negative_ent = 1
            self.negative_rel = 0
            self.workThreads = 8
            self.alpha = 0.001
            self.lmbda = 0.000
            self.log_on = 1
            self.exportName = None
            self.importName = None
            self.export_steps = 0
            self.opt_method = "SGD"
            self.optimizer = None
            self.test_link_prediction = False
            self.test_triple_classification = False
            self.valid_triple_classification = False

    def init_link_prediction(self):
        r'''
        import essential files and set essential interfaces for link prediction
        '''
        self.lib.importTestFiles()
        self.lib.importTypeFiles()
        self.N_THREADS_LP = 10
        self.lp_res = []
        for _ in range(self.N_THREADS_LP): self.lp_res.append({})

    def init_triple_classification(self):
        r'''
        import essential files and set essential interfaces for triple classification
        '''
        self.lib.importTestFiles()
        self.lib.importTypeFiles()

        self.acc = np.zeros(1, dtype = np.float32)
        self.acc_addr = self.acc.__array_interface__['data'][0]

        self.test_pos_h = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
        self.test_pos_t = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
        self.test_pos_r = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
        self.test_neg_h = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
        self.test_neg_t = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
        self.test_neg_r = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
        self.test_pos_h_addr = self.test_pos_h.__array_interface__['data'][0]
        self.test_pos_t_addr = self.test_pos_t.__array_interface__['data'][0]
        self.test_pos_r_addr = self.test_pos_r.__array_interface__['data'][0]
        self.test_neg_h_addr = self.test_neg_h.__array_interface__['data'][0]
        self.test_neg_t_addr = self.test_neg_t.__array_interface__['data'][0]
        self.test_neg_r_addr = self.test_neg_r.__array_interface__['data'][0]

        self.valid_pos_h = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
        self.valid_pos_t = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
        self.valid_pos_r = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
        self.valid_neg_h = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
        self.valid_neg_t = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
        self.valid_neg_r = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
        self.valid_pos_h_addr = self.valid_pos_h.__array_interface__['data'][0]
        self.valid_pos_t_addr = self.valid_pos_t.__array_interface__['data'][0]
        self.valid_pos_r_addr = self.valid_pos_r.__array_interface__['data'][0]
        self.valid_neg_h_addr = self.valid_neg_h.__array_interface__['data'][0]
        self.valid_neg_t_addr = self.valid_neg_t.__array_interface__['data'][0]
        self.valid_neg_r_addr = self.valid_neg_r.__array_interface__['data'][0]

        self.relThresh = np.zeros(self.lib.getRelationTotal(), dtype = np.float32)
        self.relThresh_addr = self.relThresh.__array_interface__['data'][0]


    def init_valid_triple_classification(self):
        self.lib.importTestFiles()
        self.lib.importTypeFiles()

        self.acc = np.zeros(1, dtype = np.float32)
        self.acc_addr = self.acc.__array_interface__['data'][0]

        self.valid_pos_h = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
        self.valid_pos_t = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
        self.valid_pos_r = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
        self.valid_neg_h = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
        self.valid_neg_t = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
        self.valid_neg_r = np.zeros(self.lib.getValidTotal(), dtype = np.int64)

        self.valid_pos_h_addr = self.valid_pos_h.__array_interface__['data'][0]
        self.valid_pos_t_addr = self.valid_pos_t.__array_interface__['data'][0]
        self.valid_pos_r_addr = self.valid_pos_r.__array_interface__['data'][0]
        self.valid_neg_h_addr = self.valid_neg_h.__array_interface__['data'][0]
        self.valid_neg_t_addr = self.valid_neg_t.__array_interface__['data'][0]
        self.valid_neg_r_addr = self.valid_neg_r.__array_interface__['data'][0]

        self.relThresh = np.zeros(self.lib.getRelationTotal(), dtype = np.float32)
        self.relThresh_addr = self.relThresh.__array_interface__['data'][0]


    # prepare for train and test
    def init(self):
        if self.init_new_entities == False:
            self.trainModel = None
            if self.in_path != None:
                self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
                self.lib.setBern(self.bern)
                self.lib.setWorkThreads(self.workThreads)
                self.lib.randReset()
                self.lib.importTrainFiles()
                self.relTotal = self.lib.getRelationTotal()
                self.entTotal = self.lib.getEntityTotal()
                self.trainTotal = self.lib.getTrainTotal_()
                self.testTotal = self.lib.getTestTotal()
                self.validTotal = self.lib.getValidTotal()
                self.bt = self.lib.getBatchTotal()

                tot = None

                if self.bt > 0:
                    tot = self.bt
                else:
                    tot = self.trainTotal

                if self.nbatches > 0:
                    self.batch_size = int(tot / self.nbatches)
                else:
                    self.batch_size = tot
                    while self.batch_size > 9999:
                        self.batch_size = int(self.batch_size / 10)
                    self.nbatches = int(tot / self.batch_size)

                print("Batch size is {}".format(self.batch_size))
                print("Number of batches: {}".format(self.nbatches))

                self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)
                self.batch_h = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
                self.batch_t = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
                self.batch_r = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
                self.batch_y = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.float32)
                self.batch_h_addr = self.batch_h.__array_interface__['data'][0]
                self.batch_t_addr = self.batch_t.__array_interface__['data'][0]
                self.batch_r_addr = self.batch_r.__array_interface__['data'][0]
                self.batch_y_addr = self.batch_y.__array_interface__['data'][0]


            if self.test_link_prediction:
                self.init_link_prediction()
            if self.test_triple_classification:
                self.init_triple_classification()
            if self.valid_triple_classification:
                self.init_valid_triple_classification()

    def set_test_log_path(self, p):
        self.test_log_path = p

    def get_ent_total(self):
        return self.entTotal

    def get_rel_total(self):
        return self.relTotal

    def set_lmbda(self, lmbda):
        self.lmbda = lmbda

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_opt_method(self, method):
        self.opt_method = method

    def set_test_link_prediction(self, flag):
        self.test_link_prediction = flag

    def set_test_triple_classification(self, flag):
        self.test_triple_classification = flag

    def set_valid_triple_classification(self, flag):
        self.valid_triple_classification = flag

    def set_log_on(self, flag):
        self.log_on = flag

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_in_path(self, path):
        self.in_path = path

    def set_out_files(self, path):
        self.out_path = path

    def set_bern(self, bern):
        self.bern = bern

    def set_test_head(self, test_head):
        self.test_head = test_head

    def set_dimension(self, dim):
        self.hidden_size = dim
        self.ent_size = dim
        self.rel_size = dim

    def set_ent_dimension(self, dim):
        self.ent_size = dim

    def set_rel_dimension(self, dim):
        self.rel_size = dim

    def set_train_times(self, times):
        self.train_times = times

    def set_nbatches(self, nbatches):
        self.nbatches = nbatches

    def set_margin(self, margin):
        self.margin = margin

    def set_ent_neg_rate(self, rate):
        self.negative_ent = rate

    def set_rel_neg_rate(self, rate):
        self.negative_rel = rate

    def set_import_files(self, path):
        self.importName = path

    def set_export_files(self, path, steps = 0):
        self.exportName = path
        self.export_steps = steps

    def set_export_steps(self, steps):
        self.export_steps = steps

    # call C function for sampling
    def sampling(self):
        self.lib.sampling(self.batch_h_addr, self.batch_t_addr, self.batch_r_addr, self.batch_y_addr, self.batch_size, self.negative_ent, self.negative_rel)

    # save model
    def save_tensorflow(self):
        with self.graph.as_default():
            with self.sess.as_default():
                self.saver.save(self.sess, self.exportName)

    def save_tensorflow_weights(self, export_name=None, write_meta_graph=False):
        if export_name == None:
            export_name = self.exportName
        with self.graph.as_default():
            with self.sess.as_default():
                self.saver.save(self.sess, export_name, write_meta_graph=write_meta_graph, write_state=False)

    def restore_tensorflow(self):
        with self.graph.as_default():
            with self.sess.as_default():
                self.saver.restore(self.sess, self.importName)


    def export_variables(self, path = None):
        with self.graph.as_default():
            with self.sess.as_default():
                if path == None:
                    self.saver.save(self.sess, self.exportName)
                else:
                    self.saver.save(self.sess, path)

    def import_variables(self, path = None):
        with self.graph.as_default():
            with self.sess.as_default():
                if path == None:
                    self.saver.restore(self.sess, self.importName)
                else:
                    self.saver.restore(self.sess, path)

    def get_parameter_lists(self):
        return self.trainModel.parameter_lists

    def get_parameters_by_name(self, var_name):
        with self.graph.as_default():
            with self.sess.as_default():
                if var_name in self.trainModel.parameter_lists:
                    return self.sess.run(self.trainModel.parameter_lists[var_name])
                else:
                    return None

    def get_parameters(self, mode = "numpy"):
        res = {}
        lists = self.get_parameter_lists()
        for var_name in lists:
            if mode == "numpy":
                res[var_name] = self.get_parameters_by_name(var_name)
            else:
                res[var_name] = self.get_parameters_by_name(var_name).tolist()
        return res

    def save_parameters(self, path = None):
        if path == None:
            path = self.out_path
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    def set_parameters_by_name(self, var_name, tensor):
        with self.graph.as_default():
            with self.sess.as_default():
                if var_name in self.trainModel.parameter_lists:
                    self.trainModel.parameter_lists[var_name].assign(tensor).eval()

    def set_parameters(self, lists):
        for i in lists:
            self.set_parameters_by_name(i, lists[i])

    def set_model(self, model):
        self.model = model


    def import_model(self, ckpt):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                print("Importing metagraph...")
                self.saver = tf.train.import_meta_graph(ckpt+".meta", clear_devices=True)
                print("Importing variables...")
                self.saver.restore(self.sess, ckpt)


    def set_model_and_session(self, model):
        self.model = model
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                initializer = tf.contrib.layers.xavier_initializer(uniform = True)
                with tf.variable_scope("", reuse=None, initializer = initializer):
                    self.trainModel = self.model(config = self, define=True)
                self.saver = tf.train.Saver()
                self.sess.run(tf.initialize_all_variables())


    def train_step(self, batch_h, batch_t, batch_r, batch_y):
        feed_dict = {
            self.trainModel.batch_h: batch_h,
            self.trainModel.batch_t: batch_t,
            self.trainModel.batch_r: batch_r,
            self.trainModel.batch_y: batch_y
        }
        _, loss = self.sess.run([self.train_op, self.trainModel.loss], feed_dict)
        return loss


    def test_step(self, test_h, test_t, test_r):
        feed_dict = {
            self.trainModel.predict_h: test_h,
            self.trainModel.predict_t: test_t,
            self.trainModel.predict_r: test_r,
        }
        predict = self.sess.run(self.trainModel.predict, feed_dict)
        return predict



    def test_lp_range(self, index, lef, rig):
        l1_filter_tot = l1_tot = r1_tot = r1_filter_tot = l_tot = r_tot = l_filter_rank = l_rank = l_filter_reci_rank = l_reci_rank = 0.0
        l3_filter_tot = l3_tot = r3_tot = r3_filter_tot = l_filter_tot = r_filter_tot = r_filter_rank = r_rank = r_filter_reci_rank = r_reci_rank = 0.0

        #TYPE_C
        l1_filter_tot_constrain = l1_tot_constrain = r1_tot_constrain = r1_filter_tot_constrain = l_tot_constrain = r_tot_constrain = l_filter_rank_constrain = l_rank_constrain = l_filter_reci_rank_constrain = l_reci_rank_constrain = 0.0
        l3_filter_tot_constrain = l3_tot_constrain = r3_tot_constrain = r3_filter_tot_constrain = l_filter_tot_constrain = r_filter_tot_constrain = r_filter_rank_constrain = r_rank_constrain = r_filter_reci_rank_constrain = r_reci_rank_constrain = 0.0


        test_h = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
        test_t = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
        test_r = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
        test_h_addr = test_h.__array_interface__['data'][0]
        test_t_addr = test_t.__array_interface__['data'][0]
        test_r_addr = test_r.__array_interface__['data'][0]

        print("Test link prediction range from {} to {}".format(lef, rig-1))

        if os.path.exists(self.test_log_path+"thread"+str(index)):
            with open(self.test_log_path+"thread"+str(index), 'r') as f:
                last_i = int(f.readline())
                print("Restoring test results from index {}".format(last_i))

                lef = last_i + 1

                r_filter_tot = float(f.readline())
                r_tot = float(f.readline())
                r3_filter_tot = float(f.readline())
                r3_tot = float(f.readline())
                r1_filter_tot = float(f.readline())
                r1_tot = float(f.readline())
                r_filter_rank = float(f.readline())
                r_rank = float(f.readline())
                r_filter_reci_rank = float(f.readline())
                r_reci_rank = float(f.readline())
                r_filter_tot_constrain = float(f.readline())
                r_tot_constrain = float(f.readline())
                r3_filter_tot_constrain = float(f.readline())
                r3_tot_constrain = float(f.readline())
                r1_filter_tot_constrain = float(f.readline())
                r1_tot_constrain = float(f.readline())
                r_filter_rank_constrain = float(f.readline())
                r_rank_constrain = float(f.readline())
                r_filter_reci_rank_constrain = float(f.readline())
                r_reci_rank_constrain = float(f.readline())

                if self.test_head != 0:
                    l_filter_tot = float(f.readline())
                    l_tot = float(f.readline())
                    l3_filter_tot = float(f.readline())
                    l3_tot = float(f.readline())
                    l1_filter_tot = float(f.readline())
                    l1_tot = float(f.readline())
                    l_filter_rank = float(f.readline())
                    l_rank = float(f.readline())
                    l_filter_reci_rank = float(f.readline())
                    l_reci_rank = float(f.readline())
                    l_filter_tot_constrain = float(f.readline())
                    l_tot_constrain = float(f.readline())
                    l3_filter_tot_constrain = float(f.readline())
                    l3_tot_constrain = float(f.readline())
                    l1_filter_tot_constrain = float(f.readline())
                    l1_tot_constrain = float(f.readline())
                    l_filter_rank_constrain = float(f.readline())
                    l_rank_constrain = float(f.readline())
                    l_filter_reci_rank_constrain = float(f.readline())
                    l_reci_rank_constrain = float(f.readline())

        test_triples_done = 0
        for i in range(lef, rig):
            #tail
            self.lib.getTailBatch(i, test_h_addr, test_t_addr, test_r_addr)
            res = self.test_step(test_h, test_t, test_r)
            test_tail_res = [j for j in self.lib.testTail(i, res.__array_interface__['data'][0]).contents]
            r_s = test_tail_res[0]
            r_filter_s = test_tail_res[1]
            r_s_constrain = test_tail_res[2]
            r_filter_s_constrain = test_tail_res[3]

            if (r_filter_s < 10): r_filter_tot += 1
            if (r_s < 10): r_tot += 1
            if (r_filter_s < 3): r3_filter_tot += 1
            if (r_s < 3): r3_tot += 1
            if (r_filter_s < 1): r1_filter_tot += 1
            if (r_s < 1): r1_tot += 1

            r_filter_rank += (1+r_filter_s)
            r_rank += (1+r_s)
            r_filter_reci_rank += np.divide(1.0, (1+r_filter_s))
            r_reci_rank += np.divide(1.0, (1+r_s))

            #TYPE_C
            if (r_filter_s_constrain < 10): r_filter_tot_constrain += 1
            if (r_s_constrain < 10): r_tot_constrain += 1
            if (r_filter_s_constrain < 3): r3_filter_tot_constrain += 1
            if (r_s_constrain < 3): r3_tot_constrain += 1
            if (r_filter_s_constrain < 1): r1_filter_tot_constrain += 1
            if (r_s_constrain < 1): r1_tot_constrain += 1

            r_filter_rank_constrain += (1+r_filter_s_constrain)
            r_rank_constrain += (1+r_s_constrain)
            r_filter_reci_rank_constrain += np.divide(1.0, (1+r_filter_s_constrain))
            r_reci_rank_constrain += np.divide(1.0, (1+r_s_constrain))

            #head
            if self.test_head != 0:
                #head
                self.lib.getHeadBatch(i, test_h_addr, test_t_addr, test_r_addr)
                res = self.test_step(test_h, test_t, test_r)
                test_head_res = [j for j in self.lib.testHead(i, res.__array_interface__['data'][0]).contents]
                l_s = test_head_res[0]
                l_filter_s = test_head_res[1]
                l_s_constrain = test_head_res[2]
                l_filter_s_constrain = test_head_res[3]

                if (l_filter_s < 10): l_filter_tot += 1
                if (l_s < 10): l_tot += 1
                if (l_filter_s < 3): l3_filter_tot += 1
                if (l_s < 3): l3_tot += 1
                if (l_filter_s < 1): l1_filter_tot += 1
                if (l_s < 1): l1_tot += 1

                l_filter_rank += (l_filter_s+1)
                l_rank += (1+l_s)
                l_filter_reci_rank += np.divide(1.0, (l_filter_s+1))
                l_reci_rank += np.divide(1.0, (l_s+1))

                #TYPE_C
                if (l_filter_s_constrain < 10): l_filter_tot_constrain += 1
                if (l_s_constrain < 10): l_tot_constrain += 1
                if (l_filter_s_constrain < 3): l3_filter_tot_constrain += 1
                if (l_s_constrain < 3): l3_tot_constrain += 1
                if (l_filter_s_constrain < 1): l1_filter_tot_constrain += 1
                if (l_s_constrain < 1): l1_tot_constrain += 1

                l_filter_rank_constrain += (l_filter_s_constrain+1)
                l_rank_constrain += (1+l_s_constrain)
                l_filter_reci_rank_constrain += np.divide(1.0, (l_filter_s_constrain+1))
                l_reci_rank_constrain += np.divide(1.0, (l_s_constrain+1))

            if index == 0: sys.stdout.write("\r# of test triples processed: {}".format(i * self.N_THREADS_LP))

            test_triples_done += 1


            if test_triples_done % 100 == 0:
                with open(self.test_log_path+"thread"+str(index), "w") as f:
                    f.write(str(i)+'\n')

                    f.write(str(r_filter_tot)+'\n')
                    f.write(str(r_tot)+'\n')
                    f.write(str(r3_filter_tot)+'\n')
                    f.write(str(r3_tot)+'\n')
                    f.write(str(r1_filter_tot)+'\n')
                    f.write(str(r1_tot)+'\n')
                    f.write(str(r_filter_rank)+'\n')
                    f.write(str(r_rank)+'\n')
                    f.write(str(r_filter_reci_rank)+'\n')
                    f.write(str(r_reci_rank)+'\n')
                    f.write(str(r_filter_tot_constrain)+'\n')
                    f.write(str(r_tot_constrain)+'\n')
                    f.write(str(r3_filter_tot_constrain)+'\n')
                    f.write(str(r3_tot_constrain)+'\n')
                    f.write(str(r1_filter_tot_constrain)+'\n')
                    f.write(str(r1_tot_constrain)+'\n')
                    f.write(str(r_filter_rank_constrain)+'\n')
                    f.write(str(r_rank_constrain)+'\n')
                    f.write(str(r_filter_reci_rank_constrain)+'\n')
                    f.write(str(r_reci_rank_constrain)+'\n')

                    if self.test_head != 0:
                        f.write(str(l_filter_tot)+'\n')
                        f.write(str(l_tot)+'\n')
                        f.write(str(l3_filter_tot)+'\n')
                        f.write(str(l3_tot)+'\n')
                        f.write(str(l1_filter_tot)+'\n')
                        f.write(str(l1_tot)+'\n')
                        f.write(str(l_filter_rank)+'\n')
                        f.write(str(l_rank)+'\n')
                        f.write(str(l_filter_reci_rank)+'\n')
                        f.write(str(l_reci_rank)+'\n')
                        f.write(str(l_filter_tot_constrain)+'\n')
                        f.write(str(l_tot_constrain)+'\n')
                        f.write(str(l3_filter_tot_constrain)+'\n')
                        f.write(str(l3_tot_constrain)+'\n')
                        f.write(str(l1_filter_tot_constrain)+'\n')
                        f.write(str(l1_tot_constrain)+'\n')
                        f.write(str(l_filter_rank_constrain)+'\n')
                        f.write(str(l_rank_constrain)+'\n')
                        f.write(str(l_filter_reci_rank_constrain)+'\n')
                        f.write(str(l_reci_rank_constrain)+'\n')


        #tail
        self.lp_res[index] = {'r_filter_tot':r_filter_tot,
                              'r_tot':r_tot,
                              'r3_filter_tot':r3_filter_tot,
                              'r3_tot':r3_tot,
                              'r1_filter_tot':r1_filter_tot,
                              'r1_tot':r1_tot,

                              'r_filter_rank':r_filter_rank,
                              'r_rank':r_rank,
                              'r_filter_reci_rank':r_filter_reci_rank,
                              'r_reci_rank':r_reci_rank,

                              'r_filter_tot_constrain':r_filter_tot_constrain,
                              'r_tot_constrain':r_tot_constrain,
                              'r3_filter_tot_constrain':r3_filter_tot_constrain,
                              'r3_tot_constrain': r3_tot_constrain,
                              'r1_filter_tot_constrain':r1_filter_tot_constrain,
                              'r1_tot_constrain':r1_tot_constrain,

                              'r_filter_rank_constrain':r_filter_rank_constrain,
                              'r_rank_constrain':r_rank_constrain,
                              'r_filter_reci_rank_constrain':r_filter_reci_rank_constrain,
                              'r_reci_rank_constrain':r_reci_rank_constrain}

        #head
        if self.test_head != 0:
            self.lp_res[index]['l_filter_tot'] = l_filter_tot
            self.lp_res[index]['l_tot'] = l_tot
            self.lp_res[index]['l3_filter_tot'] = l3_filter_tot
            self.lp_res[index]['l3_tot'] = l3_tot
            self.lp_res[index]['l1_filter_tot'] = l1_filter_tot
            self.lp_res[index]['l1_tot'] = l1_tot

            self.lp_res[index]['l_filter_rank'] = l_filter_rank
            self.lp_res[index]['l_rank'] = l_rank
            self.lp_res[index]['l_filter_reci_rank'] = l_filter_reci_rank
            self.lp_res[index]['l_reci_rank'] = l_reci_rank

            self.lp_res[index]['l_filter_tot_constrain'] = l_filter_tot_constrain
            self.lp_res[index]['l_tot_constrain'] = l_tot_constrain
            self.lp_res[index]['l3_filter_tot_constrain'] = l3_filter_tot_constrain
            self.lp_res[index]['l3_tot_constrain'] = l3_tot_constrain
            self.lp_res[index]['l1_filter_tot_constrain'] = l1_filter_tot_constrain
            self.lp_res[index]['l1_tot_constrain'] = l1_tot_constrain

            self.lp_res[index]['l_filter_rank_constrain'] = l_filter_rank_constrain
            self.lp_res[index]['l_rank_constrain'] = l_rank_constrain
            self.lp_res[index]['l_filter_reci_rank_constrain'] = l_filter_reci_rank_constrain
            self.lp_res[index]['l_reci_rank_constrain'] = l_reci_rank_constrain


    def test(self):
        with self.graph.as_default():
            with self.sess.as_default():
                if self.importName != None:
                    self.restore_tensorflow()

                test_time_start = time.time()

                if self.test_link_prediction:
                    l1_filter_tot = l1_tot = r1_tot = r1_filter_tot = l_tot = r_tot = l_filter_rank = l_rank = l_filter_reci_rank = l_reci_rank = 0.0
                    l3_filter_tot = l3_tot = r3_tot = r3_filter_tot = l_filter_tot = r_filter_tot = r_filter_rank = r_rank = r_filter_reci_rank = r_reci_rank = 0.0
                    #TYPE_C
                    l1_filter_tot_constrain = l1_tot_constrain = r1_tot_constrain = r1_filter_tot_constrain = l_tot_constrain = r_tot_constrain = l_filter_rank_constrain = l_rank_constrain = l_filter_reci_rank_constrain = l_reci_rank_constrain = 0.0
                    l3_filter_tot_constrain = l3_tot_constrain = r3_tot_constrain = r3_filter_tot_constrain = l_filter_tot_constrain = r_filter_tot_constrain = r_filter_rank_constrain = r_rank_constrain = r_filter_reci_rank_constrain = r_reci_rank_constrain = 0.0


                    testTotal = self.lib.getTestTotal()
                    triples_per_thread = int(testTotal / self.N_THREADS_LP)
                    print("Number of test triples: {} Number of triples per test thread: {}".format(testTotal, triples_per_thread))
                    threads_array = []

                    lef = 0
                    rig = 0
                    for j in range(self.N_THREADS_LP):
                        if j+1 == self.N_THREADS_LP:
                            rig = testTotal
                        else:
                            rig += triples_per_thread
                        threads_array.append(threading.Thread(target=self.test_lp_range, args=(j, lef, rig, )))
                        lef = rig


                    for t in threads_array:
                        t.start()

                    idx = 0
                    for t in threads_array:
                        t.join()

                    for res in self.lp_res:
                        #tail
                        r_filter_tot += res['r_filter_tot']
                        r_tot += res['r_tot']
                        r3_filter_tot += res['r3_filter_tot']
                        r3_tot += res['r3_tot']
                        r1_filter_tot += res['r1_filter_tot']
                        r1_tot += res['r1_tot']
                        r_filter_rank += res['r_filter_rank']
                        r_rank += res['r_rank']
                        r_filter_reci_rank += res['r_filter_reci_rank']
                        r_reci_rank += res['r_reci_rank']

                        #tail TYPE_C
                        r_filter_tot_constrain += res['r_filter_tot_constrain']
                        r_tot_constrain += res['r_tot_constrain']
                        r3_filter_tot_constrain += res['r3_filter_tot_constrain']
                        r3_tot_constrain += res['r3_tot_constrain']
                        r1_filter_tot_constrain += res['r1_filter_tot_constrain']
                        r1_tot_constrain += res['r1_tot_constrain']
                        r_filter_rank_constrain += res['r_filter_rank_constrain']
                        r_rank_constrain += res['r_rank_constrain']
                        r_filter_reci_rank_constrain += res['r_filter_reci_rank_constrain']
                        r_reci_rank_constrain += res['r_reci_rank_constrain']

                        if self.test_head != 0:
                            #head
                            l_filter_tot += res['l_filter_tot']
                            l_tot += res['l_tot']
                            l3_filter_tot += res['l3_filter_tot']
                            l3_tot += res['l3_tot']
                            l1_filter_tot += res['l1_filter_tot']
                            l1_tot += res['l1_tot']
                            l_filter_rank += res['l_filter_rank']
                            l_rank += res['l_rank']
                            l_filter_reci_rank += res['l_filter_reci_rank']
                            l_reci_rank += res['l_reci_rank']

                            #head TYPE_C
                            l_filter_tot_constrain += res['l_filter_tot_constrain']
                            l_tot_constrain += res['l_tot_constrain']
                            l3_filter_tot_constrain += res['l3_filter_tot_constrain']
                            l3_tot_constrain += res['l3_tot_constrain']
                            l1_filter_tot_constrain += res['l1_filter_tot_constrain']
                            l1_tot_constrain += res['l1_tot_constrain']
                            l_filter_rank_constrain += res['l_filter_rank_constrain']
                            l_rank_constrain += res['l_rank_constrain']
                            l_filter_reci_rank_constrain += res['l_filter_reci_rank_constrain']
                            l_reci_rank_constrain += res['l_reci_rank_constrain']


                    #tail
                    r_rank = np.divide(r_rank, testTotal)
                    r_reci_rank = np.divide(r_reci_rank, testTotal)
                    r_tot = np.divide(r_tot, testTotal)
                    r3_tot = np.divide(r3_tot, testTotal)
                    r1_tot = np.divide(r1_tot, testTotal)
                    r_filter_rank = np.divide(r_filter_rank, testTotal)
                    r_filter_reci_rank = np.divide(r_filter_reci_rank, testTotal)
                    r_filter_tot = np.divide(r_filter_tot, testTotal)
                    r3_filter_tot = np.divide(r3_filter_tot, testTotal)
                    r1_filter_tot = np.divide(r1_filter_tot, testTotal)

                    #TYPE_C
                    r_rank_constrain = np.divide(r_rank_constrain, testTotal)
                    r_reci_rank_constrain = np.divide(r_reci_rank_constrain, testTotal)
                    r_tot_constrain = np.divide(r_tot_constrain, testTotal)
                    r3_tot_constrain = np.divide(r3_tot_constrain, testTotal)
                    r1_tot_constrain = np.divide(r1_tot_constrain, testTotal)
                    r_filter_rank_constrain = np.divide(r_filter_rank_constrain, testTotal)
                    r_filter_reci_rank_constrain = np.divide(r_filter_reci_rank_constrain, testTotal)
                    r_filter_tot_constrain = np.divide(r_filter_tot_constrain, testTotal)
                    r3_filter_tot_constrain = np.divide(r3_filter_tot_constrain, testTotal)
                    r1_filter_tot_constrain = np.divide(r1_filter_tot_constrain, testTotal)

                    if self.test_head != 0:
                        #head
                        l_rank = np.divide(l_rank, testTotal)
                        l_reci_rank = np.divide(l_reci_rank, testTotal)
                        l_tot = np.divide(l_tot, testTotal)
                        l3_tot = np.divide(l3_tot, testTotal)
                        l1_tot = np.divide(l1_tot, testTotal)
                        l_filter_rank = np.divide(l_filter_rank, testTotal)
                        l_filter_reci_rank = np.divide(l_filter_reci_rank, testTotal)
                        l_filter_tot = np.divide(l_filter_tot, testTotal)
                        l3_filter_tot = np.divide(l3_filter_tot, testTotal)
                        l1_filter_tot = np.divide(l1_filter_tot, testTotal)

                        #TYPE_C
                        l_rank_constrain = np.divide(l_rank_constrain, testTotal)
                        l_reci_rank_constrain = np.divide(l_reci_rank_constrain, testTotal)
                        l_tot_constrain = np.divide(l_tot_constrain, testTotal)
                        l3_tot_constrain = np.divide(l3_tot_constrain, testTotal)
                        l1_tot_constrain = np.divide(l1_tot_constrain, testTotal)
                        l_filter_rank_constrain = np.divide(l_filter_rank_constrain, testTotal)
                        l_filter_reci_rank_constrain = np.divide(l_filter_reci_rank_constrain, testTotal)
                        l_filter_tot_constrain = np.divide(l_filter_tot_constrain, testTotal)
                        l3_filter_tot_constrain = np.divide(l3_filter_tot_constrain, testTotal)
                        l1_filter_tot_constrain = np.divide(l1_filter_tot_constrain, testTotal)


                    print("\n ========== LINK PREDICTION RESULTS ==========\nNo type constraint results:")
                    print("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format("metric", "MRR", "MR", "hit@10", "hit@3", "hit@1"))
                    if self.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format("l(raw):",l_reci_rank, l_rank, l_tot, l3_tot, l1_tot))
                    print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format("r(raw):", r_reci_rank, r_rank, r_tot, r3_tot, r1_tot))
                    if self.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}\n".format("mean(raw):", np.divide((l_reci_rank+r_reci_rank),2), np.divide((l_rank+r_rank),2), np.divide((l_tot+r_tot),2), np.divide((l3_tot+r3_tot),2), np.divide((l1_tot+r1_tot),2)))

                    if self.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format("l(filter):", l_filter_reci_rank, l_filter_rank, l_filter_tot, l3_filter_tot, l1_filter_tot))
                    print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format('r(filter):', r_filter_reci_rank, r_filter_rank, r_filter_tot, r3_filter_tot, r1_filter_tot))
                    if self.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}\n".format('mean(filter):', np.divide((l_filter_reci_rank+r_filter_reci_rank),2), np.divide((l_filter_rank+r_filter_rank),2), np.divide((l_filter_tot+r_filter_tot),2), np.divide((l3_filter_tot+r3_filter_tot),2), np.divide((l1_filter_tot+r1_filter_tot),2)))

                    print("Type constraint results:")
                    print("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format("metric", "MRR", "MR", "hit@10", "hit@3", "hit@1"))
                    if self.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format('l(raw):', l_reci_rank_constrain, l_rank_constrain, l_tot_constrain, l3_tot_constrain, l1_tot_constrain))
                    print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format('r(raw):', r_reci_rank_constrain, r_rank_constrain, r_tot_constrain, r3_tot_constrain, r1_tot_constrain))
                    if self.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}\n".format('mean(raw):', np.divide((l_reci_rank_constrain+r_reci_rank_constrain),2), np.divide((l_rank_constrain+r_rank_constrain),2), np.divide((l_tot_constrain+r_tot_constrain),2), np.divide((l3_tot_constrain+r3_tot_constrain),2), np.divide((l1_tot_constrain+r1_tot_constrain),2)))

                    if self.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format('l(filter):', l_filter_reci_rank_constrain, l_filter_rank_constrain, l_filter_tot_constrain, l3_filter_tot_constrain, l1_filter_tot_constrain))
                    print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format('r(filter):', r_filter_reci_rank_constrain, r_filter_rank_constrain, r_filter_tot_constrain, r3_filter_tot_constrain, r1_filter_tot_constrain))
                    if self.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}\n".format('mean(filter):', np.divide((l_filter_reci_rank_constrain+r_filter_reci_rank_constrain),2), np.divide((l_filter_rank_constrain+r_filter_rank_constrain),2), np.divide((l_filter_tot_constrain+r_filter_tot_constrain),2), np.divide((l3_filter_tot_constrain+r3_filter_tot_constrain),2), np.divide((l1_filter_tot_constrain+r1_filter_tot_constrain),2)))

                    #remove test checkpoint
                    for index in range(0, self.N_THREADS_LP):
                        try:
                            os.remove(self.test_log_path+"thread"+str(index))
                        except:
                            print("File " + self.test_log_path+"thread"+str(index) + " not founded")


                if self.test_triple_classification:
                    self.lib.getValidBatch(self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr, self.valid_neg_h_addr, self.valid_neg_t_addr, self.valid_neg_r_addr)
                    res_pos = self.test_step(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
                    res_neg = self.test_step(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)
                    self.lib.getBestThreshold(self.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])

                    self.lib.getTestBatch(self.test_pos_h_addr, self.test_pos_t_addr, self.test_pos_r_addr, self.test_neg_h_addr, self.test_neg_t_addr, self.test_neg_r_addr)

                    res_pos = self.test_step(self.test_pos_h, self.test_pos_t, self.test_pos_r)
                    res_neg = self.test_step(self.test_neg_h, self.test_neg_t, self.test_neg_r)

                    self.lib.test_triple_classification(self.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0], self.acc_addr)

                test_time_elapsed = time.time() - test_time_start
                print("\nElapsed test time (seconds): {}".format(test_time_elapsed))





    #EDIT
    def valid(self):
        with self.graph.as_default():
            with self.sess.as_default():
                if self.importName != None:
                    self.restore_tensorflow()
                self.lib.getValidBatch(self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr, self.valid_neg_h_addr, self.valid_neg_t_addr, self.valid_neg_r_addr)
                res_pos = self.test_step(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
                res_neg = self.test_step(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)
                self.lib.getBestThreshold(self.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])

                self.lib.test_triple_classification(self.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0], self.acc_addr)



    def predict_head_entity(self, t, r, k):
        r'''This mothod predicts the top k head entities given tail entity and relation.

        Args:
            t (int): tail entity id
            r (int): relation id
            k (int): top k head entities

        Returns:
            list: k possible head entity ids
        '''
        self.init_link_prediction()
        if self.importName != None:
            self.restore_tensorflow()
        test_h = np.array(range(self.entTotal))
        test_r = np.array([r] * self.entTotal)
        test_t = np.array([t] * self.entTotal)
        res = self.test_step(test_h, test_t, test_r).reshape(-1).argsort()[:k]
        print(res)
        return res


    def predict_tail_entity(self, h, r, k):
        r'''This mothod predicts the top k tail entities given head entity and relation.

        Args:
            h (int): head entity id
            r (int): relation id
            k (int): top k tail entities

        Returns:
            list: k possible tail entity ids
        '''
        self.init_link_prediction()
        if self.importName != None:
            self.restore_tensorflow()
        test_h = np.array([h] * self.entTotal)
        test_r = np.array([r] * self.entTotal)
        test_t = np.array(range(self.entTotal))
        res = self.test_step(test_h, test_t, test_r).reshape(-1).argsort()[:k]
        print(res)
        return res

    def predict_relation(self, h, t, k):
        r'''This methods predict the relation id given head entity and tail entity.

        Args:
            h (int): head entity id
            t (int): tail entity id
            k (int): top k relations

        Returns:
            list: k possible relation ids
        '''
        self.init_link_prediction()
        if self.importName != None:
            self.restore_tensorflow()
        test_h = np.array([h] * self.relTotal)
        test_r = np.array(range(self.relTotal))
        test_t = np.array([t] * self.relTotal)
        res = self.test_step(test_h, test_t, test_r).reshape(-1).argsort()[:k]
        print(res)
        return res

    def predict_triple(self, h, t, r, thresh = None):
        r'''This method tells you whether the given triple (h, t, r) is correct of wrong

        Args:
            h (int): head entity id
            t (int): tail entity id
            r (int): relation id
            thresh (fload): threshold for the triple
        '''
        self.init_triple_classification()
        if self.importName != None:
            self.restore_tensorflow()
        res = self.test_step(np.array([h]), np.array([t]), np.array([r]))
        if thresh != None:
            if res < thresh:
                print("triple (%d,%d,%d) is correct" % (h, t, r))
            else:
                print("triple (%d,%d,%d) is wrong" % (h, t, r))
            return
        self.lib.getValidBatch(self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr, self.valid_neg_h_addr, self.valid_neg_t_addr, self.valid_neg_r_addr)
        res_pos = self.test_step(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
        res_neg = self.test_step(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)
        self.lib.getBestThreshold(self.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])
        if res < self.relThresh[r]:
            print("triple (%d,%d,%d) is correct" % (h, t, r))
        else:
            print("triple (%d,%d,%d) is wrong" % (h, t, r))
