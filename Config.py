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
            #set c++ lib
            if cpp_lib_path == None:
                cpp_lib_path = '/home/luigi/IdeaProjects/OpenKEonSpark/release/Base.so'
            base_file = os.path.abspath(cpp_lib_path)
            self.lib = ctypes.cdll.LoadLibrary(base_file)
            self.lib.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

            #link prediction
            self.lib.getTailBatch.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.testTail.argtypes = [ctypes.c_int64, ctypes.c_void_p]
            self.lib.testTail.restype = ctypes.POINTER(ctypes.c_int64 * 8)
            self.lib.getHeadBatch.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.testHead.argtypes = [ctypes.c_int64, ctypes.c_void_p]
            self.lib.testHead.restype = ctypes.POINTER(ctypes.c_int64 * 4)
            self.test_head = 0

            #triple classification
            self.lib.getTestBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.getValidBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.getBestThreshold.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.test_triple_classification.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

            #set other parameters
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
        self.lib.importOntologyFiles()
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
                self.set_mini_batch()
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

    def set_mini_batch(self):
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
        current_lp_res = {
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

        if self.test_head != 0:
            current_lp_res['l_tot'] = 0.0
            current_lp_res['l_filter_tot'] = 0.0
            current_lp_res['l_tot_constrain'] = 0.0
            current_lp_res['l_filter_tot_constrain'] = 0.0
            current_lp_res['l1_tot'] = 0.0
            current_lp_res['l1_filter_tot'] = 0.0
            current_lp_res['l1_tot_constrain'] = 0.0
            current_lp_res['l1_filter_tot_constrain'] = 0.0
            current_lp_res['l3_tot'] = 0.0
            current_lp_res['l3_filter_tot'] = 0.0
            current_lp_res['l3_tot_constrain'] = 0.0
            current_lp_res['l3_filter_tot_constrain'] = 0.0
            current_lp_res['l_rank'] = 0.0
            current_lp_res['l_filter_rank'] = 0.0
            current_lp_res['l_rank_constrain'] = 0.0
            current_lp_res['l_filter_rank_constrain'] = 0.0
            current_lp_res['l_reci_rank'] = 0.0
            current_lp_res['l_filter_reci_rank'] = 0.0
            current_lp_res['l_reci_rank_constrain'] = 0.0
            current_lp_res['l_filter_reci_rank_constrain'] = 0.0

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
                for key in current_lp_res.keys():
                    current_lp_res[key] = float(f.readline())

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

            r_min = test_tail_res[4]
            r_filter_min = test_tail_res[5]
            r_constrain_min = test_tail_res[6]
            r_filter_constrain_min = test_tail_res[7]

            if (r_filter_s < 10): current_lp_res['r_filter_tot'] += 1
            if (r_s < 10): current_lp_res['r_tot'] += 1
            if (r_filter_s < 3): current_lp_res['r3_filter_tot'] += 1
            if (r_s < 3): current_lp_res['r3_tot'] += 1

            if (r_filter_s < 1):
                current_lp_res['r1_filter_tot'] += 1
            elif (r_filter_min == 1):
                current_lp_res['r_filter_gen_err'] += 1
            elif (r_filter_min == 2):
                current_lp_res['r_filter_spec_err'] += 1
            else:
                current_lp_res['r_filter_mis_err'] += 1

            if (r_s < 1):
                current_lp_res['r1_tot'] += 1
            elif (r_min == 1):
                current_lp_res['r_gen_err'] += 1
            elif (r_min == 2):
                current_lp_res['r_spec_err'] += 1
            else:
                current_lp_res['r_mis_err'] += 1



            current_lp_res['r_filter_rank'] += (1+r_filter_s)
            current_lp_res['r_rank'] += (1+r_s)
            current_lp_res['r_filter_reci_rank'] += np.divide(1.0, (1+r_filter_s))
            current_lp_res['r_reci_rank'] += np.divide(1.0, (1+r_s))

            #TYPE_C
            if (r_filter_s_constrain < 10): current_lp_res['r_filter_tot_constrain'] += 1
            if (r_s_constrain < 10): current_lp_res['r_tot_constrain'] += 1
            if (r_filter_s_constrain < 3): current_lp_res['r3_filter_tot_constrain'] += 1
            if (r_s_constrain < 3): current_lp_res['r3_tot_constrain'] += 1

            if (r_filter_s_constrain < 1):
                current_lp_res['r1_filter_tot_constrain'] += 1
            elif (r_filter_constrain_min == 1):
                current_lp_res['r_filter_gen_err_constrain'] += 1
            elif (r_filter_constrain_min == 2):
                current_lp_res['r_filter_spec_err_constrain'] += 1
            else:
                current_lp_res['r_filter_mis_err_constrain'] += 1

            if (r_s_constrain < 1):
                current_lp_res['r1_tot_constrain'] += 1
            elif (r_constrain_min == 1):
                current_lp_res['r_gen_err_constrain'] += 1
            elif (r_constrain_min == 2):
                current_lp_res['r_spec_err_constrain'] += 1
            else:
                current_lp_res['r_mis_err_constrain'] += 1

            current_lp_res['r_filter_rank_constrain'] += (1+r_filter_s_constrain)
            current_lp_res['r_rank_constrain'] += (1+r_s_constrain)
            current_lp_res['r_filter_reci_rank_constrain'] += np.divide(1.0, (1+r_filter_s_constrain))
            current_lp_res['r_reci_rank_constrain'] += np.divide(1.0, (1+r_s_constrain))

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

                if (l_filter_s < 10): current_lp_res['l_filter_tot'] += 1
                if (l_s < 10): current_lp_res['l_tot'] += 1
                if (l_filter_s < 3): current_lp_res['l3_filter_tot'] += 1
                if (l_s < 3): current_lp_res['l3_tot'] += 1
                if (l_filter_s < 1): current_lp_res['l1_filter_tot'] += 1
                if (l_s < 1): current_lp_res['l1_tot'] += 1

                current_lp_res['l_filter_rank'] += (l_filter_s+1)
                current_lp_res['l_rank'] += (1+l_s)
                current_lp_res['l_filter_reci_rank'] += np.divide(1.0, (l_filter_s+1))
                current_lp_res['l_reci_rank'] += np.divide(1.0, (l_s+1))

                #TYPE_C
                if (l_filter_s_constrain < 10): current_lp_res['l_filter_tot_constrain'] += 1
                if (l_s_constrain < 10): current_lp_res['l_tot_constrain'] += 1
                if (l_filter_s_constrain < 3): current_lp_res['l3_filter_tot_constrain'] += 1
                if (l_s_constrain < 3): current_lp_res['l3_tot_constrain'] += 1
                if (l_filter_s_constrain < 1): current_lp_res['l1_filter_tot_constrain'] += 1
                if (l_s_constrain < 1): current_lp_res['l1_tot_constrain'] += 1

                current_lp_res['l_filter_rank_constrain'] += (l_filter_s_constrain+1)
                current_lp_res['l_rank_constrain'] += (1+l_s_constrain)
                current_lp_res['l_filter_reci_rank_constrain'] += np.divide(1.0, (l_filter_s_constrain+1))
                current_lp_res['l_reci_rank_constrain'] += np.divide(1.0, (l_s_constrain+1))

            if index == 0: sys.stdout.write("\r# of test triples processed: {}".format(i * self.N_THREADS_LP))

            test_triples_done += 1

            if test_triples_done % 100 == 0:
                with open(self.test_log_path+"thread"+str(index), "w") as f:
                    f.write(str(i)+'\n')
                    for key in current_lp_res.keys():
                        f.write(str(current_lp_res[key])+'\n')

        self.lp_res[index] = current_lp_res



    def test(self):
        with self.graph.as_default():
            with self.sess.as_default():
                if self.importName != None:
                    self.restore_tensorflow()

                test_time_start = time.time()

                if self.test_link_prediction:
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

                    if self.test_head != 0:
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

                    for t in threads_array:
                        t.join()

                    for res in self.lp_res:
                        for key in res.keys():
                            d[key] += res[key]

                    for key in d.keys():
                        d[key] = np.divide(d[key], testTotal)


                    print("\n ========== LINK PREDICTION RESULTS ==========\nNo type constraint results:")
                    print("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format("metric", "MRR", "MR", "hit@10", "hit@3", "hit@1", "hit@1GenError", "hit@1SpecError", "hit@1MisError"))
                    if self.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format("l(raw):",d['l_reci_rank'], d['l_rank'], d['l_tot'], d['l3_tot'], d['l1_tot']))
                    print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format("r(raw):", d['r_reci_rank'], d['r_rank'], d['r_tot'], d['r3_tot'], d['r1_tot'], d['r_gen_err'], d['r_spec_err'], d['r_mis_err']))
                    if self.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}\n".format("mean(raw):", np.divide((d['l_reci_rank']+d['r_reci_rank']),2), np.divide((d['l_rank']+d['r_rank']),2), np.divide((d['l_tot']+d['r_tot']),2), np.divide((d['l3_tot']+d['r3_tot']),2), np.divide((d['l1_tot']+d['r1_tot']),2)))

                    if self.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format("l(filter):", d['l_filter_reci_rank'], d['l_filter_rank'], d['l_filter_tot'], d['l3_filter_tot'], d['l1_filter_tot']))
                    print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format('r(filter):', d['r_filter_reci_rank'], d['r_filter_rank'], d['r_filter_tot'], d['r3_filter_tot'], d['r1_filter_tot'], d['r_filter_gen_err'], d['r_filter_spec_err'], d['r_filter_mis_err']))
                    if self.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}\n".format('mean(filter):', np.divide((d['l_filter_reci_rank']+d['r_filter_reci_rank']),2), np.divide((d['l_filter_rank']+d['r_filter_rank']),2), np.divide((d['l_filter_tot']+d['r_filter_tot']),2), np.divide((d['l3_filter_tot']+d['r3_filter_tot']),2), np.divide((d['l1_filter_tot']+d['r1_filter_tot']),2)))

                    print("Type constraint results:")
                    print("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format("metric", "MRR", "MR", "hit@10", "hit@3", "hit@1", "hit@1GenError", "hit@1SpecError", "hit@1MisError"))
                    if self.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format('l(raw):', d['l_reci_rank_constrain'], d['l_rank_constrain'], d['l_tot_constrain'], d['l3_tot_constrain'], d['l1_tot_constrain']))
                    print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format('r(raw):', d['r_reci_rank_constrain'], d['r_rank_constrain'], d['r_tot_constrain'], d['r3_tot_constrain'], d['r1_tot_constrain'], d['r_gen_err_constrain'], d['r_spec_err_constrain'], d['r_mis_err_constrain']))
                    if self.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}\n".format('mean(raw):', np.divide((d['l_reci_rank_constrain']+d['r_reci_rank_constrain']),2), np.divide((d['l_rank_constrain']+d['r_rank_constrain']),2), np.divide((d['l_tot_constrain']+d['r_tot_constrain']),2), np.divide((d['l3_tot_constrain']+d['r3_tot_constrain']),2), np.divide((d['l1_tot_constrain']+d['r1_tot_constrain']),2)))

                    if self.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format('l(filter):', d['l_filter_reci_rank_constrain'], d['l_filter_rank_constrain'], d['l_filter_tot_constrain'], d['l3_filter_tot_constrain'], d['l1_filter_tot_constrain']))
                    print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}".format('r(filter):', d['r_filter_reci_rank_constrain'], d['r_filter_rank_constrain'], d['r_filter_tot_constrain'], d['r3_filter_tot_constrain'], d['r1_filter_tot_constrain'], d['r_filter_gen_err_constrain'], d['r_filter_spec_err_constrain'], d['r_filter_mis_err_constrain']))
                    if self.test_head != 0: print("{:<20}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}\n".format('mean(filter):', np.divide((d['l_filter_reci_rank_constrain']+d['r_filter_reci_rank_constrain']),2), np.divide((d['l_filter_rank_constrain']+d['r_filter_rank_constrain']),2), np.divide((d['l_filter_tot_constrain']+d['r_filter_tot_constrain']),2), np.divide((d['l3_filter_tot_constrain']+d['r3_filter_tot_constrain']),2), np.divide((d['l1_filter_tot_constrain']+d['r1_filter_tot_constrain']),2)))

                    #remove test checkpoint
                    print()
                    for index in range(0, self.N_THREADS_LP):
                        try:
                            os.remove(self.test_log_path+"thread"+str(index))
                        except:
                            print(" LOG:\tFile " + self.test_log_path+"thread"+str(index) + " not founded")
                    print()

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
        # self.init_link_prediction()
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
        # self.init_link_prediction()
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
        # self.init_link_prediction()
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
