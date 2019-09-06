#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
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

    def __init__(self, cpp_lib_path=None, init_new_entities=False):
        '''
        Init Config Class
        :param cpp_lib_path: absolute path to .so file
        :param init_new_entities: if true training and test variables are not initialized
        '''

        self.init_new_entities = init_new_entities
        if init_new_entities == False:
            #C library
            if cpp_lib_path == None:
                cpp_lib_path = './release/Base.so'
            base_file = os.path.abspath(cpp_lib_path)
            self.lib = ctypes.cdll.LoadLibrary(base_file)
            self.lib.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

            #link prediction
            self.lib.getTailBatch.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.testTail.argtypes = [ctypes.c_int64, ctypes.c_void_p]
            self.lib.testTail.restype = ctypes.POINTER(ctypes.c_int64 * 8)
            self.lib.getHeadBatch.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.testHead.argtypes = [ctypes.c_int64, ctypes.c_void_p]
            self.lib.testHead.restype = ctypes.POINTER(ctypes.c_int64 * 8)

            #triple classification
            self.lib.getTestBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.getValidBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.getBestThreshold.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.test_triple_classification.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

            #ROC
            self.lib.get_n_interval.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.get_n_interval.restype = ctypes.c_int64
            self.lib.get_TPFP.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.get_TPFP.restype = ctypes.POINTER( ctypes.c_int64 * 2 )

            #set other parameters
            self.in_path = None
            self.out_path = None
            self.bern = 0
            self.hidden_size = 64
            self.ent_size = self.hidden_size
            self.rel_size = self.hidden_size
            self.train_times = 0
            self.margin = 1.0
            self.nbatches = 0
            self.negative_ent = 1
            self.negative_rel = 0
            self.workThreads = 8
            self.alpha = 0.001
            self.exportName = None
            self.importName = None
            self.opt_method = "SGD"
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


    def init_triple_classification(self):
        r'''
        import essential files and set essential interfaces for triple classification
        '''
        self.lib.importTestFiles()
        self.lib.importTypeFiles()

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

        self.acc = np.zeros(1, dtype = np.float32)
        self.acc_addr = self.acc.__array_interface__['data'][0]


    def init_valid_triple_classification(self):
        r'''
        import essential files and set essential interfaces for triple classification
        (on validation set, used during training)
        '''
        self.lib.importTestFiles()
        self.lib.importTypeFiles()

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

        self.acc = np.zeros(1, dtype = np.float32)
        self.acc_addr = self.acc.__array_interface__['data'][0]



    def init(self):
        '''
        prepare for train and test
        '''
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
        '''
        Set mini batch used during training
        This function checks for specified mini batch parameter
        If it has not been specified the mini batch is automatically set
        '''
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


    def get_ent_total(self):
        '''
        :return: the number of entites
        '''
        return self.entTotal

    def get_rel_total(self):
        '''
        :return: the number of relations
        '''
        return self.relTotal

    def set_opt_method(self, method):
        '''
        Set the optimization method
        :param method: a string representing the optimization method
        the current opt method supported are SGD and Adam
        '''
        self.opt_method = method

    def set_test_link_prediction(self, flag):
        '''
        If True link prediction evaluation will be performed when test method is called
        '''
        self.test_link_prediction = flag

    def set_test_triple_classification(self, flag):
        '''
        If true triple classification evaluation will be performed when test method is called
        '''
        self.test_triple_classification = flag

    def set_valid_triple_classification(self, flag):
        '''
        If true, triple classification evaluation will be performed on validation set during training
        (Early stop)
        '''
        self.valid_triple_classification = flag

    def set_alpha(self, alpha):
        '''
        Set learning rate
        '''
        self.alpha = alpha

    def set_in_path(self, path):
        '''
        Set path where training files are located
        '''
        self.in_path = path

    def set_out_files(self, path):
        '''
        Set path where the model will be saved
        '''
        self.out_path = path

    def set_bern(self, bern):
        '''
        Set whether to use bern method during sampling
        :param bern: 1 for True, 0 for False
        '''
        self.bern = bern


    def set_dimension(self, dim):
        '''
        Set embedding dimension for both the entities and relations
        '''
        self.hidden_size = dim
        self.ent_size = dim
        self.rel_size = dim

    def set_ent_dimension(self, dim):
        '''
        Set embedding dimension for entities
        '''
        self.ent_size = dim

    def set_rel_dimension(self, dim):
        '''
        Set embedding dimension for relations
        '''
        self.rel_size = dim

    def set_train_times(self, times):
        '''
        Set number of epochs
        '''
        self.train_times = times

    def set_nbatches(self, nbatches):
        '''
        Set number of batches
        '''
        self.nbatches = nbatches

    def set_margin(self, margin):
        '''
        Set margin hyperparameter
        '''
        self.margin = margin

    def set_ent_neg_rate(self, rate):
        '''
        Set number of corrupted triples generated during training for each triple
        (Corrupt head/tail)
        '''
        self.negative_ent = rate

    def set_rel_neg_rate(self, rate):
        '''
        Set number of corrupted triples generated during training for each triple
        (Corrupt rel)
        '''
        self.negative_rel = rate

    def set_import_files(self, path):
        '''
        Set path where is located the model to import
        '''
        self.importName = path

    def set_export_files(self, path):
        '''
        Set path where output model will be located
        '''
        self.exportName = path


    def sampling(self):
        '''
        Call C function for batch sampling during training
        '''
        self.lib.sampling(self.batch_h_addr, self.batch_t_addr, self.batch_r_addr, self.batch_y_addr, self.batch_size, self.negative_ent, self.negative_rel)


    def save_tensorflow(self):
        '''
        Save tensorflow model
        '''
        with self.graph.as_default():
            with self.sess.as_default():
                self.saver.save(self.sess, self.exportName)


    def save_tensorflow_weights(self, export_name=None, write_meta_graph=False):
        '''
        Save only tensorflow model weights
        :return:
        '''
        if export_name == None:
            export_name = self.exportName
        with self.graph.as_default():
            with self.sess.as_default():
                self.saver.save(self.sess, export_name, write_meta_graph=write_meta_graph, write_state=False)


    def restore_tensorflow(self):
        '''
        Restore tensorflow model defined in importName var
        '''
        with self.graph.as_default():
            with self.sess.as_default():
                self.saver.restore(self.sess, self.importName)

    def get_parameter_lists(self):
        '''
        :return: trainModel variables
        '''
        return self.trainModel.parameter_lists

    def get_parameters_by_name(self, var_name):
        '''
        :param var_name:
        :return: trainModel variable
        '''
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
        '''
        Set training model
        '''
        self.model = model


    def import_model(self, ckpt):
        '''
        Import variables from a specific trained model version
        :param ckpt: path/to/checkpoint/
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                print("Importing metagraph...")
                self.saver = tf.train.import_meta_graph(ckpt+".meta", clear_devices=True)
                print("Importing variables...")
                self.saver.restore(self.sess, ckpt)


    def set_model_and_session(self, model):
        '''
        Init the training algorithm variables and the tensorflow session
        :parm model: (TransE / TransH / TransR / TransD)
        '''
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
        '''
        Perform a single training step
        '''
        feed_dict = {
            self.trainModel.batch_h: batch_h,
            self.trainModel.batch_t: batch_t,
            self.trainModel.batch_r: batch_r,
            self.trainModel.batch_y: batch_y
        }
        _, loss = self.sess.run([self.train_op, self.trainModel.loss], feed_dict)
        return loss


    def test_step(self, test_h, test_t, test_r):
        '''
        Perform a single test step
        '''
        feed_dict = {
            self.trainModel.predict_h: test_h,
            self.trainModel.predict_t: test_t,
            self.trainModel.predict_r: test_r,
        }
        predict = self.sess.run(self.trainModel.predict, feed_dict)
        return predict


    def test(self):
        '''
        Perform triple classifcation and link prediction evaluation
        '''
        with self.graph.as_default():
            with self.sess.as_default():
                if self.importName != None:
                    self.restore_tensorflow()

                test_time_start = time.time()

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


    def plot_roc(self, rel_index, fig_name=None):
        if self.importName != None:
            self.restore_tensorflow()
        self.init_triple_classification()

        self.lib.getValidBatch(self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr, self.valid_neg_h_addr, self.valid_neg_t_addr, self.valid_neg_r_addr)
        res_pos_valid = self.test_step(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
        res_neg_valid = self.test_step(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)

        self.lib.getTestBatch(self.test_pos_h_addr, self.test_pos_t_addr, self.test_pos_r_addr, self.test_neg_h_addr, self.test_neg_t_addr, self.test_neg_r_addr)
        res_pos_test = self.test_step(self.test_pos_h, self.test_pos_t, self.test_pos_r)
        res_neg_test = self.test_step(self.test_neg_h, self.test_neg_t, self.test_neg_r)

        n_intervals = self.lib.get_n_interval(rel_index, res_pos_valid.__array_interface__['data'][0], res_neg_valid.__array_interface__['data'][0])
        self.lib.get_TPFP.restype = ctypes.POINTER( ctypes.c_int64 * ((n_intervals+1)*2) )
        res = [j for j in self.lib.get_TPFP(rel_index, res_pos_valid.__array_interface__['data'][0], res_neg_valid.__array_interface__['data'][0], res_pos_test.__array_interface__['data'][0], res_neg_test.__array_interface__['data'][0]).contents]

        TPR = []
        FPR = []

        if res[0] != 0 or res[0+n_intervals+1] != 0:
            TPR.append(0)
            FPR.append(0)


        for i in range(0, n_intervals+1):
            TPR.append(res[i])
            FPR.append(res[i+n_intervals+1])

        if TPR[len(TPR)-1] != len(res_pos_test.flatten()) or FPR[len(FPR)-1] != len(res_neg_test.flatten()):
            TPR.append(len(res_pos_test.flatten()))
            FPR.append(len(res_neg_test.flatten()))


        for i in range(len(TPR)): TPR[i] /= TPR[-1]
        for i in range(len(FPR)): FPR[i] /= FPR[-1]

        auc = np.trapz(TPR, FPR)

        plt.figure()
        lw=2
        plt.plot(FPR, TPR, color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        if fig_name == None or fig_name == '':
            plt.show()
        else:
            plt.savefig(fig_name)


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
