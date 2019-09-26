from Config import Config
from TransE import TransE
from TransH import TransH
from TransR import TransR
from TransD import TransD
import tensorflow as tf
from tensorflowonspark import TFNode
import os
import numpy as np
from tensorflow.contrib.training.python.training.device_setter import GreedyLoadBalancingStrategy, byte_size_load_fn
import sys
import time


def get_conf_to_update_model(output_path):
    '''
    Set the Config class variables necessary to update model tensors
    '''
    con = Config(init_new_entities=True)

    ckpt = None
    with open(os.path.join(output_path, "checkpoint"), 'r') as f:
        first_line = f.readline()
        ckpt = first_line.split(':')[1].strip().replace('"', '')
    print("Checkpoint file is: " + ckpt)

    con.import_model(ckpt)

    return con, ckpt


def get_conf(argv=None):
    '''
    Set the Config class using the program args
    '''
    if argv == None: argv = sys.argv

    con = Config(cpp_lib_path=argv.cpp_lib_path)
    con.set_in_path(argv.input_path)
    con.set_export_files(argv.output_path)

    if argv.mode == 'train': con.set_valid_triple_classification(True)        #to perform early stop on validation accuracy
    else: con.set_test_link_prediction(True)

    con.set_train_times(argv.train_times)
    con.set_nbatches(argv.n_mini_batches)
    con.set_alpha(argv.alpha)
    con.set_margin(argv.margin)
    con.set_bern(argv.bern_flag)

    if argv.ent_dimension != 0 and argv.rel_dimension != 0:
        con.set_ent_dimension(argv.ent_dimension)
        con.set_rel_dimension(argv.rel_dimension)
    else:
        con.set_dimension(argv.embedding_dimension)

    con.set_ent_neg_rate(argv.ent_neg_rate)
    con.set_rel_neg_rate(argv.rel_neg_rate)
    con.set_opt_method(argv.optimizer)
    con.init()

    if argv.model.lower() == "transh":
        con.set_model(TransH)
    elif argv.model.lower() == "transr":
        con.set_model(TransR)
    elif argv.model.lower() == "transd":
        con.set_model(TransD)
    else:
        con.set_model(TransE)

    return con


def create_model(con):
    '''
    create the model using the Config parameters
    '''
    with tf.variable_scope("", reuse=None, initializer = tf.contrib.layers.xavier_initializer(uniform = True)):
        trainModel = con.model(config = con)

        with tf.name_scope("input"):
            trainModel.input_def()

        with tf.name_scope("embedding"):
            trainModel.embedding_def()

        with tf.name_scope("loss"):
            trainModel.loss_def()
            tf.summary.scalar("loss", trainModel.loss)  #summary 1

        with tf.name_scope("predict"):
            trainModel.predict_def()

        #allowed Optimization algorithms are Adam and SGD
        if con.opt_method == "Adam" or con.opt_method == "adam":
            optimizer = tf.train.AdamOptimizer(con.alpha)
        else:
            optimizer = tf.train.GradientDescentOptimizer(con.alpha)

        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(trainModel.loss, global_step=global_step)
        init_op = tf.initialize_all_variables()
        saver = tf.train.Saver(max_to_keep=sys.argv.early_stop_patience+5)

        return trainModel, global_step, train_op, init_op, saver, tf.summary.merge_all()


def create_model_evaluation(con):
    '''
    create the model using the Config parameters for evaluation mode
    creates only the model and gloabal step
    '''
    with tf.variable_scope("", reuse=None, initializer = tf.contrib.layers.xavier_initializer(uniform = True)):
        trainModel = con.model(config = con)

        with tf.name_scope("input"):
            trainModel.input_def()

        with tf.name_scope("embedding"):
            trainModel.embedding_def()

        with tf.name_scope("loss"):
            trainModel.loss_def()

        with tf.name_scope("predict"):
            trainModel.predict_def()

        global_step = tf.train.get_or_create_global_step()
        init_op = tf.initialize_all_variables()

        return trainModel, init_op, global_step


def get_last_step():
    '''
    :return: last global step; 0 if is the first batch
    '''
    last_global_step = 0
    try:
        if os.path.isfile(os.path.join(sys.argv.output_path, "checkpoint")):
            if sys.argv.debug:
                print("Checkpoint file founded")
                print("Reading last global step...")

            with open(os.path.join(sys.argv.output_path, "checkpoint"), "r") as f:
                line = f.readline().replace('"','').split(":")[1].split("/")
                last = int(line[len(line)-1].split("-")[1].strip())
                last_global_step = last

            if sys.argv.debug: print("Last global step: " + str(last_global_step))
        elif sys.argv.debug: print("Checkpoint file not founded")
    except Exception as e:
        print("Error occured during last global step reading:")
        print(e)
    finally:
        return last_global_step




def main_fun(argv, ctx):
    '''
    Continue training on already seen training set / Start training on new batch
    If the new batch contains new entities, model tensors which depends from entity size are updated accordingly
    :param argv:
    :param ctx:
    '''
    job_name = ctx.job_name
    task_index = ctx.task_index
    sys.argv = argv


    if sys.argv.debug: print("Starting cluster and server...")
    cluster, server = TFNode.start_cluster_server(ctx, num_gpus=argv.num_gpus, rdma=False)
    if sys.argv.debug: print("Cluster and server started")


    if job_name == "ps":
        #parameter server
        print("PS: joining...")
        server.join()
        if sys.argv.debug: print("PS: join finished")


    elif job_name == "worker":
        #worker
        print("WORKER: training...")
        last_global_step = get_last_step()
        con = get_conf()


        if sys.argv.debug: print("Creating model...")
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % task_index,
            cluster=cluster,
            ps_strategy=GreedyLoadBalancingStrategy(num_tasks=argv.num_ps, load_fn=byte_size_load_fn))):
            if sys.argv.mode == 'train':
                trainModel, global_step, train_op, init_op, saver, summary_op = create_model(con)
            else:
                trainModel, init_op, global_step = create_model_evaluation(con)


        if sys.argv.debug: print("Creating Scaffold, FileWriter, ConfigProto...")
        if sys.argv.mode == 'train':
            iterations = con.train_times * con.nbatches + last_global_step
            scaffold = tf.train.Scaffold(init_op=init_op, saver=saver, summary_op=summary_op)
            tf.summary.FileWriter("tensorboard_%d" % ctx.worker_num, graph=tf.get_default_graph())
        else:
            scaffold = tf.train.Scaffold(init_op=init_op)

        config_monitored = tf.ConfigProto()
        if argv.num_gpus > 0:
            if sys.argv.debug: print("Setting GPU options...")
            visible_device_list = ''
            try:
                visible_device_list = os.environ["CUDA_VISIBLE_DEVICES"]
            except KeyError:
                visible_device_list = '0'
            gpu_options = tf.GPUOptions(allow_growth = True, visible_device_list = visible_device_list)
            config_monitored = tf.ConfigProto(gpu_options=gpu_options)


        if sys.argv.debug: print("Starting MonitoredTrainingSession...")
        sess = None
        if sys.argv.mode == 'train':
            sess = tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(task_index == 0),
                                               scaffold=scaffold,
                                               config=config_monitored,
                                               checkpoint_dir=argv.output_path,
                                               save_summaries_steps=1,
                                               save_checkpoint_steps=1*con.nbatches,
                                               summary_dir="tensorboard_%d" % ctx.worker_num
                                               )
        else:
            sess = tf.train.MonitoredTrainingSession(master=server.target,
                                                     is_chief=(task_index == 0),
                                                     scaffold=scaffold,
                                                     config=config_monitored,
                                                     checkpoint_dir=argv.output_path,
                                                     save_checkpoint_steps=None,
                                                     save_summaries_secs=None,
                                                     save_summaries_steps=None,
                                                     save_checkpoint_secs=None
                                                     )

        if sys.argv.debug:
            print("Monitoring training sessions started")
            print("Task index is: {}".format(task_index))

        if sys.argv.mode == 'train':
            #init local worker vars
            best_acc = np.finfo('float32').min
            best_loss = np.finfo('float32').max
            wait_steps_acc = 0
            wait_steps_loss = 0
            best_model_global_step_loss = 0
            best_model_global_step_acc = 0
            patience = sys.argv.early_stop_patience
            stopping_step = sys.argv.early_stop_stopping_step * con.nbatches
            to_reach_step = sys.argv.early_stop_start_step * con.nbatches + last_global_step
            con.lib.getValidBatch(con.valid_pos_h_addr, con.valid_pos_t_addr, con.valid_pos_r_addr, con.valid_neg_h_addr, con.valid_neg_t_addr, con.valid_neg_r_addr)
            local = 0
            g = last_global_step


            while g < iterations:
                try:
                    #gives time to other workers to connect
                    if task_index == 0 and local == 0:
                        time.sleep(30)

                    local += 1
                    con.sampling()
                    feed_dict = {
                        trainModel.batch_h: con.batch_h,
                        trainModel.batch_t: con.batch_t,
                        trainModel.batch_r: con.batch_r,
                        trainModel.batch_y: con.batch_y
                    }

                    _, loss, g = sess.run([train_op, trainModel.loss, global_step], feed_dict)
                    print('Global step: {} Epoch: {} Batch: {} loss: {}'.format(g, int( (g-last_global_step) /con.nbatches), int( (g - last_global_step) % con.nbatches), loss))


                    ################## EARLY STOP ##################
                    if (task_index != 0) and (g >= to_reach_step):
                        to_reach_step += stopping_step
                        if os.path.exists(sys.argv.output_path+"/stop.txt"):
                            print('\nEarly stop happened in chief worker\n')
                            break


                    if (task_index == 0) and (g < iterations) and (g >= to_reach_step):
                        while (g >= to_reach_step):
                            to_reach_step += stopping_step


                        ################## ACCURACY ##################
                        feed_dict[trainModel.predict_h] = con.valid_pos_h
                        feed_dict[trainModel.predict_t] = con.valid_pos_t
                        feed_dict[trainModel.predict_r] = con.valid_pos_r
                        res_pos = sess.run(trainModel.predict, feed_dict)

                        feed_dict[trainModel.predict_h] = con.valid_neg_h
                        feed_dict[trainModel.predict_t] = con.valid_neg_t
                        feed_dict[trainModel.predict_r] = con.valid_neg_r
                        res_neg = sess.run(trainModel.predict, feed_dict)


                        con.lib.getBestThreshold(con.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])
                        con.lib.test_triple_classification(con.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0], con.acc_addr)


                        acc = con.acc[0]

                        if sys.argv.debug:
                            print("\n[ Early Stop Check (Accuracy) ]")
                            print("Best Accuracy = %.10f" %(best_acc))
                            print("Accuracy after run  =  %.10f" %(acc))

                        if acc > best_acc:
                            best_acc = acc
                            wait_steps_acc = 0
                            if sys.argv.debug: print("New best Accuracy founded. Wait steps reset.")
                            best_model_global_step_acc = g

                        elif wait_steps_acc < patience:
                            wait_steps_acc += 1
                            if sys.argv.debug: print("Wait steps Accuracy incremented: {}\n".format(wait_steps_acc))


                        if wait_steps_acc >= patience:
                            print('Accuracy early stop. Accuracy has not been improved enough in {} times'.format(patience))
                            with open(sys.argv.output_path+"/stop.txt", "w") as stop_file:
                                stop_file.write(str(best_model_global_step_acc)+"\n")
                            break

                        ################## LOSS ##################
                        if sys.argv.debug:
                            print("\n[ Early Stop Checking (Loss) ]")
                            print("Best loss = %.10f" %(best_loss))
                            print("Loss after run  =  %.10f" %(loss))

                        if loss < best_loss:
                            best_loss = loss
                            wait_steps_loss = 0
                            if sys.argv.debug: print("New best loss founded. Wait steps reset.")
                            best_model_global_step_loss = g

                        elif wait_steps_loss < patience:
                            wait_steps_loss += 1
                            if sys.argv.debug: print("Wait steps loss incremented: {}\n".format(wait_steps_loss))


                        if wait_steps_loss >= patience:
                            print('Loss early stop. Losses has not been improved enough in {} times'.format(patience))
                            with open(sys.argv.output_path+"/stop.txt", "w") as stop_file:
                                stop_file.write(str(best_model_global_step_loss)+"\n")
                            break

                except:
                    print("Exception occured; stopping training")
                    break

            #gives time to chief to stop
            if task_index != 0:
                time.sleep(30)

        else:
            #gives time to other workers to connect
            if task_index == 0:
                time.sleep(30)

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

            if sys.argv.test_head != 0:
                #init head variable
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

            #init test arrays
            test_h = np.zeros(con.lib.getEntityTotal(), dtype = np.int64)
            test_t = np.zeros(con.lib.getEntityTotal(), dtype = np.int64)
            test_r = np.zeros(con.lib.getEntityTotal(), dtype = np.int64)
            test_h_addr = test_h.__array_interface__['data'][0]
            test_t_addr = test_t.__array_interface__['data'][0]
            test_r_addr = test_r.__array_interface__['data'][0]

            lef = 0
            rig = 0
            testTotal = con.lib.getTestTotal()
            num_workers = sys.argv.cluster_size - sys.argv.num_ps
            triples_per_worker = int(testTotal / num_workers)

            for j in range(0, num_workers):
                if j+1 == num_workers: rig = testTotal
                else: rig += triples_per_worker

                if task_index == j: break
                else: lef = rig

            print("Test link prediction range from {} to {}".format(lef, rig-1))
            lp_path = os.path.join(argv.output_path, "lp_worker_"+str(task_index))

            #restore from last checkpoint (if founded)
            if os.path.exists(lp_path):
                with open(lp_path, 'r') as f:
                    lines_ckpt = f.readlines()
                    if lines_ckpt[0].strip() == 'done':
                        time.sleep(30)
                        print("Link prediction evaluation task already done")
                        return

                    last_i = int(lines_ckpt[0])
                    print("Restoring lp results from index {}".format(last_i))
                    lef = last_i + 1
                    for i_ckpt in range(1, len(lines_ckpt)):
                        key = lines_ckpt[i_ckpt].split(":")[0].strip()
                        value = float(lines_ckpt[i_ckpt].split(":")[1].strip())
                        d[key] = value


            test_triples_done = 0
            for i in range(lef, rig):
                #tail link prediction on i-th test triple
                con.lib.getTailBatch(i, test_h_addr, test_t_addr, test_r_addr)

                feed_dict = {}
                feed_dict[trainModel.predict_h] = test_h
                feed_dict[trainModel.predict_t] = test_t
                feed_dict[trainModel.predict_r] = test_r
                res = sess.run(trainModel.predict, feed_dict)

                test_tail_res = [j for j in con.lib.testTail(i, res.__array_interface__['data'][0]).contents]

                r_s = test_tail_res[0]
                r_filter_s = test_tail_res[1]
                r_s_constrain = test_tail_res[2]
                r_filter_s_constrain = test_tail_res[3]
                r_min = test_tail_res[4]
                r_filter_min = test_tail_res[5]
                r_constrain_min = test_tail_res[6]
                r_filter_constrain_min = test_tail_res[7]

                #hits
                if (r_filter_s < 10): d['r_filter_tot'] += 1
                if (r_s < 10): d['r_tot'] += 1
                if (r_filter_s < 3): d['r3_filter_tot'] += 1
                if (r_s < 3): d['r3_tot'] += 1

                if (r_filter_s_constrain < 10): d['r_filter_tot_constrain'] += 1
                if (r_s_constrain < 10): d['r_tot_constrain'] += 1
                if (r_filter_s_constrain < 3): d['r3_filter_tot_constrain'] += 1
                if (r_s_constrain < 3): d['r3_tot_constrain'] += 1

                #ontology
                if (r_filter_s < 1): d['r1_filter_tot'] += 1
                elif (r_filter_min == 1): d['r_filter_gen_err'] += 1
                elif (r_filter_min == 2): d['r_filter_spec_err'] += 1
                else: d['r_filter_mis_err'] += 1

                if (r_s < 1): d['r1_tot'] += 1
                elif (r_min == 1): d['r_gen_err'] += 1
                elif (r_min == 2): d['r_spec_err'] += 1
                else: d['r_mis_err'] += 1

                if (r_filter_s_constrain < 1): d['r1_filter_tot_constrain'] += 1
                elif (r_filter_constrain_min == 1): d['r_filter_gen_err_constrain'] += 1
                elif (r_filter_constrain_min == 2): d['r_filter_spec_err_constrain'] += 1
                else: d['r_filter_mis_err_constrain'] += 1

                if (r_s_constrain < 1): d['r1_tot_constrain'] += 1
                elif (r_constrain_min == 1): d['r_gen_err_constrain'] += 1
                elif (r_constrain_min == 2): d['r_spec_err_constrain'] += 1
                else: d['r_mis_err_constrain'] += 1

                #MR
                d['r_filter_rank'] += (1+r_filter_s)
                d['r_rank'] += (1+r_s)
                d['r_filter_reci_rank'] += np.divide(1.0, (1+r_filter_s))
                d['r_reci_rank'] += np.divide(1.0, (1+r_s))

                d['r_filter_rank_constrain'] += (1+r_filter_s_constrain)
                d['r_rank_constrain'] += (1+r_s_constrain)
                d['r_filter_reci_rank_constrain'] += np.divide(1.0, (1+r_filter_s_constrain))
                d['r_reci_rank_constrain'] += np.divide(1.0, (1+r_s_constrain))


                if sys.argv.test_head != 0:
                    #head link prediction on i-th test triple
                    con.lib.getHeadBatch(i, test_h_addr, test_t_addr, test_r_addr)
                    feed_dict = {}
                    feed_dict[trainModel.predict_h] = test_h
                    feed_dict[trainModel.predict_t] = test_t
                    feed_dict[trainModel.predict_r] = test_r
                    res = sess.run(trainModel.predict, feed_dict)
                    test_head_res = [j for j in con.lib.testHead(i, res.__array_interface__['data'][0]).contents]

                    l_s = test_head_res[0]
                    l_filter_s = test_head_res[1]
                    l_s_constrain = test_head_res[2]
                    l_filter_s_constrain = test_head_res[3]
                    l_min = test_head_res[4]
                    l_filter_min = test_head_res[5]
                    l_constrain_min = test_head_res[6]
                    l_filter_constrain_min = test_head_res[7]

                    #hits
                    if (l_filter_s < 10): d['l_filter_tot'] += 1
                    if (l_s < 10): d['l_tot'] += 1
                    if (l_filter_s < 3): d['l3_filter_tot'] += 1
                    if (l_s < 3): d['l3_tot'] += 1

                    if (l_filter_s_constrain < 10): d['l_filter_tot_constrain'] += 1
                    if (l_s_constrain < 10): d['l_tot_constrain'] += 1
                    if (l_filter_s_constrain < 3): d['l3_filter_tot_constrain'] += 1
                    if (l_s_constrain < 3): d['l3_tot_constrain'] += 1

                    #ontology
                    if (l_filter_s < 1): d['l1_filter_tot'] += 1
                    elif (l_filter_min == 1): d['l_filter_gen_err'] += 1
                    elif (l_filter_min == 2): d['l_filter_spec_err'] += 1
                    else: d['l_filter_mis_err'] += 1

                    if (l_s < 1): d['l1_tot'] += 1
                    elif (l_min == 1): d['l_gen_err'] += 1
                    elif (l_min == 2): d['l_spec_err'] += 1
                    else: d['l_mis_err'] += 1

                    if (l_filter_s_constrain < 1): d['l1_filter_tot_constrain'] += 1
                    elif (l_filter_constrain_min == 1): d['l_filter_gen_err_constrain'] += 1
                    elif (l_filter_constrain_min == 2): d['l_filter_spec_err_constrain'] += 1
                    else: d['l_filter_mis_err_constrain'] += 1

                    if (l_s_constrain < 1): d['l1_tot_constrain'] += 1
                    elif (l_constrain_min == 1): d['l_gen_err_constrain'] += 1
                    elif (l_constrain_min == 2): d['l_spec_err_constrain'] += 1
                    else: d['l_mis_err_constrain'] += 1

                    #MR
                    d['l_filter_rank'] += (l_filter_s+1)
                    d['l_rank'] += (1+l_s)
                    d['l_filter_reci_rank'] += np.divide(1.0, (l_filter_s+1))
                    d['l_reci_rank'] += np.divide(1.0, (l_s+1))

                    d['l_filter_rank_constrain'] += (l_filter_s_constrain+1)
                    d['l_rank_constrain'] += (1+l_s_constrain)
                    d['l_filter_reci_rank_constrain'] += np.divide(1.0, (l_filter_s_constrain+1))
                    d['l_reci_rank_constrain'] += np.divide(1.0, (l_s_constrain+1))


                if sys.argv.debug: sys.stdout.write("\r# of test triples processed: {}/{}".format(i, rig))
                test_triples_done += 1
                #save checkpoint
                if test_triples_done % 100 == 0:
                    with open(lp_path, "w") as f:
                        f.write(str(i)+'\n')
                        for key in d.keys():
                            f.write(str(key) + ":" + str(d[key])+'\n')


            #write final results
            with open(lp_path, "w") as f:
                f.write('done\n')
                for key in d.keys():
                    f.write(str(key) + ":" + str(d[key])+'\n')


            #gives time to chief to stop
            if task_index != 0:
                time.sleep(30)






