from Config import Config
from TransE import TransE
from TransH import TransH
from TransR import TransR
from TransD import TransD
import tensorflow as tf
from tensorflowonspark import TFNode
import os
import numpy as np
from tensorflow.python.training import session_run_hook
from tensorflow.contrib.training.python.training.device_setter import GreedyLoadBalancingStrategy, byte_size_load_fn
import sys


#if set to Ture prints additional debug information
DEBUG = True


def get_conf_to_update_model(output_path):
    con = Config(init_new_entities=True)

    ckpt = None
    with open(output_path + "/checkpoint", 'r') as f:
        first_line = f.readline()
        ckpt = first_line.split(':')[1].strip().replace('"', '')
    if DEBUG: print("Checkpoint file is: " + ckpt)

    con.import_model(ckpt)

    return con, ckpt


def get_conf(argv=None):
    if argv == None: argv = sys.argv

    con = Config()
    con.set_in_path(argv.input_path)
    con.set_export_files(argv.output_path)
    con.set_valid_triple_classification(True)        #to perform early stop on validation accuracy
    # con.set_work_threads(argv.working_threads)
    con.set_train_times(argv.train_times)
    con.set_nbatches(argv.n_batches)
    con.set_alpha(argv.alpha)
    con.set_margin(argv.margin)

    if argv.bern_flag:
        con.set_bern(1)

    if argv.ent_dimension != 0 and argv.rel_dimension != 0:
        con.set_ent_dimension(argv.ent_dimension)
        con.set_rel_dimension(argv.rel_dimension)
    else:
        con.set_dimension(argv.embedding_dimension)

    con.set_ent_neg_rate(argv.ent_neg_rate)
    con.set_rel_neg_rate(argv.rel_neg_rate)
    con.set_opt_method(argv.optimizer)
    con.init()

    if argv.model.lower() == "transe":
        con.set_model(TransE)
    elif argv.model.lower() == "transh":
        con.set_model(TransH)
    elif argv.model.lower() == "transr":
        con.set_model(TransR)
    else:
        con.set_model(TransD)

    return con


def create_model(con):
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

        if con.opt_method == "Adam" or con.opt_method == "adam":
            optimizer = tf.train.AdamOptimizer(con.alpha)
        else:
            optimizer = tf.train.GradientDescentOptimizer(con.alpha)

        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(trainModel.loss, global_step=global_step)
        init_op = tf.initialize_all_variables()
        saver = tf.train.Saver(max_to_keep=sys.argv.early_stop_patience+5)

        return trainModel, global_step, train_op, init_op, saver, tf.summary.merge_all()


def get_last_step():
    last_global_step = 0
    try:
        if os.path.isfile(sys.argv.output_path+"/checkpoint"):
            if DEBUG:
                print("Checkpoint file founded")
                print("Reading last global step...")

            with open(sys.argv.output_path+"/checkpoint", "r") as f:
                line = f.readline().replace('"','').split(":")[1].split("/")
                last = int(line[len(line)-1].split("-")[1].strip())
                last_global_step = last

            if DEBUG: print("Last global step: " + str(last_global_step))
        else:
            if DEBUG: print("Checkpoint file not founded")
    except Exception as e:
        print("Error occured during last global step reading:")
        print(e)
    finally:
        return last_global_step


def main_fun(argv, ctx):
    job_name = ctx.job_name
    task_index = ctx.task_index
    sys.argv = argv


    if DEBUG: print("Starting cluster and server...")
    cluster, server = TFNode.start_cluster_server(ctx, num_gpus=argv.num_gpus, rdma=False)
    if DEBUG: print("Cluster and server started")


    if job_name == "ps":
        print("PS: joining...")
        server.join()
        if DEBUG: print("PS: join finished")

    elif job_name == "worker":
        print("WORKER: training...")

        #Online learning
        last_global_step = get_last_step()


        #set config
        if DEBUG: print("Creating conf...")
        con = get_conf()


        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % task_index,
            cluster=cluster,
            ps_strategy=GreedyLoadBalancingStrategy(num_tasks=argv.num_ps, load_fn=byte_size_load_fn))):

            if DEBUG: print("Creating model...")
            trainModel, global_step, train_op, init_op, saver, summary_op = create_model(con)



        if DEBUG: print("Creating Hooks, Scaffold, FileWriter...")

        iterations = con.train_times * con.nbatches + last_global_step
        hooks=[tf.train.StopAtStepHook(last_step=iterations)]
        scaffold = tf.train.Scaffold(init_op=init_op, saver=saver, summary_op=summary_op)
        tf.summary.FileWriter("tensorboard_%d" % ctx.worker_num, graph=tf.get_default_graph())


        if DEBUG: print("Starting MonitoredTrainingSession...")
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(task_index == 0),
                                               scaffold=scaffold,
                                               checkpoint_dir=argv.output_path,
                                               save_summaries_steps=1,
                                               save_checkpoint_steps=1*con.nbatches,
                                               hooks=hooks,
                                               summary_dir="tensorboard_%d" % ctx.worker_num
                                               ) as sess:
            if DEBUG:
                print("Monitoring training sessions started")
                print("Task index is: {}".format(task_index))

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

            while not sess.should_stop():
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


                if (task_index == 0) and (not sess.should_stop()) and (g >= to_reach_step):
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

                    if DEBUG:
                        print("\n[ Early Stop Check (Accuracy) ]")
                        print("Best Accuracy = %.10f" %(best_acc))
                        print("Accuracy after run  =  %.10f" %(acc))

                    if acc > best_acc:
                        best_acc = acc
                        wait_steps_acc = 0
                        if DEBUG: print("New best Accuracy founded. Wait steps reset.")
                        best_model_global_step_acc = g

                    elif wait_steps_acc < patience:
                        wait_steps_acc += 1
                        if DEBUG: print("Wait steps Accuracy incremented: {}\n".format(wait_steps_acc))


                    if wait_steps_acc >= patience:
                        print('Accuracy early stop. Accuracy has not been improved enough in {} times'.format(patience))
                        with open(sys.argv.output_path+"/stop.txt", "w") as stop_file:
                            stop_file.write(str(best_model_global_step_acc)+"\n")
                        break


                    ################## LOSS ##################
                    if DEBUG:
                        print("\n[ Early Stop Checking (Loss) ]")
                        print("Best loss = %.10f" %(best_loss))
                        print("Loss after run  =  %.10f" %(loss))

                    if loss < best_loss:
                        best_loss = loss
                        wait_steps_loss = 0
                        if DEBUG: print("New best loss founded. Wait steps reset.")
                        best_model_global_step_loss = g

                    elif wait_steps_loss < patience:
                        wait_steps_loss += 1
                        if DEBUG: print("Wait steps loss incremented: {}\n".format(wait_steps_loss))


                    if wait_steps_loss >= patience:
                        print('Loss early stop. Losses has not been improved enough in {} times'.format(patience))
                        with open(sys.argv.output_path+"/stop.txt", "w") as stop_file:
                            stop_file.write(str(best_model_global_step_loss)+"\n")
                        break

        if DEBUG: print("Monitoring training sessions should stop now")









