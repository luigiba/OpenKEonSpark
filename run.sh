#!/bin/bash
# On ps0.example.com:
python3 example_train_transe_d.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=0
# On ps1.example.com:
python3 example_train_transe_d.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=1
# On worker0.example.com:
python3 example_train_transe_d.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=0
# On worker1.example.com:
python3 example_train_transe_d.py \
     --ps_hosts=ps0.example.com:2223,ps1.example.com:2223 \
     --worker_hosts=worker0.example.com:2223,worker1.example.com:2223 \
     --job_name=worker --task_index=1


python3 example_train_transe_d.py \
     --ps_hosts=ps0.example.com:2225 \
     --worker_hosts=worker0.example.com:2225 \
     --job_name=ps --task_index=0

python3 example_train_transe_d.py \
     --ps_hosts=ps0.example.com:2224,ps1.example.com:2224 \
     --worker_hosts=worker0.example.com:2224,worker1.example.com:2224 \
     --job_name=worker --task_index=0

lsof -i -P -n | grep LISTEN | grep python



python3 example_train_transe_d.py --job_name "ps" --task_index 0 &
python3 example_train_transe_d.py --job_name "worker" --task_index 0 &


python3 example_train_transe_d.py --ps_hosts=localhost:2222,localhost:2223 --worker_hosts=localhost:2224,localhost:2225 --job_name=ps --task_index=0 &
python3 example_train_transe_d.py --ps_hosts=localhost:2222,localhost:2223 --worker_hosts=localhost:2224,localhost:2225 --job_name=ps --task_index=1 &
python3 example_train_transe_d.py --ps_hosts=localhost:2222,localhost:2223 --worker_hosts=localhost:2224,localhost:2225 --job_name=worker --task_index=0 &
python3 example_train_transe_d.py --ps_hosts=localhost:2222,localhost:2223 --worker_hosts=localhost:2224,localhost:2225 --job_name=worker --task_index=1 &


python3 example_train_transe_d.py --ps_hosts=localhost:2222 --worker_hosts=localhost:2223 --job_name=ps --task_index=0 &
python3 example_train_transe_d.py --ps_hosts=localhost:2222 --worker_hosts=localhost:2223 --job_name=ps --task_index=1 &
python3 example_train_transe_d.py --ps_hosts=localhost:2222 --worker_hosts=localhost:2223 --job_name=worker --task_index=0 &
python3 example_train_transe_d.py --ps_hosts=localhost:2222 --worker_hosts=localhost:2223 --job_name=worker --task_index=1 &



python3 example_train_transe_d.py --ps_hosts=localhost:2222 --worker_hosts=localhost:2223 --job_name=ps --task_index=0 &
python3 example_train_transe_d.py --ps_hosts=localhost:2222 --worker_hosts=localhost:2223 --job_name=worker --task_index=0 &



${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--py-files /home/luigi/IdeaProjects/OpenKE_Spark/example_train_transe_spark.py \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--conf spark.executorEnv.JAVA_HOME="${JAVA_HOME}" \
/home/luigi/IdeaProjects/OpenKE_Spark/example_train_transe_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES}




${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
/home/luigi/IdeaProjects/OpenKE_Spark/example_train_transe_spark.py \
--cluster_size ${SPARK_EXECUTOR_INSTANCES}



##################################################################################################################


${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-slave.sh -c ${CORES_PER_WORKER} -m ${MEMORY_PER_WORKER} ${MASTER}


${SPARK_HOME}/sbin/stop-slave.sh; ${SPARK_HOME}/sbin/stop-master.sh

#CPU
${SPARK_HOME}/bin/spark-submit --master ${MASTER} \
--py-files /home/luigi/IdeaProjects/OpenKE_new_Spark/distribute_training.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/Config.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/Model.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/TransE.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/Model.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/TransH.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/Model.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/TransR.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/Model.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/TransD.py \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.cores.max=${TOTAL_CORES} --conf spark.task.cpus=${CORES_PER_WORKER} --executor-memory ${MEMORY_PER_WORKER} \
--num-executors ${SPARK_WORKER_INSTANCES} \
/home/luigi/IdeaProjects/OpenKE_new_Spark/main_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} --num_ps 1 \
--input_path /home/luigi/IdeaProjects/OpenKE_new_Spark/benchmarks/DBpedia/5/0/ \
--output_path /home/luigi/IdeaProjects/OpenKE_new_Spark/res_spark \
--alpha 0.0001 --optimizer SGD --train_times 50 --ent_neg_rate 1 --embedding_dimension 64 --margin 1.0 --model TransE



${SPARK_HOME}/bin/spark-submit --master ${MASTER} \
--py-files /home/luigi/IdeaProjects/OpenKE_new_Spark/distribute_training.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/Config.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/Model.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/TransE.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/Model.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/TransH.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/Model.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/TransR.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/Model.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/TransD.py \
--conf spark.cores.max=${TOTAL_CORES} --conf spark.task.cpus=${CORES_PER_WORKER} --executor-memory ${MEMORY_PER_WORKER} \
--num-executors ${SPARK_WORKER_INSTANCES} \
/home/luigi/IdeaProjects/OpenKE_new_Spark/main_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} --num_ps 1 \
--input_path /home/luigi/IdeaProjects/OpenKE_new_Spark/benchmarks/superuser_1/5/ \
--output_path /home/luigi/IdeaProjects/OpenKE_new_Spark/res_spark \
--early_stop_patience 10 --early_stop_tolerance 0 \
--working_threads 8 --alpha 0.0001 --optimizer Adam --train_times 50 --ent_neg_rate 5 --embedding_dimension 64 --margin 1.0 --model TransE


#GPU
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--py-files /home/luigi/IdeaProjects/OpenKE_new_Spark/distribute_training.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/Config.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/Model.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/TransE.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/Model.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/TransH.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/Model.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/TransR.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/Model.py,/home/luigi/IdeaProjects/OpenKE_new_Spark/TransD.py \
--conf spark.executorEnv.LD_LIBRARY_PATH=${LIB_CUDA} \
--driver-library-path=${LIB_CUDA} \
--num-executors ${SPARK_WORKER_INSTANCES} \
/home/luigi/IdeaProjects/OpenKE_new_Spark/main_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--num_ps 1 \
--input_path /home/luigi/IdeaProjects/OpenKE_new_Spark/benchmarks/superuser_1/5/ \
--output_path /home/luigi/IdeaProjects/OpenKE_new_Spark/res_spark \
--working_threads 8 \
--alpha 0.0001 \
--early_stop_patience 10 --early_stop_tolerance 0 --optimizer Adam \
--train_times 200 --ent_neg_rate 5 --embedding_dimension 64 --margin 1.0 --model TransE




--conf spark.driver.memory ${MEMORY_PER_WORKER} \
--conf spark.executor.memory ${TOTAL_MEMORY} \




${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--py-files /home/luigi/IdeaProjects/OpenKE_Spark/distribute_training.py,/home/luigi/IdeaProjects/OpenKE_Spark/Config.py,/home/luigi/IdeaProjects/OpenKE_Spark/Model.py,/home/luigi/IdeaProjects/OpenKE_Spark/TransE.py,/home/luigi/IdeaProjects/OpenKE_Spark/Model.py,/home/luigi/IdeaProjects/OpenKE_Spark/TransH.py,/home/luigi/IdeaProjects/OpenKE_Spark/Model.py,/home/luigi/IdeaProjects/OpenKE_Spark/TransR.py,/home/luigi/IdeaProjects/OpenKE_Spark/Model.py,/home/luigi/IdeaProjects/OpenKE_Spark/TransD.py \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--num-executors ${SPARK_WORKER_INSTANCES} \
/home/luigi/IdeaProjects/OpenKE_Spark/main_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--num_ps 1 \
--input_path /home/luigi/IdeaProjects/OpenKE_Spark/benchmarks/superuser/ \
--output_path /home/luigi/IdeaProjects/OpenKE_Spark/res_spark \
--working_threads 8 \
--train_times 100 --n_batches 100 \
--alpha 0.0001 --margin 1.0 --embedding_dimension 100 \
--ent_neg_rate 10 --rel_neg_rate 0 --optimizer Adam --model TransE





${SPARK_HOME}/sbin/stop-slave.sh; ${SPARK_HOME}/sbin/stop-master.sh


##################################################################################################################



${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--num-executors 2 \
--queue default \
--py-files /home/luigi/IdeaProjects/OpenKE_Spark/example_train_transe_spark.py,/home/luigi/IdeaProjects/OpenKE_Spark/Config.py,/home/luigi/IdeaProjects/OpenKE_Spark/Model.py,/home/luigi/IdeaProjects/OpenKE_Spark/TransE.py \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
/home/luigi/IdeaProjects/OpenKE_Spark/main_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--num_ps 1









${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--py-files /home/luigi/IdeaProjects/OpenKE_Spark/example_train_transe_spark.py,/home/luigi/IdeaProjects/OpenKE_Spark/Config.py,/home/luigi/IdeaProjects/OpenKE_Spark/Model.py,/home/luigi/IdeaProjects/OpenKE_Spark/TransE.py \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--conf spark.task.maxFailures=1 \
--conf spark.stage.maxConsecutiveAttempts=1 \
--executor-cores 1 \
--num-executors ${SPARK_WORKER_INSTANCES} \
/home/luigi/IdeaProjects/OpenKE_Spark/main_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--num_ps 1





 \
--images examples/mnist/csv/train/images \
--labels examples/mnist/csv/train/labels \
--format csv \
--mode train \
--model mnist_model



JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre
export PYSPARK_PYTHON=/usr/bin/python3 



 echo 0 | sudo tee /sys/devices/system/cpu/cpu1/online
 echo 0 | sudo tee /sys/devices/system/cpu/cpu2/online
 echo 0 | sudo tee /sys/devices/system/cpu/cpu3/online
 echo 0 | sudo tee /sys/devices/system/cpu/cpu4/online
 echo 0 | sudo tee /sys/devices/system/cpu/cpu5/online
 echo 0 | sudo tee /sys/devices/system/cpu/cpu6/online
 echo 0 | sudo tee /sys/devices/system/cpu/cpu7/online


 echo 1 | sudo tee /sys/devices/system/cpu/cpu1/online
 echo 1 | sudo tee /sys/devices/system/cpu/cpu2/online
 echo 1 | sudo tee /sys/devices/system/cpu/cpu3/online
 echo 1 | sudo tee /sys/devices/system/cpu/cpu4/online
 echo 1 | sudo tee /sys/devices/system/cpu/cpu5/online
 echo 1 | sudo tee /sys/devices/system/cpu/cpu6/online
 echo 1 | sudo tee /sys/devices/system/cpu/cpu7/online















