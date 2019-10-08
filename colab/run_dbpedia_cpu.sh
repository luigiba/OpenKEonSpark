#this is an example of script that can be used on google colab to train the embedding and evaluate them
# the link prediction evaluation will be performed only for the last batch
# for the other batch it will be performed only triple classification evaluation
#before starting the script: run the commands in commands.txt
#This script has been written in order to be recoverable: if your Google Colab crashes, you can restart the training/evaluation
#from the last checkpoint.


echo "====================================== Parameters ======================================"
echo "Number of batches: $1"
echo "Embedding dimensionality: $2"
echo "Model to use: $3"
echo "Learning rate: $4"

echo "====================================== Clearning res_spark directory ======================================"
rm $WORK_DIR_PREFIX/res_spark/*

echo "====================================== Stopping Spark Master & slaves ======================================"
$SPARK_HOME/sbin/stop-slave.sh
$SPARK_HOME/sbin/stop-master.sh


echo "====================================== Starting Spark Master & slaves ======================================"
$SPARK_HOME/sbin/start-master.sh; $SPARK_HOME/sbin/start-slave.sh -c $CORES_PER_WORKER -m $MEMORY_PER_WORKER spark://$(hostname):7077
n=$1
m=$((n-1))

#iterate over batches
for i in `seq 0 $m`
do
  #restore link prediction evaluation if evaluation checkpoints have been founded
  if [ -f /content/drive/My\ Drive/DBpedia/$n/$i/model/thread0 ]; then
    echo "====================================== Restart link prediction evaluation for batch $i ======================================"
	  $SPARK_HOME/bin/spark-submit --master spark://$(hostname):7077 \
    --py-files $WORK_DIR_PREFIX/distribute_training.py,$WORK_DIR_PREFIX/Config.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransE.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransH.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransR.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransD.py \
    --driver-library-path=$LIB_CUDA --conf spark.dynamicAllocation.enabled=false --conf spark.task.cpus=$CORES_PER_WORKER --executor-memory $MEMORY_PER_WORKER --num-executors $SPARK_WORKER_INSTANCES \
	  $WORK_DIR_PREFIX/main_spark.py \
    --cluster_size $SPARK_WORKER_INSTANCES --num_ps 1 --num_gpus 0 --cpp_lib_path $WORK_DIR_PREFIX/release/Base.so \
	  --input_path /content/drive/My\ Drive/DBpedia/$n/$i/ --output_path $WORK_DIR_PREFIX/res_spark --cpp_lib_path $WORK_DIR_PREFIX/release/Base.so \
    --embedding_dimension $2 --model $3 --mode evaluation | tee /content/drive/My\ Drive/DBpedia/$n/$i/res.txt

	  continue
  fi


  if [ -f /content/drive/My\ Drive/DBpedia/$n/$i/res.txt ]; then
    echo "Batch $i already done; Skipping batch $i"
	  continue
  fi


  if [ -f /content/drive/My\ Drive/DBpedia/$n/$i/model/checkpoint ]; then
    echo "====================================== Start Link Prediction evaluation for batch $i ======================================"
    if [ $i -eq $m ]; then
	    $SPARK_HOME/bin/spark-submit --master spark://$(hostname):7077 \
    --py-files $WORK_DIR_PREFIX/distribute_training.py,$WORK_DIR_PREFIX/Config.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransE.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransH.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransR.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransD.py \
    --driver-library-path=$LIB_CUDA --conf spark.dynamicAllocation.enabled=false --conf spark.task.cpus=$CORES_PER_WORKER --executor-memory $MEMORY_PER_WORKER --num-executors $SPARK_WORKER_INSTANCES \
	  $WORK_DIR_PREFIX/main_spark.py \
    --cluster_size $SPARK_WORKER_INSTANCES --num_ps 1 --num_gpus 0 --cpp_lib_path $WORK_DIR_PREFIX/release/Base.so \
	  --input_path /content/drive/My\ Drive/DBpedia/$n/$i/ --output_path $WORK_DIR_PREFIX/res_spark --cpp_lib_path $WORK_DIR_PREFIX/release/Base.so \
    --embedding_dimension $2 --model $3 --mode evaluation | tee /content/drive/My\ Drive/DBpedia/$n/$i/res.txt
    else
      python3 $WORK_DIR_PREFIX/test.py /content/drive/My\ Drive/DBpedia/$n/$i/ /content/drive/My\ Drive/DBpedia/$n/$i/model/ $WORK_DIR_PREFIX/release/Base.so $2 $3 | tee /content/drive/My\ Drive/DBpedia/$n/$i/res.txt
    fi
  fi


  if [ $i != 0 ]; then
    k=$((i-1))
    cd $WORK_DIR_PREFIX/res_spark
    if [ "$(ls -1A | wc -l)" -eq 0 ] ; then
      echo "Copying model into res_spark dir"
      cp /content/drive/My\ Drive/DBpedia/$n/$k/model/* $WORK_DIR_PREFIX/res_spark/
    fi

    cd /content/drive/My\ Drive/DBpedia/$n/$i
    if [ "$(ls -1A | wc -l)" -le 10 ] ; then
      echo "Copying data into new batch dir"
		  cp /content/drive/My\ Drive/DBpedia/$n/$k/entity2id.txt /content/drive/My\ Drive/DBpedia/$n/$k/relation2id.txt /content/drive/My\ Drive/DBpedia/$n/$k/test2id.txt /content/drive/My\ Drive/DBpedia/$n/$k/valid2id.txt /content/drive/My\ Drive/DBpedia/$n/$k/train2id.txt /content/drive/My\ Drive/DBpedia/$n/$i/
    fi

    cd /content
  fi


	echo "====================================== Starting Training for batch $i ======================================"
	$SPARK_HOME/bin/spark-submit --master spark://$(hostname):7077 \
	--py-files $WORK_DIR_PREFIX/distribute_training.py,$WORK_DIR_PREFIX/Config.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransE.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransH.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransR.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransD.py \
    --driver-library-path=$LIB_CUDA --conf spark.dynamicAllocation.enabled=false --conf spark.task.cpus=$CORES_PER_WORKER --executor-memory $MEMORY_PER_WORKER \
    --num-executors $SPARK_WORKER_INSTANCES \
	$WORK_DIR_PREFIX/main_spark.py \
    --cluster_size $SPARK_WORKER_INSTANCES --num_ps 1 --num_gpus 0 --cpp_lib_path $WORK_DIR_PREFIX/release/Base.so \
	--input_path /content/drive/My\ Drive/DBpedia/$n/$i/ \
    --output_path $WORK_DIR_PREFIX/res_spark \
    --alpha $4 --optimizer SGD --train_times 50 --ent_neg_rate 1 --embedding_dimension $2 --margin 1.0 --model $3


	echo "====================================== Copying model for batch $i ======================================"
	cp $WORK_DIR_PREFIX/res_spark/* /content/drive/My\ Drive/DBpedia/$n/$i/model/

	
	echo "====================================== Start evaluation for batch $i ======================================"
	if [ $i -eq $m ]; then
	  $SPARK_HOME/bin/spark-submit --master spark://$(hostname):7077 \
    --py-files $WORK_DIR_PREFIX/distribute_training.py,$WORK_DIR_PREFIX/Config.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransE.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransH.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransR.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransD.py \
    --driver-library-path=$LIB_CUDA --conf spark.dynamicAllocation.enabled=false --conf spark.task.cpus=$CORES_PER_WORKER --executor-memory $MEMORY_PER_WORKER --num-executors $SPARK_WORKER_INSTANCES \
	  $WORK_DIR_PREFIX/main_spark.py \
    --cluster_size $SPARK_WORKER_INSTANCES --num_ps 1 --num_gpus 0 --cpp_lib_path $WORK_DIR_PREFIX/release/Base.so \
	  --input_path /content/drive/My\ Drive/DBpedia/$n/$i/ --output_path $WORK_DIR_PREFIX/res_spark --cpp_lib_path $WORK_DIR_PREFIX/release/Base.so \
    --embedding_dimension $2 --model $3 --mode evaluation | tee /content/drive/My\ Drive/DBpedia/$n/$i/res.txt
	else
	  python3 $WORK_DIR_PREFIX/test.py /content/drive/My\ Drive/DBpedia/$n/$i/ /content/drive/My\ Drive/DBpedia/$n/$i/model/ $WORK_DIR_PREFIX/release/Base.so $2 $3 | tee /content/drive/My\ Drive/DBpedia/$n/$i/res.txt
	fi

done

