echo "====================================== Clearning res_spark directory ======================================"
rm /home/luigi/IdeaProjects/OpenKE_new_Spark/res_spark/*

echo "====================================== Stopping Spark Master & slaves ======================================"
$SPARK_HOME/sbin/stop-slave.sh
$SPARK_HOME/sbin/stop-master.sh


echo "====================================== Starting Spark Master & slaves ======================================"
$SPARK_HOME/sbin/start-master.sh; $SPARK_HOME/sbin/start-slave.sh -c $CORES_PER_WORKER -m $MEMORY_PER_WORKER spark://$(hostname):7077
n=15
m=$((n-1))

for i in `seq 0 $m`
do
  if [ -f /content/drive/My\ Drive/DBpedia/$n/$i/res.txt ]; then
    echo "Batch $i already done; Skipping batch $i"
	  continue
  fi

  if [ $i != 0 ]; then
    k=$((i-1))
    cd $WORK_DIR_PREFIX/res_spark
    if [ "$(ls -1A | wc -l)" -eq 0 ] ; then
      echo "Copying model into res_spark dir"
      cp /content/drive/My\ Drive/DBpedia/$n/$k/model/* $WORK_DIR_PREFIX/res_spark/
    fi

    cd /content/drive/My\ Drive/DBpedia/$n/$i
    if [ "$(ls -1A | wc -l)" -le 9 ] ; then
      echo "Copying data into new batch dir"
		  cp /content/drive/My\ Drive/DBpedia/$n/$k/entity2id.txt /content/drive/My\ Drive/DBpedia/$n/$k/relation2id.txt /content/drive/My\ Drive/DBpedia/$n/$k/test2id.txt /content/drive/My\ Drive/DBpedia/$n/$k/valid2id.txt /content/drive/My\ Drive/DBpedia/$n/$k/train2id.txt /content/drive/My\ Drive/DBpedia/$n/$i/
    fi

    cd /content
  fi


	echo "====================================== Starting Training for batch $i ======================================"
	$SPARK_HOME/bin/spark-submit --master spark://$(hostname):7077 \
	--py-files $WORK_DIR_PREFIX/distribute_training.py,$WORK_DIR_PREFIX/Config.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransE.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransH.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransR.py,$WORK_DIR_PREFIX/Model.py,$WORK_DIR_PREFIX/TransD.py \
    --driver-library-path=$LIB_CUDA \
    --conf spark.dynamicAllocation.enabled=false \
    --conf spark.task.cpus=$CORES_PER_WORKER --executor-memory $MEMORY_PER_WORKER \
    --num-executors $SPARK_WORKER_INSTANCES \
	$WORK_DIR_PREFIX/main_spark.py \
    --cluster_size $SPARK_WORKER_INSTANCES --num_ps 1 --num_gpus 1 --cpp_lib_path $WORK_DIR_PREFIX/release/Base.so \
	--input_path /content/drive/My\ Drive/DBpedia/$n/$i/ \
    --output_path $WORK_DIR_PREFIX/res_spark \
    --alpha 0.0001 --optimizer SGD --train_times 50 --ent_neg_rate 1 --embedding_dimension 64 --margin 1.0 --model TransE


	echo "====================================== Copying data for batch $i ======================================"
	cp $WORK_DIR_PREFIX/res_spark/* /content/drive/My\ Drive/DBpedia/$n/$i/model/
	j=$((i+1))
	if [ $j != $n ]; then
		cp /content/drive/My\ Drive/DBpedia/$n/$i/entity2id.txt /content/drive/My\ Drive/DBpedia/$n/$i/relation2id.txt /content/drive/My\ Drive/DBpedia/$n/$i/test2id.txt /content/drive/My\ Drive/DBpedia/$n/$i/valid2id.txt /content/drive/My\ Drive/DBpedia/$n/$i/train2id.txt /content/drive/My\ Drive/DBpedia/$n/$j/
	fi

	echo "====================================== Test for batch $i ======================================"
	python3 $WORK_DIR_PREFIX/test.py $i $n 64 | tee /content/drive/My\ Drive/DBpedia/$n/$i/res.txt

done

