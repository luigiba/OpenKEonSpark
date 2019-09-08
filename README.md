
# OpenKEonSpark  
This is a distributed version of the framework OpenKE (https://github.com/thunlp/OpenKE) using the library TensorFlowOnSpark (https://github.com/yahoo/TensorFlowOnSpark).  
Please refer to https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/distributed.md to have an overview about how Distributed Tensorflow training works.
  
  
## Index  

- [OpenKEonSpark](#openkeonspark)
  * [Index](#index)
  * [Overview](#overview)
  * [Installation](#installation)
    + [General installation](#general-installation)
    + [Install on Google Colab](#install-on-google-colab)
  * [Train mode](#train-mode)
  * [How to use the model learned?](#how-to-use-the-model-learned-)
    + [Link Prediction](#link-prediction)
    + [Triple classification](#triple-classification)
    + [Usage example](#usage-example)
    + [Predict head entity](#predict-head-entity)
    + [Predict tail entity](#predict-tail-entity)
    + [Predict relation](#predict-relation)
    + [Classify a triple](#classify-a-triple)
  * [Evaluation mode](#evaluation-mode)
      - [Link Prediction Evaluation](#link-prediction-evaluation)
      - [Triple Classification Evaluation](#triple-classification-evaluation)
  * [How to generate all these files?](#how-to-generate-all-these-files-)
  * [How to run the distributed application?](#how-to-run-the-distributed-application-)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

  
  
## Overview  
  
**OpenKE** is an Efficient implementation based on TensorFlow for knowledge representation learning.  C++ is used to implement some underlying operations such as data preprocessing and negative sampling. Knowledege Graph Embedding models (TransE, TransH, TransR, TransD) are implemented using TensorFlow with Python interfaces so that there is a convenient platform to run models on GPUs and CPUs.   
  
**OpenKEonSpark** is the is the distributed version of OpenKE using the library **TensorflowOnSpark** (which allows to distribute existing Tensorflow application on Spark). The motivations that have driven this project are the following: 
  1. Create a tool which efficiently allows to train and evaluate knowledge graph embedding, distributing the job among a set of resources;  
  2. As in Big data scenarios, the model created in the first point should be updatable as new batch of data arrives.  
    
The tool can be installed locally (Standalone cluster) or using an Hadoop cluster. Only the **Standalone mode** has been tested.  The framework can be executed in two different modalities: (1) **Train mode**; (2) **Evaluation mode**.   
  
## Installation  
### General installation  
  
1. Install Java 8, Python3, Scala, Spark (tested version spark-2.1.1-bin-hadoop2.7), Tensorflow, TensorflowOnSpark  
  
2. Clone the OpenKEonSpark repository  
  
3. Compile C++ files  
     
```bash  
$ cd OpenKEonSpark  
$ bash make.sh  
```  
  
### Install on Google Colab  
  
Please refer to colab directory to installation, running and evaluation pipelines
  
  
## Train mode  
  
The train mode aims to perform the training of knowledge graph embedding, distributing the job respect to the cluster of machines. The tool implements data parallelism using the between-graph replication. Moreover, it employs the asynchronous techniques to perform gradient updates. The embedding can be learnt using translational distance models (TransE, TransH, TransR and TransD) and specifying an optimization algorithm (which could be either SGD or Adam optimizer).  The training phase is performed respect to the following files:  
  
 1. **train2id.txt**: training file, the first line is the number of triples for training. Then the following lines are all in the format ***(head, tail, rel)*** which indicates there is a relation ***rel*** between ***head*** and ***tail*** .  Note that train2id.txt contains ids from entitiy2id.txt and relation2id.txt instead of the names of the entities and relations. 
  
 2. **entity2id.txt**: all entities and corresponding ids, one per line. The first line is the number of entities. 
  Note that entity2id.txt ids have to start from zero and have to be continuous (0,1,2,...).  
  
 3. **relation2id.txt**: all relations and corresponding ids, one per line. The first line is the number of relations. 
 4. **test2id.txt** (this file is used for evaluation mode): test set file, the first line is the number of triples for testing. Then the following lines are all in the format ***(head, tail, rel)*** .  
 5. **valid2id.txt**: validation set file, the first line is the number of triples which compose the validation set. Then the following lines are all in the format ***(head, tail, rel)*** . 
 
 **Test set and validation set will contain a set of triples about one or more target relations.**
 
At each epoch a bunch of triples is randomly selected from the training set and for each of it, *c* corrupted triples (where *c* is specified by the user) are generated to compute the loss function. The number of valid triples to be selected, i.e. the mini-batch size, can be specified by the user or automatically computed by the software. The second approach is the recommended one for large training set, since it keeps the mini-batch size small . Moreover, during the training, early stop can be computed respect to the validation set  accuracy or the training set loss. The accuracy is computed by means of the triple classification task, since it is a very efficient evaluation task and it does not slow the training. Each *n* number of epochs (where *n* is a parameter specified by the user), the tool check the triple classification accuracy on the validation set; if it does not improve for a set of specified *m* epochs, the training is stopped early and the model with the best accuracy is restored. The second approach is roughly the same but, instead of computing the accuracy on the validation set, the training set loss is taken into account.   
  
Once the training phase has finished, the model can be updated as new batches of data arrives. In this case has to be specified four additional files:  
 1. **batch2id.txt**: contains the new batch of training triples. The first line is the number of new triples, while the following lines are all in the format **(head,tail,rel)**.  
    
 2. **batchEntity2id.txt**: contains the new nodes that should be inserted in the knowledge graph; if the new batch triples are only about already existing nodes, no entities will be reported here. This file has the number of new entities written in the first line and all new entities name with corresponding ids written in the following lines;  
    
 3. **batchTest2id.txt**: contains the new batch of test triples. The first line is the number of new triples, while the following lines are all in the format **(head,tail,rel)**.  
    
 4. **batchValid2id.txt**: contains the new batch of validation triples. The first line is the number of new triples, while the following lines are all in the format **(head,tail,rel)**.  
    
When a new batch of data is fed, the data are firstly incorporated in the original dataset. Secondly, if there are new entities to add in the knowledge graph, the tensors which depends from them are updated accordingly. Finally, the training phase is performed to update the model using the available data. In this case the training will be executed in a different way respect to the standard one:  
  
 1. The training triples will be selected only from the new batch of triples;  
 2. In order to avoid generating corrupted triples that end up to be valid ones, the corruption phase is performed by taking into account both new batch and old batch of triples.  

The output of the training phase (the model learned) will be saved into a specified directory together with the following files:
* **time.txt**: which contains the training time in seconds
* **stop.txt**: which contains the best model epoch founded during training (if early stop has happened)

 
  
  
  
  
## How to use the model learned?  
There are two tasks already implemented in the repository, which can be used once the embeddings have been learned: **Link prediction** and **Triple classification**.  
  
### Link Prediction  
Link prediction aims to predict the missing head (using the method *predict_head_entity*), or  tail (*predict_tail_entity*) or  relation (*predict_relation*) for a relation fact triple *(h,r,t)*. In this task, for each position of missing entity, the system is asked to rank a set of *k* (additional parameter of the methods) candidate entities from the knowledge graph, instead of only giving one best result. Given a specific test triple *(h,r,t)* from which to predict a missing component (either the head or the tail or the relation), the component is replaced by all entities or relations in the knowledge graph, and these triples are ranked in descending order respect to their scores (which depend from the specific model used). Prediction can be performed respect to the missing component using the corresponding methods to get the top predictions from the ranked list.
  
### Triple classification  
Triple classification aims to judge whether a given triple *(h,r,t)* is correct or not (a binary classification task). For triple classification, is used a threshold δ: for a given triple if the score obtained by the triple is below δ, the triple will be classified as positive, otherwise as negative. The threshold can be passed as a parameter to the method *predict_triple*, or can be optimized by maximizing the classification accuracy on the validation set. 
  
### Usage example  
To use a knowledge graph embedding model (already learned) first import the embeddings and then use methods for link prediction and triple classification. For example, the following script load and use a learned model:  
  
```python  
from Config import Config  
from TransE import TransE  
  
con = Config(cpp_lib_path='OpenKEonSpark/release/Base.so')  
con.set_in_path("path/to/dataset/")   
con.set_dimension(64)   #embedding dimension
con.init()  
con.set_model_and_session(TransE)
con.set_import_files("path/to/model/")
  
'''perform your operations'''  
con.predict_head_entity(1928, 1, 5)  
con.predict_tail_entity(0, 1, 5)  
con.predict_relation(0, 1928, 5)  
con.predict_triple(0, 1928, 1)  
```  
  
### Predict head entity  
The method *predict_head_entity(t, r, k)* predicts the top *k* head entities given the tail entity (*t*) and the relation (*r*).  
```python  
con.predict_head_entity(1928, 1, 5)  
```  
### Predict tail entity  
This is similar to predicting the head entity. The method to use is *predict_tail_entity(h, r, k)* which predicts the top *k* tail entities given the head entity (*h*) and the relation (*r*).  
```python  
con.predict_tail_entity(0, 1, 5)  
```  
### Predict relation  
Given the head entity and tail entity, predict the top k possible relations. All the objects are represented by their id. 
```python  
con.predict_relation(0, 1928, 5)  
```  
  
### Classify a triple  
Given a triple *(h, r, t)*, this funtion tells us whether the triple is correct or not. If the threshold is not given, this function calculates the threshold for the relation from the validation set .  
```python  
con.predict_triple(0, 1928, 1)  
```  
  
  
## Evaluation mode 
  
Using the **evaluation mode**, the link prediction evaluation task is distributed among the workers, since this is a very expensive task. Two additional files are required:
  
  1. **type_constrain.txt**: the first line is the number of relations; the following lines are type constraints for each relation. For example, the line “1200 4 3123 1034 58 5733” means that the relation with id 1200 has 4 types of head entities (if another line with relation id equals to 1200 is written, it will refer to tail entities constraints), which are 3123, 1034, 58 and 5733.  
  2. **ontology_constrain.txt**: the first line is the number of classes in the ontology; the following lines are ontology constraints for each ontology class. For example, the line "100 3 10 20 30" means that the class with id 100 has three **super classes** (if another line with class id equals to 100 is written, it will refer to **sub-classes** ontology constraints), which are 10, 20 and 30.  

The first file is used to incorporate additional information (i.e. the entity types) during the prediction phase. The second file usage will be better explained in the next section.
  
#### Link Prediction Evaluation  
The protocol used for link prediction iterates over the set of test triples and, for each triple, the following operations are performed:    
* a set of corrupted triplets is created by removing the test tiples’ head and replacing it by each of the existing entities in turn;   
*  the set composed by the original test triple with its corrupted triples is created;
*  for each triple in the set is computed the score function of the trained model; 
* the triples in the set are ranked respect to the score function (so that on the top of the list, we should have the most recommendable triples); logically, the algorithm performs well if on top of the list are present valid triplets.
    
The evaluation metrics (which will be explained later) are computed using this list. The procedure is repeated for each test triple and, at the end, average results are reported. Moreover, these steps are performed again while removing the tail (referred as **r** in this process, i.e. the right part of the triple) instead of the head (referred as **l** in this process, i.e. the left part of the triple). Using this procedure, it could be possible that, the set of corrupted triples  for a specific test triplet, may contain triples which end up being valid ones because are already contained either in the training, validation or test set. For this reason, the results are divided respect to two different criteria: the original corrupted set (called **raw**, i.e. using the procedure explained above) and **filter** i.e. without the corrupted triples (except the test triple of interest of course) which appear in either the training, validation or the test set (to ensure that all corrupted triples do not belong to the data set). Moreover, the results are further divided respect to if they are taking in account the triple constraints (present in the file **type_constrain.txt**) or not. Taking into account the constraints means that, during the creation of the corrupted set, the triples that does not respect the relations’ constraints will be discarded. For link prediction evaluation, the metrics given as output are the following: Mean rank (**MR**), Mean Reciprocal Rank (**MRR**) and **Hit@N** where N could be ten, three or one.
The results are then divided using different criteria:  
* Using / not using the relation constraints available in the file type_constrain.txt;  
* Using raw / filter (without the corrupted triples which appear in either the training or the validation or the test set) set;  
* Using the evaluation process for the head ( l ) or the tail ( r ).   

In the **distributed evaluation mode**, test triples are equally assigned to each worker. So, each of them has the responsibility to perform the link prediction evaluation for a specific set of test triples. Once all workers have finished, the results are collected and merged from the client, which shows the final results. This task is easily distributable because each worker has just to compute triples’ rank; the order in which they are computed doesn’t matter. Then, it will be a client’s responsibility to merge these results.

Moreover, since many of the published linked data are based on **ontologies**, the link prediction task has been empowered to take into account this information. The ontology has to be written in the file **ontology_constrain.txt**, which has been described early.  The ontology information is used while computing the metric **Hit@1**, i.e. the proportion of correct entities in the top-1 ranked entities. Since this metric returns a number of predictions equals to one, this task can be seen as a hierarchical (because of the ontology) multi-class classification task, where the missing piece of the triple represent the class to predict. The errors made from the model can be divided considering the ontology structure:
* **Generalization Error**, which is the percentage of classes classified as a super-category of the correct category;
* **Specialization Error**, which is the percentage of classes classified into a subcategory of the correct category;
* **Misclassification Error**, which is the percentage of classes classified as a category which is in a different path with respect to the correct category in the hierarchy.


  
  
#### Triple Classification Evaluation  
The protocol used for triple classification evaluation is easier respect to the previous one. Since this task needs negative labels, each golden test triple is corrupted to get only one negative triple (by corrupting the tail). The resulting set of triples will contain twice the number of test triples. The same procedure is repeated for the validation set.  As explained before, the classification task needs a threshold, which is learned using the validation set. In this case, the validation set is used only to tune the threshold for classification purpose. Finally, the learned threshold, is used to classify the whole set of test triples.
The metric reported from this task depend from the number of target relations present in the test set:
* If there is only one target relation, it will be reported standard metrics (accuracy, precision, recall and f-measure)
* If there are more than one target relations it will be reported micro averaged metrics.

This task  has not been distributed because it is already very efficient. However, since it depends from a decision threshold, a better performance estimates of this task can be get using ROC curves. The method *plot_roc* of the *Config* class, can be used to plot it respect to a specific relation. Moreover it is possible to compute its area under the curve using the test triples and the decision thresholds computed from the validation set. Please refer to the file *test.py* to perform this task.
  
## How to generate all these files?
In order to generate all the files mentioned here from a set of triples, the script *generate.py* can be used: it assigns a numerical identifier to each resource, split the data into training, test and validation set and one or more batches. It accepts triples serialized using N-Triples format. The test and validation set are created by selecting from the available data a specified percentage of triples. These triples regard a set of one or more target relations that we are seeking to learn.


## How to run the distributed application?
Before launching the main program (**main_spark.py**), common Spark parameters have to be specified, such as the cluster dimension, the resources to allocate for each cluster (CPUs, GPUs, RAM), the number of parameter servers and the number of workers. The script below reports the essential ones:
```
#CUDA
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LIB_CUDA=/usr/local/cuda-10.0/lib64

#JAVA
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre

#PYTHON
export PYSPARK_PYTHON=/usr/bin/python3 

#SPARK
export SPARK_HOME=/path/to/spark/spark-2.1.1-bin-hadoop2.7
export PATH=$SPARK_HOME/bin:$PATH
export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=3
export CORES_PER_WORKER=1
export MEMORY_PER_WORKER=2g
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES})) 

``` 
Subsequently, master and slaves have to be started using the following command:
```
${SPARK_HOME}/sbin/start-master.sh
${SPARK_HOME}/sbin/start-slave.sh -c ${CORES_PER_WORKER} -m ${MEMORY_PER_WORKER} ${MASTER}

```
The distributed application can be finally launched using the *main_spark.py* script; it accepts the following parameters:
* --cluster_size: number of nodes in the cluster
* --num_ps: number of ps (parameter server) nodes
* --num_gpus: number of gpus to use
* --cpp_lib_path: cpp lib.so absolute path
* --input_path: dataset absolute path
* --output_path: model output absolute path
* --train_times: number of epochs
* --n_mini_batches: number of mini batches; if set to zero it will be automatically computed
* --alpha: learning rate
* --margin: margin hyperparameter used during training
* --bern_flag: whether to use or not bern method for sampling;
* --embedding_dimension: embedding dimension (for both entities and rel)
* --ent_dimension: entities embedding dimension
* --rel_dimension: relations embedding dimension
* --ent_neg_rate: number of negative triples generated by corrupting the entity
* --rel_neg_rate: number of negative triples generated by corrupting the relation
* --optimizer: Optimization algorithm (SGD/Adam)
* --early_stop_patience: no. epochs to wait for accuracy/loss improvement before early stop
* --early_stop_stopping_step: perfroms early stop each stopping step
* --early_stop_start_step: perfroms early stop from start step
* --model: model to be used (TransE/TransH/TransR/TransD)
* --debug: if Ture prints additional debug information
* --mode: whether to perform train or evaluation mode
* --test_head: perform link prediction evaluation on missing head, too (only if mode != 'train')

The following script reports an example to launch the distributed application using the train mode.
```
${SPARK_HOME}/bin/spark-submit --master ${MASTER} \
--py-files /path/to/OpenKEonSpark/distribute_training.py,/path/to/OpenKEonSpark/Config.py,/path/to/OpenKEonSpark/Model.py,/path/to/OpenKEonSpark/TransE.py,/path/to/OpenKEonSpark/Model.py,/path/to/OpenKEonSpark/TransH.py,/path/to/OpenKEonSpark/Model.py,/path/to/OpenKEonSpark/TransR.py,/path/to/OpenKEonSpark/Model.py,/path/to/OpenKEonSpark/TransD.py \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.cores.max=${TOTAL_CORES} --conf spark.task.cpus=${CORES_PER_WORKER} --executor-memory ${MEMORY_PER_WORKER} --num-executors ${SPARK_WORKER_INSTANCES} \
/path/to/OpenKEonSpark/main_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} --num_ps 1 --num_gpus 0 \
--input_path /path/to/dataset/ --output_path /path/where/to/store/model/ --cpp_lib_path /path/to/OpenKEonSpark/release/Base.so \
--alpha 0.00001 --optimizer SGD --train_times 50 --ent_neg_rate 1 --embedding_dimension 64 --model TransE --mode train 
```
When the program finished, master and slaves can be stopped using the following command:
```
${SPARK_HOME}/sbin/stop-slave.sh; ${SPARK_HOME}/sbin/stop-master.sh
```

