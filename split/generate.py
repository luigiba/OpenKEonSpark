#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 20:10:44 2019


Starting from your N-triples files, you can use this script to
    assign numerical identifiers to resources
    split data into training/test/validation set
    split data into batches
    generate ontology_constrain file
    print useful statistics

This script will output a directory which is named as n, where n is number of batches used to split the dataset.
Inside the directory there will be n-1 (named from 0 to n-1) sub-folders. Each sub-folder contains a different batch.
Moreover, inside each sub-batch folder i, it will be created the folder "model" that will be used to save the model up to batch i.

@author: Luigi Baldari

@:param
    TARGET_RELATION
        Specify in this array the target relation/s
        Training set will contain all the relations (target included)
        test set and validation set will contain ontly target relation/s
    path_t
        Path to N-Triples dataset containing all the triples with only the target relation/s
    path_r
        Path to N-Triples dataset containing all the other triples (except the target relation/s)
    path_h
        Path which contains the class hierarchy (from which the ontology_constrain.txt will be created)
        The root node (i.e. the first line of the class hierarchy file) contains the most general class.
        As new specific classes are met, a new level of indentation has to be added to the new line.
        In this way the classes which have the same number of tabs are on the same level of the tree.
        The file class_hierachy.txt contains an example of how the class hierarchy should be formatted to be fed into this scipt
    SKIP_DATA_PROPERTY
        if set to True, triples with Data Propertiy relations will be discarded and only triples with Object Propertiy relations will be taken into account
    GENERATE_ONTOLOGY_FILES
        if set to True, it will generate ontology_constrain.txt
    N_BATCHES
        number of batches to split the dataset into
    TEST_SET_PERCENTAGE
        percentage of triples with target relation/s for each batch test set
    VALIDATION_SET_PERCENTAGE
        percentage of triples with target relation/s for each batch validation set
"""

import math
from random import shuffle
import os


TARGET_RELATION = ['<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>']
path_t = "/home/luigi/Desktop/DBpedia/instancetype_en.nt"
path_r = "/home/luigi/Desktop/DBpedia/newstrictmappingbased_en.nt"
path_h = "class_hierarchy.txt"
SKIP_DATA_PROPERTY = True
GENERATE_ONTOLOGY_FILES = True
N_BATCHES = 10
TEST_SET_PERCENTAGE = 10
VALIDATION_SET_PERCENTAGE = 10





print(" ===== Generating triples ===== ")
lines_t = []
lines_r = []
triples = []
skipped = []
entities = set()
relations = set()
count_skipped_data = 0
type_tails = set()
type_tails_dict = {}


print(" ===== Reading TARGET triples ===== ")
with open(path_t, "r") as f:
    lines_t = f.readlines()
    
for l in lines_t:
    l = l.replace(" .\n", "")
    entities.add(l.split(" ")[0].strip())
    type_tails.add(l.split(" ")[2].strip())
    triples.append(l)
    
for type_t in type_tails:
    type_t_arr = type_t.split("/")
    type_tails_dict[type_t_arr[len(type_t_arr) - 1].replace('>', '')] = type_t      #.replace('owl#', '') deleted


if GENERATE_ONTOLOGY_FILES:                                       
    print(" ===== Reading Class Hierarchy ===== ")                                       
    lines_h = []
    with open(path_h, 'r') as file_h:
        lines_h = file_h.readlines()
    
    i = 0
    classes = {}
    current_class = type_tails_dict[lines_h[i].strip()]
    classes[current_class] = {}
    classes[current_class]['sup'] = set()
    classes[current_class]['sub'] = set()
    levels = {str(i):current_class}
    
    for i in range(1, len(lines_h)):
        try:
            current_level = lines_h[i].count('\t')
            current_class = type_tails_dict[lines_h[i].strip()]
            
            levels[str(current_level)] = current_class
            classes[current_class] = {}
            classes[current_class]['sup'] = set()
            classes[current_class]['sub'] = set()
            
            #register i's super classes
            for j in range(0, current_level):
                classes[current_class]['sup'].add(levels[str(j)])
                
            #register sub classes
            for j in range(0, current_level):
                classes[levels[str(j)]]['sub'].add(current_class)  
                
        except KeyError:
            print(" WARN:\tClass " + lines_h[i].strip() + " does not exists in the dataset; It will be discarded")                                         
                            
                        
print(" ===== Reading other triples ===== ")
with open(path_r, "r") as f:
    lines_r = f.readlines()
    
for l in lines_r:
    l = l.replace(" .\n", "")
    arr = l.split(" ")
    
    head = arr[0].strip()
    tail = arr[2].strip()
    

    if SKIP_DATA_PROPERTY and tail.startswith('<') == False: 
        count_skipped_data += 1
        skipped.append(l)
        continue
    
    triples.append(l) 

if SKIP_DATA_PROPERTY: print(" LOG:\tSkipped Triples (Literals): {}".format(count_skipped_data))

triples.sort()


        


###############################################################################
print(" ===== Setting entities & relations ids ===== ")

entities = {}
relations = {}
ent_count = 0
rel_count = 0

for l in triples:
    l = l.split(" ")
    
    h = l[0].strip()
    if h not in entities.keys():
        entities[h] = ent_count
        ent_count += 1
        
    t = l[2].strip()
    if t not in entities.keys():
        entities[t] = ent_count
        ent_count += 1
        
    r = l[1].strip()
    if r not in relations.keys():
        relations[r] = rel_count
        rel_count += 1



###############################################################################

print(" ===== Splitting into batches ===== ")

n_triples = len(triples)
BATCH_SIZE = math.floor(n_triples / N_BATCHES)

print(" LOG:\tNumber of triples: {}\n".format(n_triples))
print(" LOG:\tNumber of entities: {}\n".format(len(entities.keys())))
print(" LOG:\tNumber of relations: {}\n".format(len(relations.keys())))


lef = 0
rig = 0
processed_entities = set()
try: os.mkdir(str(N_BATCHES))
except: pass


for b in range(0, N_BATCHES):
    try: os.mkdir(str(N_BATCHES)+'/'+str(b))
    except: pass
    
    try: os.mkdir((str(N_BATCHES)+'/'+str(b)+'/model'))     #create dir where the model will be stored
    except: pass
    
    if b+1 == N_BATCHES: rig = n_triples
    else: rig += int(BATCH_SIZE)    
        
    batch_triples_ids = [i for i in range(lef, rig)]
    batch_training_set = []
    batch_entities = []
    batch_test_set = []
    batch_valid_set = []
    
    print(" LOG:\tBatch # " + str(b))
    print(" LOG:\tNumber of batch triples: {}".format(len(batch_triples_ids)))
    
    #get and write entities for this batch
    for bt_index in batch_triples_ids:
        bt = triples[bt_index].split(" ")
        head = bt[0].strip()
        tail = bt[2].strip()
        
        if head not in processed_entities:
            processed_entities.add(head)
            batch_entities.append(head)
        if tail not in processed_entities:
            processed_entities.add(tail)
            batch_entities.append(tail)
    
    
    print(" LOG:\tGenerating entity file")
    file_name_batch = "./"+str(N_BATCHES)+'/'+str(b)+"/"
    if b == 0:
        file_name_batch += "entity2id.txt"
    else:
        file_name_batch += "batchEntity2id.txt"
    with open(file_name_batch, 'w') as f:
        f.write(str(len(batch_entities))+"\n")
        for ent in batch_entities:
            f.write(ent + "\t" + str(entities[ent]) + "\n")         
    print(" LOG:\tLast entity id is: " + str(entities[ent]))       
    
    
    if GENERATE_ONTOLOGY_FILES:
        print(" LOG:\tGenerating ontology constrain file")
        with open("./"+str(N_BATCHES)+'/'+str(b)+"/ontology_constrain.txt", 'w') as f_ont_con:
            classes_len = 0
            for key in classes.keys():
                if entities[key] <= entities[ent]: 
                    classes_len += 1    
            f_ont_con.write(str(classes_len) + '\n')
            
            for key, values in classes.items():
                if entities[key] <= entities[ent]:
                    
                    f_ont_con.write(str(entities[key]))
                    sup_list = set()
                    for v in values['sup']:
                        if entities[v] <= entities[ent]:
                            sup_list.add(entities[v])                
                    f_ont_con.write('\t'+str(len(sup_list)))
                    for v in sup_list:
                        f_ont_con.write('\t'+str(v))
                    f_ont_con.write('\n')
                
                    f_ont_con.write(str(entities[key]))
                    sub_list = set()
                    for v in values['sub']:
                        if entities[v] <= entities[ent]:
                            sub_list.add(entities[v])                
                    f_ont_con.write('\t'+str(len(sub_list)))
                    for v in sub_list:
                        f_ont_con.write('\t'+str(v))
                    f_ont_con.write('\n')
            
            
    
    #split triples in training/test/valid set
    
    #shuffle the triples
    shuffle(batch_triples_ids)
    
    #get ids of test triples
    test_triples_ids = []
    remaining = []
    for i in batch_triples_ids:
        if triples[i].split(" ")[1].strip() in TARGET_RELATION:
            test_triples_ids.append(i)
        else:
            remaining.append(i)
            
    
    TEST_SET_SIZE = int(len(test_triples_ids) * TEST_SET_PERCENTAGE / 100)
    VALIDATION_SET_SIZE = int(len(test_triples_ids) * VALIDATION_SET_PERCENTAGE / 100)
    TRAINING_SET_SIZE = BATCH_SIZE - TEST_SET_SIZE - VALIDATION_SET_SIZE
    
    batch_test_set = test_triples_ids[0 : TEST_SET_SIZE]
    batch_valid_set = test_triples_ids[TEST_SET_SIZE : TEST_SET_SIZE+VALIDATION_SET_SIZE]
    batch_training_set = test_triples_ids[TEST_SET_SIZE+VALIDATION_SET_SIZE : len(test_triples_ids)] + remaining
    
    shuffle(batch_training_set)
    
    print(" LOG:\t# of test triples: " + str(len(batch_test_set)))
    print(" LOG:\t# of valid triples: " + str(len(batch_valid_set)))
    print(" LOG:\t# of training triples: " + str(len(batch_training_set)))
          
          
    
    print(" LOG:\tGenerating train file")
    file_name_batch = "./"+str(N_BATCHES)+'/'+str(b)+"/"
    if b == 0:
        file_name_batch += "train2id.txt"
    else:
        file_name_batch += "batch2id.txt"
    with open(file_name_batch, 'w') as f:
        f.write(str(len(batch_training_set))+"\n")
        for index in batch_training_set:
            s = triples[index].split(' ')
            h = str(entities[s[0].strip()])
            t = str(entities[s[2].strip()])
            r = str(relations[s[1].strip()])
            f.write(h+' '+t+' '+r+'\n')
           
            
    print(" LOG:\tGenerating test file")
    file_name_batch = "./"+str(N_BATCHES)+'/'+str(b)+"/"
    if b == 0:
        file_name_batch += "test2id.txt"
    else:
        file_name_batch += "batchTest2id.txt"
    with open(file_name_batch, 'w') as f:
        f.write(str(len(batch_test_set))+"\n")
        for index in batch_test_set:
            s = triples[index].split(' ')
            h = str(entities[s[0].strip()])
            t = str(entities[s[2].strip()])
            r = str(relations[s[1].strip()])
            f.write(h+' '+t+' '+r+'\n')
        
        
    print(" LOG:\tGenerating valid file")
    file_name_batch = "./"+str(N_BATCHES)+'/'+str(b)+"/"
    if b == 0:
        file_name_batch += "valid2id.txt"
    else:
        file_name_batch += "batchValid2id.txt"
    with open(file_name_batch, 'w') as f:
        f.write(str(len(batch_valid_set))+"\n")
        for index in batch_valid_set:
            s = triples[index].split(' ')
            h = str(entities[s[0].strip()])
            t = str(entities[s[2].strip()])
            r = str(relations[s[1].strip()])
            f.write(h+' '+t+' '+r+'\n')
    
    
    
    
    
    print()
    lef = rig
    

print(" LOG:\tGenerating relation file")
with open("./"+str(N_BATCHES)+"/0/relation2id.txt", 'w') as f:
    f.write(str(len(relations.items()))+"\n")
    for key, value in relations.items():
        f.write(key + "\t" + str(value) + "\n")


###############################################################################

print("\n ===== Relations structure statistics ===== ")

for b in range(N_BATCHES):
    print(" ===== Batch {} ===== ".format(b))
    
    lef = {}
    rig = {}
    
    if b == 0: triple = open( "./"+str(N_BATCHES)+'/'+str(b)+"/train2id.txt", "r")
    else: triple = open( "./"+str(N_BATCHES)+'/'+str(b)+"/batch2id.txt", "r")
    
    if b == 0: valid = open("./"+str(N_BATCHES)+'/'+str(b)+"/valid2id.txt", "r")
    else: valid = open("./"+str(N_BATCHES)+'/'+str(b)+"/batchValid2id.txt", "r")
    
    if b == 0: test = open("./"+str(N_BATCHES)+'/'+str(b)+"/test2id.txt", "r")
    else: test = open("./"+str(N_BATCHES)+'/'+str(b)+"/batchTest2id.txt", "r")
    
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
    
    test.close()
    valid.close()
    triple.close()
    
    
    rellef = {}
    totlef = {}
    relrig = {}
    totrig = {}
    
    for i in lef:
    	if not i[1] in rellef:
    		rellef[i[1]] = 0
    		totlef[i[1]] = 0
    	rellef[i[1]] += len(lef[i])
    	totlef[i[1]] += 1.0
    
    for i in rig:
    	if not i[0] in relrig:
    		relrig[i[0]] = 0
    		totrig[i[0]] = 0
    	relrig[i[0]] += len(rig[i])
    	totrig[i[0]] += 1.0
        
        
    
    for file in {"batch2id.txt", "batchTest2id.txt", "batchValid2id.txt"}:
        if b == 0 and file == "batch2id.txt": file = "train2id.txt"
        if b == 0 and file == "batchTest2id.txt": file = "test2id.txt"
        if b == 0 and file == "batchValid2id.txt": file = "valid2id.txt"
        
        s11=0
        s1n=0
        sn1=0
        snn=0
        f = open("./"+str(N_BATCHES)+'/'+str(b)+"/"+file, "r")
        tot = (int)(f.readline())
        for i in range(tot):
        	content = f.readline()
        	h,t,r = content.strip().split()
        	rign = rellef[r] / totlef[r]
        	lefn = relrig[r] / totrig[r]
        	if (rign <= 1.5 and lefn <= 1.5):
        		s11+=1
        	if (rign > 1.5 and lefn <= 1.5):
        		s1n+=1
        	if (rign <= 1.5 and lefn > 1.5):
        		sn1+=1
        	if (rign > 1.5 and lefn > 1.5):
        		snn+=1
        f.close()
        
        print(" LOG:\t# of 1-to-1 triples in {}:{}".format(file, s11))
        print(" LOG:\t# of 1-to-N triples in {}:{}".format(file, s1n))
        print(" LOG:\t# of N-to-1 triples in {}:{}".format(file, sn1))
        print(" LOG:\t# of N-to-N triples in {}:{}".format(file, snn))
        print()
        
        









            
            
    
    