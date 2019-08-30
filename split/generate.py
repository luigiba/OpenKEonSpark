#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 20:10:44 2019

You can use this script to split your dataset into batches 
to feed into OpenKEonSpark

@author: Luigi Baldari
"""
import math
from random import shuffle
import os

#The relation of interest
TARGET_RELATION = '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'

#dataset containing all the triples with only the target relation
path_t = "/home/luigi/Desktop/DBpedia/instancetype_en.nt"

#dataset containing all the other triples (except the target relation)
path_r = "/home/luigi/Desktop/DBpedia/newstrictmappingbased_en.nt"

#path which contains the class hierarchy
path_h = "class_hierarchy.txt"

#path that will contains the ontology constraints for each batch
path_ont_con = "ontology_constrain.txt"

#if set to True, triples with Data Propertiy relations will be discarded
# only triples with Object Propertiy relations will be taken into account
SKIP_DATA_PROPERTY = True

#if set to True, it will generate ontology files constraints
GENERATE_ONTOLOGY_FILES = True

#number of batches to split the dataset into
N_BATCHES = 1

#percentage of triples with target relation for each batch test set 
TEST_SET_PERCENTAGE = 10

#percentage of triples with target relation for each batch validation set 
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



lef = 0
rig = 0
processed_entities = set()
try: os.mkdir(str(N_BATCHES))
except: pass


for b in range(0, N_BATCHES):
    try: os.mkdir(str(N_BATCHES)+'/'+str(b))
    except: pass
    
    try: os.mkdir((str(N_BATCHES)+'/'+str(b)+'/model'))
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
        with open("./"+str(N_BATCHES)+'/'+str(b)+"/"+path_ont_con, 'w') as f_ont_con:
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
        if triples[i].split(" ")[1].strip() == TARGET_RELATION:
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









            
            
    
    