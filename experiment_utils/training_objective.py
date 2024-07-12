from utils.splits import set_train_val_test_split,set_train_val_test_split_frac
from experiment_utils.experimentclass import Experiment
from experiment_utils.create_rewired_edge_index import create_rewired_edge_index

from utils.seeds import val_seeds

import numpy as np

from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", device)

def objective(config,datasetname,dataset_lcc,data_lcc,curvature_type,int_node = False,rewire = False):

    val_acc = []
    test_acc = []

    if rewire:
        print("===Starting Rewiring===")
        _,edge_index_rewired = create_rewired_edge_index(data_lcc,config,intermediate_node=int_node,remove_edges=True,curvaturetype=curvature_type)
        print(" ")

    print(" == Starting Runs == ")
    for _,k in tqdm(enumerate(val_seeds)):

        if datasetname == "Cora" or datasetname == "Citeseer" or datasetname == "Pubmed":
            data_undirected_split = set_train_val_test_split(k,data_lcc)
        else:
            data_undirected_split = set_train_val_test_split_frac(k,data_lcc,0.2,0.2)

        if rewire:
            
            data_undirected_split.edge_index = edge_index_rewired

        data_undirected_split.to(device)

        Exp = Experiment(device,datasetname,dataset_lcc,data_undirected_split,config)

        
        counter = 0
        for epoch in range(1, Exp.epoch):
            loss = Exp.train()
            val = Exp.validate()
            
            if epoch ==1:
                best_val = val
            elif epoch > 1 and val > best_val:
                best_val = val
                counter = 0
            else:
                counter += 1
            if counter > 100:
                break  
        final_accuracy = Exp.validate()
        final_test_acc = Exp.test()
        val_acc.append(final_accuracy)
        test_acc.append(final_test_acc)
    print("")
    return np.mean(np.array(val_acc)),np.mean(np.array(test_acc))