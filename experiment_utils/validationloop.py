
import torch_geometric 
import numpy as np
from utils.seeds import val_seeds
from utils.splits import set_train_val_test_split, set_train_val_test_split_frac
from .experimentclass import Experiment
from torch_geometric.data import Data

def validation_loop(datasetname: int,dataset,data,data_rewired,iterations:int = 100,citationgraph = False,runnorewire = False):
    test_accuracies = []
    test_accuracies_rewired = []
    test_accuracies_rewired_2 = []
    print("===============")
    for it in range(iterations):
        if citationgraph:
            data_undirected_split = set_train_val_test_split(val_seeds[it],data)
        else:
            data_undirected_split = set_train_val_test_split_frac(val_seeds[it],data, val_frac = 0.2, test_frac = 0.2)


        data_undirected_split_rewired = Data(x = data_undirected_split.x,y = data_undirected_split.y,
                                             test_mask = data_undirected_split.test_mask,val_mask =  data_undirected_split.val_mask, train_mask = data_undirected_split.train_mask,
                                             edge_index = data_rewired.edge_index)

    
        if it %2 ==0:
            print(f"Iterations Number {it} started")

        Exp_rewired = Experiment(datasetname,dataset,data_undirected_split_rewired)

        losses_rewired,validations_rewired = Exp_rewired.training()
        test_accuracies_rewired.append(Exp_rewired.test())

        Exp_rewired_2 = Experiment(datasetname + "_rewired",dataset,data_undirected_split_rewired)
        
        losses_rewired_2,validations_rewired_2 = Exp_rewired_2.training()
        test_accuracies_rewired_2.append(Exp_rewired_2.test())

    

        if runnorewire:
            Exp = Experiment(datasetname,dataset,data_undirected_split)
            losses,validations = Exp.training()
            test_accuracies.append(Exp.test())
    test_accuracies_rewired = np.array(test_accuracies_rewired)
    test_accuracies_rewired_2 = np.array(test_accuracies_rewired_2)

    if runnorewire:
        test_accuracies = np.array(test_accuracies)
        return test_accuracies,test_accuracies_rewired,test_accuracies_rewired_2
    else:
        return test_accuracies_rewired,test_accuracies_rewired_2
