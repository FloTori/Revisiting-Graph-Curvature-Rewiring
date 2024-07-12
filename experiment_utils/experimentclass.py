from models.GCN import GCN,GCN_Toppingetal,GCN_Graph
import torch 
import torch.nn.functional as F
import json
import os

def load_hyperparameters_gnn(dataset_name):
    try:
        with open(os.path.join('experiment_utils','hyperparameters_gnn.json'), 'r') as file:
            hyperparameters_data = json.load(file)
            return hyperparameters_data.get(dataset_name, {})
    except FileNotFoundError:
        print("Hyperparameters file not found.")
        return {}

class Experiment():
    def __init__(self,device,datasetname,dataset,data,hyperparameters):

        #self.model = GCN(dataset.num_features,dataset.num_classes,hidden_channels=64 )
        
        self.data = data

        self.lr = hyperparameters["learning_rate"]
        self.layers = hyperparameters["layers"]
        self.weight_decay = hyperparameters["weight_decay"]
        self.dropout = hyperparameters["dropout"]

        self.model = GCN_Toppingetal(dataset,self.layers,self.dropout).to(device = device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay= self.weight_decay)
    
        self.epoch = 10000#hyperparameters["epochs"]
    def train(self):
        self.model.train()
        self.optimizer.zero_grad()  # Clear gradients.
        
        out = self.model(self.data,self.device)  # Perform a single forward pass.
        loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask].to(self.device))  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        self.optimizer.step()  # Update parameters based on gradients.
        return loss
        
    def validate(self):
        self.model.eval()
        out = self.model(self.data,self.device)  # Perform a single forward pass.
        pred = out[self.data.val_mask].max(1)[1]
        val_acc = pred.eq(self.data.y[self.data.val_mask]).sum().item() / self.data.val_mask.sum().item()
        
        return val_acc
    def test(self):
        self.model.eval()
        out = self.model(self.data,self.device)  # Perform a single forward pass.
        pred = out[self.data.test_mask].max(1)[1]
        test_acc = pred.eq(self.data.y[self.data.test_mask]).sum().item() / self.data.test_mask.sum().item()
        
        return test_acc
        
    def training(self):
        losses = []
        validations = []
        counter = 0
        for epoch in range(1, self.epoch):
            loss = self.train()
            losses.append(loss.detach().cpu().numpy())
            val = self.validate()
            validations.append(val)
            if epoch ==1:
                best_val = val
            elif epoch > 1 and val >= best_val:
                best_val = val
                counter = 0
            else:
                counter += 1

            if counter > 100:
                #print("Early stopping at Epoch: ", epoch)
                break
        return losses,validations

class GraphExperiment():
    def __init__(self,device,dataset,hyperparameters):

        self.device = device
        
        self.lr = hyperparameters["learning_rate"]
        self.layers = hyperparameters["layers"]
        self.weight_decay = hyperparameters["weight_decay"]
        self.dropout = hyperparameters["dropout"]

        self.model = GCN_Graph(dataset, self.layers,self.dropout).to(device = device)

        self.loss_fn =  torch.nn.NLLLoss()
        self.softmax =  torch.nn.LogSoftmax(dim=1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay= self.weight_decay)

        self.epoch = 10000#hyperparameters["epochs"]
    def train(self,train_loader):

        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()  # Clear gradients.

        for graph in train_loader:
            graph = graph.to("cuda")
            y = graph.y.to("cuda")

            out = self.model(graph)
            loss = self.loss_fn(self.softmax(out), y)
            total_loss += loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return total_loss

    def eval(self, loader):
        """"
        Depending on the loader this can be either the validation or the test set
        """
        self.model.eval()
        sample_size = len(loader.dataset)
        with torch.no_grad():
            total_correct = 0
            for graph in loader:
                graph = graph.to(self.device)
                y = graph.y.to(self.device)
                out = self.model(graph)
                _, pred = out.max(dim=1)
                total_correct += pred.eq(y).sum().item()
                
        return total_correct / sample_size