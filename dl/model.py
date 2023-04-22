import time
import torch

class Model:
    '''
    A class representing the neural network training and evaluation.
    '''
    
    def __init__(self, network, optimizer, loss_fn):

        '''
        Initializing the Model instance

        Args:
            network (torch.nn.Module): the neural network architecture
            optimizer (torch.optim.Optimizer): the optimizer used for training the model
            loss_fn (torch.nn.CrossEntropyLoss): the loss function used for training the model 
        '''
        
        self.network = network
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, dataloader, epoch):

        '''
        Train model on given data for a specified number of epochs

        Args:
            dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader object containing the training data.
            epoch (int): The number of epochs to train the model.

        Returns:
            None
        '''
        
        self.network.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()

        for idx, (label, text, offsets) in enumerate(dataloader):
            self.optimizer.zero_grad()
            predicted_label = self.network(text, offsets)
            loss = self.loss_fn(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
            self.optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                    '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                                total_acc/total_count))
                total_acc, total_count = 0, 0
                start_time = time.time()

    def evaluate(self, dataloader):

        '''
        Evaluate the model on the given data.

        Args:
            dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader object containing the evaluation data.

        Returns:
            float: The accuracy of the model on the evaluation data.
        '''
        
        self.network.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = self.network(text, offsets)
                loss = self.loss_fn(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc/total_count