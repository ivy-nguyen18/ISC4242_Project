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
            network: the neural network architecture
            optimizer: the optimizer used for training the model
            loss_fn: the loss function used for training the model 
        '''
        
        self.network = network
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, dataloader, epoch):
        '''
        Train model on given data for a specified number of epochs. 
        For each label, text, and offsets:
        - set the gradients of the networks parameters to 0
        - send the text and offsets to the network to get a predicted label
        - get the loss by comparing the actual and predicted dataset
        - perform a backpropagation to update the embedded weights based on the loss
        - perform gradient clipping (clip_grad_norm) to avoid exploding gradients
        - use the optimizer to update the network parameters
        - calculate the total accuracy by comparing the predicted labels with the true labels over the total samples
        - for each log interval (controls how often progress gets printed), we print the epoch number, the number of batches, and the accuracy

        Args:
            dataloader: A PyTorch DataLoader object containing the training data.
            epoch: The number of epochs to train the model.

        Returns:
            None
        '''
        
        self.network.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()
        train_acc = 0

        for idx, (label, text, offsets) in enumerate(dataloader):
            self.optimizer.zero_grad()
            predicted_label = self.network(text, offsets)
            loss = self.loss_fn(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
            self.optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            idx +=1
            if (idx % log_interval == 0 or idx%len(dataloader) == 0) and (idx+1 > 0):
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                    '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                                total_acc/total_count))
                if idx%len(dataloader) == 0:
                    train_acc = total_acc/total_count
                
                total_acc, total_count = 0, 0
                start_time = time.time()

        return train_acc

    def evaluate(self, dataloader):
        '''
        Evaluate the model on the given data, thus the gradients are not needed. For each label, text, and offset, they are sent to the network
        to evaluate the output's loss, using the given loss function. We calculate loss accuracy by comparing the predicted labels with the actual label,
        which should output the number of correct predictions. We return the number of correct predictions over the total number of samples in the dataset.

        Args:
            dataloader: A PyTorch DataLoader object containing the evaluation data.

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