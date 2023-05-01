import torch
from torchtext.datasets import AG_NEWS
from dl.data_preprocessing import DataPreprocessingPipeline
from dl.network import TextClassificationNetwork
from dl.model import Model
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import matplotlib.pyplot as plt
import time
import pandas as pd
from dl.parameters import standard_params, param1, param2, param3, param4, param5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    '''
    Given a batch of pairs of text data and label, each pair is processed as follows:
    - converts labels to a numeric representation using DataPreprocessingPipeline's label_transform
    - convert text data to a tensor of integers using DataPreprocessingPipeline's text_transform which tokenizes the text
    - keeps track of offsets, which tracks the length of each tensor in the batch. Offsets are needed in the neural network because text data has variable length, Offsets are used in place of padding.  
    And since we concatenated all the text data, we need to somehow keep track of when a piece of text ends and starts.
    Then, everything is converted to tensors
    - labels are converted to int64 tensors
    - take everything but the last offset and then take cummulative sum. This would return where each text tensor starts. Then convert it to a tensor.
    - concatenate text_list to a single tensor

    Args:
        batch: pairs of text data and labels
    Returns:
        tuple:
            label_list: tensor of labels
            text_list: tensor of tokenized texts
            offsets: tensor of offsets
    '''
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(dataPipeline.label_transform(_label))
         processed_text = torch.tensor(dataPipeline.text_transform(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

def predict(text, text_pipeline):
    '''
    Predicting the text. Since we are not training the model again, we don't use the gradients. 
    We put the text through the text processing pipeline. And then predict on the text by putting it 
    through the network. Then, since we are returned the softmax of the model, we want to get the index
    with the maximum value. We add 1 to the index because python starts at 0, this should return the
    category.
    
    Args:
        text: text data we want to predict on
        text_pipeline: The pipeline that preprocesses the text and transforms into tokens
    Returns:
        int: predicted category {1,2,3,4}
    '''
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = network(text, torch.tensor([0]))
        return output.argmax(1).item() + 1
        
def predict_model(network,ex_text_str):
    '''
    The function that takes the example text and network, and calls the prediction function to return the predicted label.

    Args:
        network: the trained network
        ex_text_str: the text we want to predict on
    Returns:
        int: the predicted label
    '''
    ag_news_label = {1: "World",
                2: "Sports",
                3: "Business",
                4: "Sci/Tec"}
    
    network = network.to("cpu")
    return ag_news_label[predict(ex_text_str, dataPipeline.text_transform)]

def train_model(train_dataloader,valid_dataloader, scheduler, model, standard_flag):
    '''
    This function trains the dataset and evaluates the dataset on a validation dataset over
    multiple epochs. It also keeps track of how long it takes to execute each batch.
    For each epoch:
        - Train the model
        - Evaluate the model using validation set
        - If the validation accuracy is worse than the previous best accuracy, adjust the scheduler, else the validation accuracy is made the best accuracy. 
        - print out epoch number, total time it took, and the validation accuracy
    
    Args: 
        train_dataloader: Pytorch Dataloader that is the training data
        valid_dataloader: Pytorch Dataloader that is the validation data
        scheduler: used to adjust the learning rate so it is dynamically changing to get better accuracy
        model: neural network model to be trained
    Returns:
        None
        
    '''
    total_accu = None
    train_accu = []
    valid_accu = []
    total_time = 0
    for epoch in range(1, params['epochs'] + 1):
        epoch_start_time = time.time()
        accu_train = model.train(train_dataloader, epoch)
        train_accu.append(accu_train)
        accu_val = model.evaluate(valid_dataloader)
        valid_accu.append(accu_val)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        epoch_time = time.time() - epoch_start_time
        print('| end of epoch {:3d} | time: {:5.2f}s | '
            'valid accuracy {:8.3f} '.format(epoch, epoch_time, accu_val))
        print('-' * 59)
        total_time += epoch_time

    # Plot graph for training and validation accuracy (only for standard parameters)
    if standard_flag == 0:
        plot_graph('train_val', train_accu, valid_accu)
        standard_flag = 1

    return total_time/epoch_time, standard_flag

def plot_graph(type, data1, data2 = None, param_list = None):
    '''
    Additional function to visualize various parameter combinations and ranges as well as plot the train-validation data. 
    To keep it simple, only using embed_dim, but applicable to other params as well. 

    Args:
        type: what type of graph we want to display
        data1: first dataset
        data2: Default None, second dataset
        param_list: Default None, list of parameters to evaluate
    '''
    if type == 'train_val':
        x = list(range(params['epochs']))
        plt.plot(x, data1, label='Training data')
        plt.plot(x, data2, label='Validation data')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Train vs. Validation Accuracy')
        plt.show()
    elif type == 'embed':
        x = []
        for param in param_list:
            x.append(param['emsize'])
        sorted_lists = sorted(zip(x, data1, data2))
        x, data1, data2 = zip(*sorted_lists)
        x, data1 = zip(*sorted(zip(x, data1)))
        plt.plot(x, data1)
        plt.xlabel("Embedded Dimension size")
        plt.ylabel("Test Accuracy")
        plt.title('Comparing Embedded Dimension Sizes')
        plt.show()

        plt.plot(x, data2)
        plt.xlabel("Embedded Dimension size")
        plt.ylabel("Avg Time")
        plt.title('Comparing Embedded Dimension Sizes')
        plt.show()


if __name__ == '__main__':
    standard_flag = 0
    params_list = [standard_params, param1, param2, param3, param4, param5]
    test_acc_list = []
    epoch_time = []
    for params in params_list:
        print(params)
        #Preparing data to be sent into the neural network
        train_iter = AG_NEWS(split='train')
        dataPipeline = DataPreprocessingPipeline(train_iter)
        dataloader = DataLoader(train_iter, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_batch)
        num_class = len(set([label for (label, text) in train_iter]))
        vocab_size = len(dataPipeline.vocab)
        network = TextClassificationNetwork(vocab_size, params['emsize'], num_class).to(device)

        #Parameters to be used in the model
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(network.parameters(), params['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, params['step_size'], params['gamma'])
        model = Model(network, optimizer, criterion)

        #Getting Train-Test-Validation dataloaders
        train_iter, test_iter = AG_NEWS()
        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)
        num_train = int(len(train_dataset) * params['percent_train'])
        split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

        train_dataloader = DataLoader(split_train_, batch_size=params['batch_size'],
                                    shuffle=True, collate_fn=collate_batch)
        valid_dataloader = DataLoader(split_valid_, batch_size=params['batch_size'],
                                    shuffle=True, collate_fn=collate_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'],
                                    shuffle=True, collate_fn=collate_batch)

        #Training the model
        time_epoch, standard_flag = train_model(train_dataloader, valid_dataloader, scheduler, model, standard_flag)
        epoch_time.append(time_epoch)

        #Testing the model
        print('Checking the results of test dataset.')
        accu_test = model.evaluate(test_dataloader)
        test_acc_list.append(accu_test)
        print('test accuracy {:8.3f}'.format(accu_test))

        #Predicting using the model    
        print('\nPredicting on new data...')
        df = pd.read_csv("dl/sample_text.txt", sep="|")
        for index, row in df.iterrows():
            print(f"Expected: {predict_model(network, row['text'])}, Actual: {row['label']}")

    plot_graph(type = 'embed', data1 = test_acc_list,data2 = epoch_time, param_list = params_list)
   




