import torch
from torchtext.datasets import AG_NEWS
from dl.data_preprocessing import DataPreprocessingPipeline
from dl.network import TextClassificationModel
from dl.model import Model
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import time
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    'emsize' : 64,
    'epochs': 10,
    'lr': 5,
    'batch_size': 64,
    'percent_train': 0.95,
    'step_size': 1.0,
    'gamma': 0.1
,}

def collate_batch(batch):
    '''
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
    '''
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = network(text, torch.tensor([0]))
        return output.argmax(1).item() + 1
        
def predict_model(network,ex_text_str):
    '''
    '''
    ag_news_label = {1: "World",
                2: "Sports",
                3: "Business",
                4: "Sci/Tec"}
    
    network = network.to("cpu")
    return ag_news_label[predict(ex_text_str, dataPipeline.text_transform)]

def train_model(train_dataloader,valid_dataloader, scheduler, model):
    '''
    
    '''
    total_accu = None
    for epoch in range(1, params['epochs'] + 1):
        epoch_start_time = time.time()
        model.train(train_dataloader, epoch)
        accu_val = model.evaluate(valid_dataloader)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
            'valid accuracy {:8.3f} '.format(epoch, time.time() - epoch_start_time, accu_val))
        print('-' * 59)
     
if __name__ == '__main__':
    
    #Preparing data to be sent into the neural network
    train_iter = AG_NEWS(split='train')
    dataPipeline = DataPreprocessingPipeline(train_iter)
    dataloader = DataLoader(train_iter, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_batch)
    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(dataPipeline.vocab)
    network = TextClassificationModel(vocab_size, params['emsize'], num_class).to(device)

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
    train_model(train_dataloader, valid_dataloader, scheduler, model)
    
    #Testing the model
    print('Checking the results of test dataset.')
    accu_test = model.evaluate(test_dataloader)
    print('test accuracy {:8.3f}'.format(accu_test))

    #Predicting using the model    
    print('\nPredicting on new data...')
    df = pd.read_csv("dl/sample_text.txt", sep="|")
    for index, row in df.iterrows():
        print(f"Expected: {predict_model(network, row['text'])}, Actual: {row['label']}")







