from torch import nn

class TextClassificationModel(nn.Module):
    '''
    A module representing the Neural Network Architecture for Text Classification
    '''

    def __init__(self, vocab_size, embed_dim, num_class):
        '''
        Initialization of TextClassificationModel module

        Args:
            vocab_size(int): The size of the vocabulary
            embed_dim(int): The dimensionality of the embedding space
            num_class(int): The number of classes
        '''
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        '''
        Initializes the weights of the embedding and fully connected layers.
        '''
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        '''
        Performs the forward pass of the model. 

        Args:
            text (torch.Tensor): A tensor containing the text.
            offsets (torch.Tensor): A tensor containing the starting indices of each sequence.

        '''
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)