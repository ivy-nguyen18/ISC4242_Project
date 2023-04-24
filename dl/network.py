from torch import nn

class TextClassificationModel(nn.Module):
    '''
    A module representing the Neural Network Architecture for Text Classification
    '''

    def __init__(self, vocab_size, embed_dim, num_class):
        '''
        Initialization of TextClassificationModel module.
        The layers are defined as follows:
        - EmbeddingBag: break down the word tokens to numbers which are called word embeddings. EmbeddingBag combines all word embeddings into a bag of embeddings, sums it up and then takes the average.
        - Linear: a linear layer which predicts what category the text belongs to
        - init_weights: used to "randomly" initialize weights for the parameters of the model, so the model can "learn"
        Args:
            vocab_size: The size of the vocabulary
            embed_dim: The dimensionality of the embedding space
            num_class: The number of classes
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
        Performs the forward pass of the model. Using nn.EmbeddingBag, we convert the input word indices to dense vectors by looking up the
        embeddings of each index in a learned embedding table, this essentially converts the words to numbers. Then using self.fc (an instance od the nn.Linear), which does
        a linear transformation of the embedded representation to get a score for each class.

        Args:
            text: A tensor containing the text.
            offsets: A tensor containing the starting indices of each sequence.

        '''
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)