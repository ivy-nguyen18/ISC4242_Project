from torch import nn

class TextClassificationNetwork(nn.Module):
    '''
    A module representing the Neural Network Architecture for Text Classification
    '''

    def __init__(self, vocab_size, embed_dim, num_class):
        '''
        Initialization of TextClassificationNetwork module.
        The layers are defined as follows:
        - EmbeddingBag: Initialized with the vocab size (the number of unique words in the dataset) and the embed_dim (the number of attributes used to represent each word - similar to complexity)
            - Instead of treating words independently, we want to use Embeddings to encode semantic similarity in words
            - Given a tensor of word indices (after the preprocessing step):
                - Looks up the embedding for each word index in the input tensor using a weight matrix (vocab X embed_dim), that is initially random before training
                - The weight matrix is like a look up table where each row is a word and each column is some attribute. It contains vectors of weights that can be used to represent each word in the vocab.
                - The embedded vectors of each word in the input vector is reduced by taking the mean (default reduction mode)
                - Outputs a tensor that represents the overall embedding of the input text
        - Linear: a linear layer which predicts what category the text belongs to
        - init_weights: used to randomly initialize weights for the parameters of the model, so the model can learn
        Args:
            vocab_size: The size of the vocabulary
            embed_dim: The dimensionality of the embedding space
            num_class: The number of classes
        '''
        super(TextClassificationNetwork, self).__init__()
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
        embeddings of each index in a learned embedding table, this essentially converts the words to vectors. Then using self.fc (an instance of the nn.Linear), do
        a linear transformation of the embedded representation to get a score for each class.

        Args:
            text: A tensor containing the text.
            offsets: A tensor containing the starting indices of each sequence.

        '''
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)