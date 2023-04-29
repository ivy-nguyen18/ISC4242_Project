from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class DataPreprocessingPipeline:
    '''
    A class that preprocesses data for text classification tasks.
    '''

    def __init__(self, data_iter):
        '''
        Initializes the DataPreprocessingPipeline object with the given dataset iterator. Uses the basic english tokenizer
        and calls the get_vocab function to get the text_transformer and label_transformer, and the vocab instance.

        Args:
            data_iter: A dataset iterator
        '''
        self.data_iter = data_iter
        self.tokenizer = get_tokenizer('basic_english')
        self.text_transform, self.label_transform, self.vocab = self.get_vocab()

    def yield_tokens(self, data_iter):
        '''
        A generator method that tokenizes text data into individual words based on the "basic_english" tokenizer.

        Args:
            data_iter: A dataset iterator
        Returns:
            list of strings: A list of individual words in a text data instance
        '''
        for _, text in data_iter:
            yield self.tokenizer(text)

    def get_vocab(self):
        '''
        Creates a vocabulary object based on the tokenized text data. The vocab object maps tokens to indices. 
        This ensures that each word in the text is represented by a unique index (like a dictionary)
        Any unknown tokens will be considered <unk>
        Returns:
            tuple of (callable, callable, torchtext.vocab.Vocab)
                first callable: A function that transforms text into a tensor of numerical values based on the vocabulary
                second callable: A function that transforms the label into a numerical value.
                vocab: A vocabulary object that maps words to numerical indices.
        '''
        vocab = build_vocab_from_iterator(self.yield_tokens(self.data_iter), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        return lambda x: vocab(self.tokenizer(x)), lambda x: int(x) - 1, vocab