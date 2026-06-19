import torch
import torch.nn as nn
import math

class DataPreprocessing:
    """
    A class used to preprocess input and output data for a sequence-to-sequence model.

    Attributes:
    ----------
    d_model : int
        The dimensionality of the model.
    max_len : int
        The maximum length of the input and output sequences.
    input_embedding : nn.Embedding
        The embedding layer for the input sequence.
    output_embedding : nn.Embedding
        The embedding layer for the output sequence.
    """

    def __init__(self, d_model: int, max_len: int, vocab_size: int):
        """
        Initializes the DataPreprocessing class.

        Parameters:
        ----------
        d_model : int
            The dimensionality of the model.
        max_len : int
            The maximum length of the input and output sequences.
        vocab_size : int
            The size of the vocabulary.
        """
        self.d_model = d_model
        self.max_len = max_len
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.output_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.positional_encoding(max_len, d_model)

    def positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Calculates the positional encoding for the input sequence.

        Parameters:
        ----------
        max_len : int
            The maximum length of the input sequence.
        d_model : int
            The dimensionality of the model.

        Returns:
        -------
        torch.Tensor
            The positional encoding tensor.
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def preprocess_input(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses the input sequence by adding positional encoding to the input embeddings.

        Parameters:
        ----------
        input_seq : torch.Tensor
            The input sequence tensor.

        Returns:
        -------
        torch.Tensor
            The preprocessed input sequence tensor.
        """
        input_emb = self.input_embedding(input_seq)
        input_emb = input_emb + self.positional_encoding[:input_seq.size(0), :]
        return input_emb

    def preprocess_output(self, output_seq: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses the output sequence by using the output embeddings.

        Parameters:
        ----------
        output_seq : torch.Tensor
            The output sequence tensor.

        Returns:
        -------
        torch.Tensor
            The preprocessed output sequence tensor.
        """
        output_emb = self.output_embedding(output_seq)
        return output_emb


if __name__ == "__main__":
    d_model = 512
    max_len = 100
    vocab_size = 10000
    data_preprocessing = DataPreprocessing(d_model, max_len, vocab_size)
    input_seq = torch.randint(0, vocab_size, (10,))
    output_seq = torch.randint(0, vocab_size, (10,))
    preprocessed_input = data_preprocessing.preprocess_input(input_seq)
    preprocessed_output = data_preprocessing.preprocess_output(output_seq)
    print(preprocessed_input.shape)
    print(preprocessed_output.shape)