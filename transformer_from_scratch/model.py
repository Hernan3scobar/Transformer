import torch
import torch.nn as nn
import math

#### Conversion of Words to Numbers:

# Embedding Techniques: Each word in the input sequence is mapped to a vector of real numbers
# using embedding techniques like Word2Vec, GloVe, or learned embeddings within the transformer itself.






# Positional Embeddings: In addition to the word embeddings, positional embeddings are added to provide 
# information about the position of each word in the sequence. This is crucial for capturing the order 
# of words since transformers process the entire sequence simultaneously and do not inherently understand the order.

# Vector Representation:

# The resulting numerical vectors (word embeddings combined with positional embeddings) are
# then fed into the transformer model. These vectors serve as the input that the transformer 
# uses for all subsequent processing steps, including attention mechanisms and layer transformations.
# This process of converting words into numerical representations allows the transformer to handle and
#  process sequences of text effectively, enabling it to perform tasks such as translation, 
# summarization, and more.




class ImputEmbedding(nn.Module):

    def __init__(self,d_model:int, vocab_size:int):
    # Calls the constructor of the parent class (nn.Module)

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Creates an embedding layer that maps vocabulary 
        # indices to dense vectors and these vectors
        # are learned by the model during training.
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        # Applies the embedding layer to the input and scales the embeddings
        return self.embedding(x)*math.sqrt(self.d_model)
    
## Positional Encoding

## Remember that the original sentence is mapped to a list of vectors by the 
## embedding layers. We want to encode the position of every word in the 
## sentence, and this is done by adding another vector of the same size as the 
## embedding that tells the model the position occupied by each word.


# This can be done in a clever way, in the paper "Attention is all you need", they used sinusoidal
# function, this has great advantages, in the very dinamical way. it step or every position can be cover
# in a simple discretized way function and partionized for bigs numbers

class PositionalEncoding(nn.Module):

    def __init__(self,d_model:int, seq_len:int,dropout:float)->None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout= nn.Dropout(dropout)
        
        
        
        # A matrix of zeros is created with a shape of 
        # (seq_len, d_model) to store positional encodings.

        pe = torch.zeros(seq_len, d_model)
        
        ## Create a vector of shape (seq_len,1)
           
        # A position vector is created with values 
        # ranging from 0 to seq_len - 1. 
        # This provides a unique positional representation for each position in the sequence.

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Division terms are calculated to be used in 
        # sinusoidal and cosinusoidal functions for generating positional encodings. as the paper says.
   
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        #Notice that in the paper, the real div_term is a power of 10^4, the log representation, i.e. the log domain is less 
        #dense, this has computer adavantages, the result is the same so is recomended to used.
        
        # Apply the sin to even position
       
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply the cos  ot odd   position

        pe[:, 1::2] = torch.cos(position * div_term)
        
        # We need the batch dimention of the tensor,
        #  An extra dimension is added to the tensor to represent 
        # the batch dimension. This ensures that positional encodings 
        # can be added consistently to each batch of input data.
        pe = pe.unsqueeze(0) # Tehsor of the size (1,seq_len,dim_len)

        #e positional encoding is registered as a 
        # buffer so that it's treated as a model parameter but
        #  doesn't require gradients during training.

        self.register_buffer('pe', pe)

    def forward(self, x):
        # Adds the positional encodings to the input embeddings

        #In the forward method, positional encodings are added to the
        #  input embeddings to provide information about the position
        #  of each element in the sequence. self.pe[:x.size(1), :] selects
        #  positional encodings corresponding to the length of the input
        #  sequence x. Additionally, requires_grad_(False) is used to ensure
        #  that positional encodings are not trainable (don't require gradients).Because
        #  are fixed.
        #  Finally, the modified embeddings are returned after applying a dropout layer.

        x = x + (self.pe[:x.size(1), :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        # Small constant to avoid division by zero during normalization
        self.eps = eps
        # Learnable parameter for scaling (initialized to 1)
        self.alpha = nn.Parameter(torch.ones(1))  # Scaling factor (gamma)
        # Learnable parameter for shifting (initialized to 0)
        self.bias = nn.Parameter(torch.zeros(1))  # Bias term (beta)

    def forward(self, x):
        # Calculate the mean of the input tensor along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        # Calculate the standard deviation of the input tensor along the last dimension
        std = x.std(dim=-1, keepdim=True)
        # Normalize the input tensor, then scale and shift it using alpha and bias
        return self.alpha * (x - mean) / (std + self.eps) + self.bias