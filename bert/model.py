"""Bert模型
    模型实现参考：https://github.com/dreamgonfly/BERT-pytorch
"""
import torch
from torch import nn
import math
import numpy as np


class PositionalEmbedding(nn.Module):
    
    def __init__(self, max_len, hidden_size):
        super(PositionalEmbedding, self).__init__()
        self.positional_embedding = nn.Embedding(max_len, hidden_size)
        self.positions = torch.arange(0, max_len)
        self.register_buffer('positions', self.positions)

    def forward(self, sequence):
        batch_size, seq_len = sequence.size()
        positions = self.positions[:seq_len].unsqueeze(0).repeat(batch_size, 1)
        return self.positional_embedding(positions)


class SegmentEmbedding(nn.Module):

    def __init__(self, hidden_size):
        super(SegmentEmbedding, self).__init__()
        self.segment_embedding = nn.Embedding(2, hidden_size)

    def forward(self, segments):
        """segments: (batch_size, seq_len)"""
        return self.segment_embedding(segments)  # (batch_size, seq_len, hidden_size)


class BertEncoder(nn.Module):
    def __init__(self, layers_count, d_model, heads_count, d_ff, dropout_prob):
        super(BertEncoder, self).__init__()

        self.d_model = d_model
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, heads_count, d_ff, dropout_prob) for _ in range(layers_count)]
        )

    def forward(self, sources, mask):
        """ Args:
           sources: embedded_sequence, (batch_size, seq_len, embed_size)
        """
        for encoder_layer in self.encoder_layers:
            sources = encoder_layer(sources, mask)
        return sources


class MultiHeadAttention(nn.Module):
    
    def __init__(self, heads_count, d_model, dropout_prob, mode='self-attention'):
        super(MultiHeadAttention, self).__init__()

        assert d_model % heads_count == 0
        assert mode in ('self-attention', 'memory-attention')

        self.d_head = d_model // heads_count
        self.heads_count = heads_count
        self.mode = mode
        self.query_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.key_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.value_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.final_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=3)

        self.attention = None
        # For cache
        self.key_projected = None
        self.value_projected = None

    def forward(self, query, key, value):
        """ Args:
            query: (batch_size, query_len, model_dim)
            key: (batch_size, key_len, model_dim)
            value: (batch_size, value_len, model_dim)
        """
        batch_size, query_len, d_model = query.size()
        d_head = d_model // self.heads_count

        key_projected = self.key_projection(key)
        value_projected = self.value_projection(value)

        query_projected = self.query_projection(query)
        self.key_projected = key_projected
        self.value_projected = value_projected

        batch_size, key_len, d_model = key_projected.size()
        batch_size, value_len, d_model = value_projected.size()

        query_heads = query_projected.view(batch_size, query_len, self.heads_count, d_head).transpose(1, 2)  # (batch_size, heads_count, query_len, d_head)
        key_heads = key_projected.view(batch_size, key_len, self.heads_count, d_head).transpose(1, 2)  # (batch_size, heads_count, key_len, d_head)
        value_heads = value_projected.view(batch_size, value_len, self.heads_count, d_head).transpose(1, 2)  # (batch_size, heads_count, value_len, d_head)

        attention_weights = self.scaled_dot_product(query_heads, key_heads)  # (batch_size, heads_count, query_len, key_len)

        self.attention = self.softmax(attention_weights)  # Save attention to the object
        attention_dropped = self.dropout(self.attention)
        context_heads = torch.matmul(attention_dropped, value_heads)  # (batch_size, heads_count, query_len, d_head)
        context_sequence = context_heads.transpose(1, 2).contiguous()  # (batch_size, query_len, heads_count, d_head)
        context = context_sequence.view(batch_size, query_len, d_model)  # (batch_size, query_len, d_model)
        return self.final_projection(context)

    def scaled_dot_product(self, query_heads, key_heads):
        """ Args:
             query_heads: (batch_size, heads_count, query_len, d_head)
             key_heads: (batch_size, heads_count, key_len, d_head)
        """
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = torch.matmul(query_heads, key_heads_transposed)  # (batch_size, heads_count, query_len, key_len)
        attention_weights = dot_product / np.sqrt(self.d_head)
        return attention_weights


class PointwiseFeedForwardNetwork(nn.Module):   
    def __init__(self, d_ff, d_model, dropout_prob):
        super(PointwiseFeedForwardNetwork, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout_prob),
            GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        """ Args:
             x: (batch_size, seq_len, d_model)
        """
        return self.feed_forward(x)


class GELU():
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Sublayer(nn.Module):
    
    def __init__(self, sublayer, d_model):
        super(Sublayer, self).__init__()

        self.sublayer = sublayer
        self.layer_normalization = LayerNormalization(d_model)

    def forward(self, *args):
        x = args[0]
        x = self.sublayer(*args) + x
        return self.layer_normalization(x)


class LayerNormalization(nn.Module):
    
    def __init__(self, features_count, epsilon=1e-6):
        super(LayerNormalization, self).__init__()
        self.gain = nn.Parameter(torch.ones(features_count))
        self.bias = nn.Parameter(torch.zeros(features_count))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gain * (x - mean) / (std + self.epsilon) + self.bias


class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, heads_count, d_ff, dropout_prob):
        super(EncoderLayer, self).__init__()

        self.self_attention_layer = Sublayer(MultiHeadAttention(heads_count, d_model, dropout_prob), d_model)
        self.pointwise_feedforward_layer = Sublayer(PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob), d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, sources, sources_mask):
        sources = self.self_attention_layer(sources, sources, sources, sources_mask)
        sources = self.dropout(sources)
        sources = self.pointwise_feedforward_layer(sources)
        return sources


class BERT(nn.Modlue):
    def __init__(self, config, token_embedding, positional_embedding, encoder, segment_embedding, hidden_size):
        super(BERT, self).__init__()

        self.config = config
        self.token_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=config.hidden_size)
        self.positional_embedding = PositionalEmbedding(config.max_len, config.hidden_size)
        self.segment_embedding = SegmentEmbedding(config.hidden_size)
        self.encoder = BertEncoder(config.layers_count, config.hidden_size, config.heads_count, config.dropout_prob)  
    
    def forward(self, inputs):
        sequence, segment = inputs
        token_embedded = self.token_embedding(sequence)
        positional_embedded = self.positional_embedding(sequence)
        segment_embedded = self.segment_embedding(segment)
        embedded_sources = token_embedded + positional_embedded + segment_embedded

        mask = self.padding_mask(sequence)
        encoded_sources = self.encoder(embedded_sources, mask)
        return encoded_sources
    
    def pad_masking(x):
        # x: (batch_size, seq_len)
        padded_positions = x == 0
        return padded_positions.unsqueeze(1)