from embedding import Embedding, RelativePositionEncoding, ObjectIndexEncoding, AbsolutePositionEncoding
import torch
import torch.nn as nn
import math
from transformer_modules import EncoderLayer, DecoderLayer
from boundary_encoder import get_boundary_encoder
from sampling import Sampler

class cofs_network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config['network']['n_layers']
        self.n_heads = config['network']['n_heads']
        self.dimensions = config['network']['dimensions']
        self.d_ff = config['network']['feed_forward_dimensions']
        self.droupout = config['training']['droupout']
        self.max_sequence_length = config['network']['max_sequence_length']
        self.attributes_num = config['data']['attributes_num']
        self.object_max_num = config['data']['object_max_num']
        self.class_num = config['data']['class_num'] + 1 # 需要预留一个额外的类别用于标记起始与结束

        # 用max_sequence_length替代sequence_length
        self.embedding = Embedding(
            class_num=self.class_num,
            sequence_length=self.max_sequence_length,
            encode_levels=self.dimensions / 2,
            E_dims=self.dimensions
        )

        self.relative_position_encoding = RelativePositionEncoding(
            max_sequence_length=self.max_sequence_length,
            attributes_num=self.attributes_num,
            E_dims=self.dimensions
        )
        self.object_index_encoding = ObjectIndexEncoding(
            max_sequence_length=self.max_sequence_length,
            object_max_num=self.object_max_num,
            attributes_num=self.attributes_num,
            E_dims=self.dimensions
        )
        self.absolute_position_encoding = AbsolutePositionEncoding(
            max_sequence_length=self.max_sequence_length,
            object_num=self.object_max_num,
            E_dims=self.dimensions
        )

        self.condition_encoder = nn.ModuleList(
            [EncoderLayer(self.dimensions, self.n_heads, self.d_ff, self.dropout) for _ in range(self.n_layers)]
        )
        self.generative_decoder = nn.ModuleList(
            [DecoderLayer(self.dimensions, self.n_heads, self.d_ff, self.dropout) for _ in range(self.n_layers)]
        )
        self.boundary_encoder = get_boundary_encoder(
            config['boundary_encoder']['name'],
            config['boundary_encoder']['freeze_bn'],
            config['boundary_encoder']['input_channels'],
            config['boundary_encoder']['feature_size']
        )

        self.sampler = Sampler(self.dimensions, self.class_num, config['network']['sampler'])

    def forward(self, sequence, layout_image, last_sequence):
        # Boundary Encoder
        layout_feature = self.boundary_encoder(layout_image)

        # Positional Encoding
        relative_position = self.relative_position_encoding(sequence)
        object_index = self.object_index_encoding(sequence)
        absolute_position = self.absolute_position_encoding(last_sequence)

        # Embedding
        sequence = self.embedding(sequence)
        last_sequence = self.embedding(last_sequence)

        # Input Blend & Concatenate
        sequence = sequence + relative_position + object_index
        sequence = torch.concat((layout_feature, sequence), dim=0)
        last_sequence = last_sequence + absolute_position
        last_sequence = torch.concat((torch.zeros((1, self.dimensions)), last_sequence), dim=0)

        # Encoders Process
        encoder_output = sequence
        for encoder_layer in self.condition_encoder:
            encoder_output = encoder_layer(encoder_output, None)

        # Decoders Process
        decoder_output = last_sequence
        for decoder_layer in self.generative_decoder:
            decoder_output = decoder_layer(decoder_output, encoder_output, None, None)

        # Output Sample
        output = self.sampler(decoder_output)

        return output
