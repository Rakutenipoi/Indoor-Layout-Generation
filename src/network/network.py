import torch
import torch.nn as nn
import torchvision.models as models
import time

from network.boundary_encoder import get_boundary_encoder
from network.embedding import Embedding, RelativePositionEncoding, ObjectIndexEncoding, AbsolutePositionEncoding
from network.sampling import Sampler

# GPU
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)

class cofs_network(nn.Module):
    def __init__(self, config):
        super().__init__()
        # config
        ## network
        self.n_enc_layers = config['network']['n_enc_layers']
        self.n_dec_layers = config['network']['n_dec_layers']
        self.n_heads = config['network']['n_heads']
        self.dimensions = config['network']['dimensions']
        self.d_ff = config['network']['feed_forward_dimensions']
        self.dropout = config['training']['dropout']
        self.activation = config['network']['activation']
        ## data
        self.attributes_num = config['data']['attributes_num']
        self.object_max_num = config['data']['object_max_num']
        self.max_sequence_length = self.attributes_num * self.object_max_num
        self.class_num = config['data']['class_num'] + 2 # 需要预留两个额外的类别用于标记起始与结束
        self.batch_size = config['training']['batch_size']
        self.start_token = config['network']['start_token']
        self.end_token = config['network']['end_token']

        # 用max_sequence_length替代sequence_length
        self.embedding = Embedding(
            class_num=self.class_num,
            sequence_length=self.max_sequence_length,
            encode_levels=self.dimensions / 2,
            E_dims=self.dimensions,
            attribute_num=self.attributes_num,
            dropout=self.dropout
        )

        self.relative_position_encoding = RelativePositionEncoding(
            attributes_num=self.attributes_num,
            E_dims=self.dimensions,
            object_max_num=self.object_max_num
        )
        self.object_index_encoding = ObjectIndexEncoding(
            object_max_num=self.object_max_num,
            attributes_num=self.attributes_num,
            E_dims=self.dimensions
        )
        self.absolute_position_encoding = AbsolutePositionEncoding(
            object_max_num=self.object_max_num,
            attributes_num=self.attributes_num,
            E_dims=self.dimensions
        )

        self.transformer = nn.Transformer(d_model=self.dimensions, nhead=self.n_heads,
                                          num_encoder_layers=self.n_enc_layers,
                                          num_decoder_layers=self.n_dec_layers, dim_feedforward=self.d_ff,
                                          dropout=self.dropout,
                                          batch_first=True, activation=self.activation)

        self.boundary_encoder = get_boundary_encoder(
            config['boundary_encoder']['name'],
            config['boundary_encoder']['freeze_bn'],
            config['boundary_encoder']['input_channels'],
            self.dimensions,
        )

        self.decoder_to_output = nn.Linear((self.max_sequence_length + 1) * self.dimensions, self.max_sequence_length * self.dimensions)

        self.sampler = Sampler(config)

    def get_padding_mask(self, seq_length):
        padding_mask = torch.zeros((self.batch_size, self.max_sequence_length + 1), device=device)
        for i, length in enumerate(seq_length):
            padding_mask[i, length.item():] = -torch.inf

        return padding_mask

    def forward(self, sequence, layout_image, last_sequence, seq_length, last_seq_length):
        # Boundary Encoder
        layout_feature = self.boundary_encoder(layout_image.unsqueeze(1).squeeze(-1))
        layout_feature = layout_feature.unsqueeze(1)

        # Positional Encoding
        relative_position = self.relative_position_encoding(sequence)
        object_index = self.object_index_encoding(sequence)
        absolute_position = self.absolute_position_encoding(last_sequence)

        # Embedding
        sequence = self.embedding(sequence)
        last_sequence = self.embedding(last_sequence)

        # Input Blend & Concatenate
        ## sequence
        sequence = sequence + relative_position + object_index
        sos = self.embedding.get_embedding(torch.tensor([self.start_token], device=device), self.batch_size)
        sequence = torch.concat((layout_feature, sequence), dim=1)
        ## last_sequence
        last_sequence = last_sequence + absolute_position
        last_sequence = torch.concat((sos, last_sequence), dim=1)

        # Mask
        seq_length = seq_length * self.attributes_num
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.max_sequence_length + 1, device=device)
        src_key_padding_mask = self.get_padding_mask(seq_length + 1)
        tgt_key_padding_mask = self.get_padding_mask(last_seq_length * self.attributes_num + 1)

        # Transformer
        decoder_output = self.transformer(sequence, last_sequence, tgt_mask=tgt_mask,
                                          src_key_padding_mask=src_key_padding_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask)

        # 此时的decoder_output的尺寸为[batch_size, sequence_size + 1, dimension_size]
        ## 我们需要将这个尺寸转化为[batch_size, attr_size, dimension_size]
        ## 将输入张量形状转换为[batch_size, (sequence_size + 1) * dimension_size]
        decoder_output = decoder_output.view(self.batch_size, -1)
        decoder_output = self.decoder_to_output(decoder_output)
        ## 将输出张量形状转换为[batch_size, attr_size, dimension_size]
        decoder_output = decoder_output.view(self.batch_size, -1, self.dimensions)

        # Output Sample
        output = self.sampler(decoder_output)

        return output

