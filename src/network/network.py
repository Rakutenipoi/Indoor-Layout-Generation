import torch
import torch.nn as nn

from .boundary_encoder import get_boundary_encoder
from .embedding import Embedding, RelativePositionEncoding, ObjectIndexEncoding, AbsolutePositionEncoding
from .sampling import Sampler
from .transformer_modules import EncoderLayer, DecoderLayer

# GPU
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)

class cofs_network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config['network']['n_layers']
        self.n_heads = config['network']['n_heads']
        self.dimensions = config['network']['dimensions']
        self.d_ff = config['network']['feed_forward_dimensions']
        self.dropout = config['training']['dropout']
        self.max_sequence_length = config['network']['max_sequence_length']
        self.attributes_num = config['data']['attributes_num']
        self.object_max_num = config['data']['object_max_num']
        self.class_num = config['data']['class_num'] + 1 # 需要预留一个额外的类别用于标记起始与结束
        self.batch_size = config['training']['batch_size']

        # 用max_sequence_length替代sequence_length
        self.embedding = Embedding(
            class_num=self.class_num,
            sequence_length=self.max_sequence_length,
            encode_levels=self.dimensions / 2,
            E_dims=self.dimensions
        )

        self.relative_position_encoding = RelativePositionEncoding(
            attributes_num=self.attributes_num,
            E_dims=self.dimensions
        )
        self.object_index_encoding = ObjectIndexEncoding(
            object_max_num=self.object_max_num,
            attributes_num=self.attributes_num,
            E_dims=self.dimensions
        )
        self.absolute_position_encoding = AbsolutePositionEncoding(
            object_num=self.object_max_num,
            attributes_num=self.attributes_num,
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
            config['boundary_encoder']['feature_size'],
        )

        self.decoder_to_output_attr = nn.Linear((self.max_sequence_length + 1) * self.dimensions, (self.attributes_num - 1) * self.dimensions)
        self.decoder_to_output_type = nn.Linear((self.max_sequence_length + 1) * self.dimensions, 1 * self.dimensions)

        self.sampler = Sampler(self.dimensions, self.class_num, config['network']['sampler'])

    def forward(self, sequence, layout_image, last_sequence):
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
        sequence = sequence + relative_position + object_index
        sequence = torch.concat((layout_feature, sequence), dim=1)
        sequence = torch.concat((torch.zeros((self.batch_size, 1, self.dimensions)).to(torch.device(device)), sequence), dim=1)
        eos = torch.zeros((self.batch_size, 1, self.dimensions)).to(torch.device(device))
        eos[:, :, 0] = 22.0
        sequence = torch.concat((sequence, eos), dim=1)
        last_sequence = last_sequence + absolute_position
        last_sequence = torch.concat((torch.zeros((self.batch_size, 1, self.dimensions)).to(torch.device(device)), last_sequence), dim=1)

        # Encoders Process
        encoder_output = sequence
        for encoder_layer in self.condition_encoder:
            encoder_output = encoder_layer(encoder_output, None)

        # Decoders Process
        decoder_output = last_sequence
        tgt_mask = torch.tril(torch.ones((self.max_sequence_length + 1, self.max_sequence_length + 1))).to(torch.device(device))
        for decoder_layer in self.generative_decoder:
            decoder_output = decoder_layer(decoder_output, encoder_output, None, tgt_mask)

        # 此时的decoder_output的尺寸为[batch_size, sequence_size + 1, dimension_size]
        ## 我们需要将这个尺寸转化为[batch_size, attr_size, dimension_size]
        ## 将输入张量形状转换为[batch_size, (sequence_size + 1) * dimension_size]
        decoder_output = decoder_output.view(self.batch_size, -1)
        decoder_output_type = self.decoder_to_output_type(decoder_output)
        decoder_output_attr = self.decoder_to_output_attr(decoder_output)
        ## 将输出张量形状转换为[batch_size, attr_size, dimension_size]
        decoder_output_type = decoder_output_type.view(self.batch_size, -1, self.dimensions)
        decoder_output_attr = decoder_output_attr.view(self.batch_size, -1, self.dimensions)

        # Output Sample
        output_type = self.sampler(decoder_output_type, True)
        output_attr = self.sampler(decoder_output_attr, False)

        return output_type, output_attr





























