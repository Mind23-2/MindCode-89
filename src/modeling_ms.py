# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import copy

import mindspore
from mindspore import Parameter, Tensor
import mindspore.nn as nn
import mindspore.ops.operations as P
import numpy as np

# from .modeling_resnet import ResNetV2


def swish(x):
    return x * P.Sigmoid()(x)


ACT2FN = {"gelu": nn.GELU(), "relu": nn.ReLU(), "swish": swish}


class Attention(nn.Cell):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer_num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.attention_head_size2 = Tensor(int(config.hidden_size / self.num_attention_heads))
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.out = nn.Dense(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer_attention_dropout_rate)
        self.proj_dropout = nn.Dropout(config.transformer_attention_dropout_rate)

        self.softmax = nn.Softmax(axis=-1)

    def transpose_for_scores(self, x):
        new_x_shape = P.Shape()(x)[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = P.Reshape()(x, new_x_shape)
        return P.Transpose()(x, (0, 2, 1, 3,))

    def transpose2(self, x):
        return P.Transpose()(x, (0, 1, 3, 2,))

    def construct(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # attention_scores = nn.MatMul(query_layer, self.transpose2(key_layer))
        attention_scores = mindspore.ops.matmul(query_layer, self.transpose2(key_layer))
        attention_scores = attention_scores / P.Sqrt()(self.attention_head_size2)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        # context_layer = nn.MatMul(attention_probs, value_layer)
        context_layer = mindspore.ops.matmul(attention_probs, value_layer)
        context_layer = P.Transpose()(context_layer, (0, 2, 1, 3,))
        new_context_layer_shape = P.Shape()(context_layer)[:-2] + (self.all_head_size,)
        context_layer = P.Reshape()(context_layer, new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Cell):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Dense(config.hidden_size, config.transformer_mlp_dim)
        self.fc2 = nn.Dense(config.transformer_mlp_dim, config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(config.transformer_dropout_rate)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Cell):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None

        if config.patches_grid is not None:
            grid_size = config.patches_grid
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = config.patches_size
            n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet_num_layers,
                                         width_factor=config.resnet_width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = Parameter(Tensor(np.zeros((1, n_patches+1, config.hidden_size)).astype(np.float32)), name="q1", requires_grad=True)
        self.cls_token = Parameter(Tensor(np.zeros((1, 1, config.hidden_size)).astype(np.float32)), name="q2", requires_grad=True)

        self.dropout = nn.Dropout(config.transformer_dropout_rate)

    def construct(self, x):
        B = x.shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = P.BroadcastTo((B, self.cls_token.shape[1], self.cls_token.shape[2]))(self.cls_token)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        # x = P.Flatten()(x)
        x = P.Reshape()(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = P.Transpose()(x, (0, 2, 1))
        x = P.Concat(1)((cls_tokens, x))

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Cell):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.attention_norm = nn.LayerNorm([config.hidden_size], epsilon=1e-6)  # ///
        self.ffn_norm = nn.LayerNorm([config.hidden_size], epsilon=1e-6)  # ///
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def construct(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Cell):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.CellList([])
        self.encoder_norm = nn.LayerNorm([config.hidden_size], epsilon=1e-6)  # ///
        for _ in range(config.transformer_num_layers):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def construct(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Cell):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def construct(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Cell):
    def __init__(self, config, img_size=(224, 224), num_classes=1000, vis=False):  # num_classes=21843
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = nn.Dense(config.hidden_size, num_classes)

    def construct(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])
        return logits


# CONFIGS = {
    # 'ViT-B_16': configs.get_b16_config(),
    # 'ViT-B_32': configs.get_b32_config(),
    # 'ViT-L_16': configs.get_l16_config(),
    # 'ViT-L_32': configs.get_l32_config(),
    # 'ViT-H_14': configs.get_h14_config(),
    # 'R50-ViT-B_16': configs.get_r50_b16_config(),
    # 'testing': configs.get_testing(),
# }
