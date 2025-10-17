from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.utils.checkpoint as checkpoint
import math
from torch.nn import init


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., skip_connect=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.skip_connect = skip_connect

    def forward(self, x):
        xs = self.fc1(x)
        xs = self.act(xs)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        xs = self.fc2(xs)
        xs = self.drop(xs)
        if self.skip_connect:
            x = x + xs
        else: 
            x = xs
        return x
 

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(
                self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# from https://github.com/AviSoori1x/makeMoE/tree/main
# noisy top-k gating
 
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        # layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)
        self.layer_loss = None
        self.gate_logits = None

    def get_layer_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        Get the load balancing loss by following the Switch Transformer
        """
        _, selected_experts = torch.topk(gate_logits, self.top_k, dim=-1) # [num_layers, num_tokens, top_k]
        expert_mask = torch.nn.functional.one_hot(selected_experts, self.num_experts) # [num_layers, num_tokens, top_k, num_experts]
        # For a given token, determine if it was routed to a given expert. Think of this as a collection of top_k-hot vectors.
        expert_mask = torch.max(expert_mask, dim=-2).values.float() # [num_layers, num_tokens, num_experts]
        tokens_per_layer_and_expert = torch.mean(expert_mask, dim=-2) # [num_layers, num_experts]
        router_prob_per_layer_and_expert = torch.mean(gate_logits, dim=-2) # [num_layers, num_experts]
        return torch.mean(tokens_per_layer_and_expert * router_prob_per_layer_and_expert) * self.num_experts**2

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)
        # Noise logits
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise
        
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)

        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        self.gate_logits=indices
        if mh_output.requires_grad:
            self.layer_loss = self.get_layer_loss(noisy_logits)
        return router_output, indices

def kaiming_init_weights(m):
    if isinstance(m, (nn.Linear)):
        init.kaiming_normal_(m.weight)
 
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25,
                 act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

        self.apply(self._init_weights)
        nn.init.constant_(self.D_fc2.weight, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x




class SparseMoAE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, act_layer=None, drop=0., num_experts=2, top_k=1, capacity_factor=1.0, mlp_ratio=0.25, skip_connect=True):
        super(SparseMoAE, self).__init__()
        self.router = NoisyTopkRouter(in_features, num_experts, top_k)
        self.experts = nn.ModuleList([Adapter(D_features=in_features, mlp_ratio=mlp_ratio,
                                     act_layer=act_layer, skip_connect=skip_connect) for _ in range(num_experts)])

        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts
        # kaimini_init
        self.apply(kaiming_init_weights)

    def forward(self, x):
        # Assuming x has shape [batch_size, seq_len, n_embd]
        batch_size, seq_len, _ = x.shape
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Flatten the batch and sequence dimensions to treat each token independently
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        tokens_per_batch = batch_size * seq_len * self.top_k
        expert_capacity = int(
            (tokens_per_batch / self.num_experts) * self.capacity_factor)

        updates = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)
            limited_indices = selected_indices[:expert_capacity] if selected_indices.numel(
            ) > expert_capacity else selected_indices
            if limited_indices.numel() > 0:
                expert_input = flat_x[limited_indices]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[limited_indices, i].unsqueeze(
                    1)
                weighted_output = expert_output * gating_scores
                updates.index_add_(0, limited_indices, weighted_output)

        # Reshape updates to match the original dimensions of x
        final_output += updates.view(batch_size, seq_len, -1)

        return final_output

class SparseMoEWithCapacity(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, act_layer=None, drop=0., num_experts=2, top_k=1, capacity_factor=1.0):
        super(SparseMoEWithCapacity, self).__init__()
        self.router = NoisyTopkRouter(in_features, num_experts, top_k)
        self.experts = nn.ModuleList([Mlp(in_features=in_features, hidden_features=hidden_features,
                                     act_layer=act_layer, drop=drop) for _ in range(num_experts)])

        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts
        # kaimini_init
        self.apply(kaiming_init_weights)

    def forward(self, x):
        # Assuming x has shape [batch_size, seq_len, n_embd]
        batch_size, seq_len, _ = x.shape
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Flatten the batch and sequence dimensions to treat each token independently
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        tokens_per_batch = batch_size * seq_len * self.top_k
        expert_capacity = int(
            (tokens_per_batch / self.num_experts) * self.capacity_factor)

        updates = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)
            limited_indices = selected_indices[:expert_capacity] if selected_indices.numel(
            ) > expert_capacity else selected_indices
            if limited_indices.numel() > 0:
                expert_input = flat_x[limited_indices]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[limited_indices, i].unsqueeze(
                    1)
                weighted_output = expert_output * gating_scores
                updates.index_add_(0, limited_indices, weighted_output)

        # Reshape updates to match the original dimensions of x
        final_output += updates.view(batch_size, seq_len, -1)

        return final_output

class SparseMoE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, act_layer=None, drop=0., num_experts=2, top_k=1, capacity_factor=1.0):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(in_features, num_experts, top_k)
        self.experts = nn.ModuleList([Mlp(in_features=in_features, hidden_features=hidden_features,
                                     act_layer=act_layer, drop=drop) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output




class BlockMoEAdapters(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=0.25, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, moe=False, num_experts=4, top_k=2, capacity_factor=1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.moe_layer = SparseMoEWithCapacity(
            in_features=dim,
            hidden_features=int(dim * 0.25),
            act_layer=act_layer,
            drop=drop,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
        )
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x1 = self.norm2(x)
            x = x + self.drop_path(self.mlp(x1) + self.moe_layer(x1))
            del x1
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PadIm2Video(torch.nn.Module):
    def __init__(self, ntimes, pad_type, time_dim=2):
        super().__init__()
        self.time_dim = time_dim
        assert ntimes > 0
        assert pad_type in ["zero", "repeat"]
        self.ntimes = ntimes
        self.pad_type = pad_type

    def forward(self, x):
        if x.ndim == 4:
            # B, C, H, W -> B, C, T, H, W
            x = x.unsqueeze(self.time_dim)

        if x.shape[self.time_dim] == 1:
            if self.pad_type == "repeat":
                new_shape = [1] * len(x.shape)
                new_shape[self.time_dim] = self.ntimes
                x = x.repeat(new_shape)
            elif self.pad_type == "zero":
                padarg = [0, 0] * len(x.shape)
                padarg[2 * self.time_dim + 1] = self.ntimes - \
                    x.shape[self.time_dim]
                x = torch.nn.functional.pad(x, padarg)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2, pad_type="repeat"):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] //
                                                        patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                              kernel_size=(self.tubelet_size,
                                           patch_size[0], patch_size[1]),
                              stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))
        self.padim2video = PadIm2Video(self.tubelet_size, pad_type)

    def forward(self, x, **kwargs):
        x = self.padim2video(x)
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31


def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i)
                              for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)


class VisionTransformerMoE(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 fc_drop_rate=0.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_checkpoint=False,
                 use_mean_pooling=True,
                 moe=None,
                 multi_head=[7, 7],
                 keys=['image', 'video']):
        super().__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = all_frames
        self.first_patch_idx = 0

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(
                num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if moe is not None:
            moe_layers = moe.get('layers', [])
            moe_num_experts = moe.get('num_experts', 2)
            moe_top_k = moe.get('top_k', 1)
            moe_capacity_factor = moe.get('capacity_factor', 1.0)
            moe_type = moe.get('type', 'sparse')
            if moe_type == 'moe_adapters':
                MoEBlock = BlockMoEAdapters
            else:
                raise ValueError(f"Unknown moe type {moe_type}")
            blocks = []
            for i in range(depth):
                if i in moe_layers:
                    if moe_type != 'moe_lora':
                        blocks.append(MoEBlock(
                            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[
                                i], norm_layer=norm_layer,
                            init_values=init_values, moe=True, num_experts=moe_num_experts, top_k=moe_top_k, capacity_factor=moe_capacity_factor))
                    else:
                        blocks.append(MoEBlock(
                            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[
                                i], norm_layer=norm_layer,
                            init_values=init_values, moe=True, num_experts=moe_num_experts, top_k=moe_top_k, capacity_factor=moe_capacity_factor,
                            lora_rank=moe.get('lora_rank', 16), lora_alpha=moe.get('lora_alpha', 16)))
                else:
                    blocks.append(Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[
                            i], norm_layer=norm_layer,
                        init_values=init_values))
            self.blocks = nn.ModuleList(blocks)

        else:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
                for i in range(depth)])

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None

        self.fc_dropout = nn.Dropout(
            p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        # self.head = nn.Linear(
        #     embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head = nn.ModuleDict()
        for i, key in enumerate(keys):
            self.head[key] = nn.Linear(embed_dim, multi_head[i])

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        try:
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)
        except:
            pass

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def interpolate_pos_encoding_2d(self, target_spatial_size, pos_embed):
        N = pos_embed.shape[1]
        if N == target_spatial_size:
            return pos_embed
        dim = pos_embed.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=math.sqrt(target_spatial_size / N),
            mode="bicubic",
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed

    def interpolate_pos_encoding(
        self,
        npatch_per_img,
        pos_embed,
        patches_layout,
        input_shape=None,
        first_patch_idx=1,
    ):
        assert (
            first_patch_idx == 0 or first_patch_idx == 1
        ), "there is 1 CLS token or none"
        # since it's 1 if cls_token exists
        N = pos_embed.shape[1] - first_patch_idx
        if npatch_per_img == N:
            return pos_embed
        class_emb = pos_embed[:, :first_patch_idx]
        pos_embed = pos_embed[:, first_patch_idx:]

        if input_shape is None or patches_layout[0] == 1:
            # simple 2D pos embedding, no temporal component
            pos_embed = self.interpolate_pos_encoding_2d(
                npatch_per_img, pos_embed)
        elif patches_layout[0] > 1:
            # pos embed has a temporal component
            assert len(input_shape) == 4, "temporal interpolation not supported"
            # we only support 2D interpolation in this case
            num_frames = patches_layout[0]
            num_spatial_tokens = patches_layout[1] * patches_layout[2]
            pos_embed = pos_embed.view(1, num_frames, num_spatial_tokens, -1)
            # interpolate embedding for zeroth frame
            pos_embed = self.interpolate_pos_encoding_2d(
                npatch_per_img, pos_embed=pos_embed[0, 0, ...].unsqueeze(0)
            )
        else:
            raise ValueError("This type of interpolation isn't implemented")
        return torch.cat((class_emb, pos_embed), dim=1)

    def forward_features(self, x):
        orig_shape = x.shape

        x = self.patch_embed(x)

        B, _, _ = x.size()

        if self.pos_embed is not None:
            pos_embed = self.interpolate_pos_encoding(
                x.size(1), self.pos_embed, (self.num_frames // self.tubelet_size, self.img_size // self.patch_size, self.img_size // self.patch_size), orig_shape, self.first_patch_idx)
            x = x + \
                pos_embed.expand(
                    B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

    def forward(self, x, save_feature=False):
        if x.ndim == 5:
            key = 'video'
        else:
            key = 'image'
        x = self.forward_features(x)

        if save_feature:
            return self.head[key](self.fc_dropout(x)), self.fc_dropout(x)
        else:
            return self.head[key](self.fc_dropout(x))

    def get_aux_loss(self) -> torch.Tensor:
        """
        Get the load balancing loss for the whole model
        """
        model_loss = torch.tensor(
            0, dtype=torch.float).to(self.pos_embed.device)

        for name, module in self.named_modules():
            if name.endswith('router'):
                layer_loss = module.layer_loss
                model_loss += layer_loss

        return model_loss
 

 

@register_model
def vitmoe_small_patch16_160(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformerMoE(
        img_size=160, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vitmoe_base_patch16_160(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformerMoE(
        img_size=160, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vitmoe_large_patch16_160(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformerMoE(
        img_size=160, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
