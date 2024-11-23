""" MLP-Mixer, ResMLP, and gMLP in PyTorch

This impl originally based on MLP-Mixer paper.

Official JAX impl: https://github.com/google-research/vision_transformer/blob/linen/vit_jax/models_mixer.py

Paper: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601

@article{tolstikhin2021,
  title={MLP-Mixer: An all-MLP Architecture for Vision},
  author={Tolstikhin, Ilya and Houlsby, Neil and Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner,
        Thomas and Yung, Jessica and Keysers, Daniel and Uszkoreit, Jakob and Lucic, Mario and Dosovitskiy, Alexey},
  journal={arXiv preprint arXiv:2105.01601},
  year={2021}
}

Also supporting ResMlp, and a preliminary (not verified) implementations of gMLP

Code: https://github.com/facebookresearch/deit
Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
@misc{touvron2021resmlp,
      title={ResMLP: Feedforward networks for image classification with data-efficient training},
      author={Hugo Touvron and Piotr Bojanowski and Mathilde Caron and Matthieu Cord and Alaaeldin El-Nouby and
        Edouard Grave and Armand Joulin and Gabriel Synnaeve and Jakob Verbeek and Hervé Jégou},
      year={2021},
      eprint={2105.03404},
}

Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
@misc{liu2021pay,
      title={Pay Attention to MLPs},
      author={Hanxiao Liu and Zihang Dai and David R. So and Quoc V. Le},
      year={2021},
      eprint={2105.08050},
}

A thank you to paper authors for releasing code and weights.

Hacked together by / Copyright 2021 Ross Wightman
"""
from IPython import embed
import math
from functools import partial
from typing import List, Optional, Union, Tuple

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import PatchEmbed, GatedMlp, DropPath, lecun_normal_, to_2tuple # , Mlp, GluMlp
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import named_apply, checkpoint_seq
from ._registry import generate_default_cfgs, register_model, register_model_deprecations


from timm.models.decomposition import MLPDecomposed as Mlp
from timm.models.decomposition import FairSwiGLU as SwiGLU
from timm.models.decomposition import FairGluMlp as GluMlp
from timm.models.decomposition import FairSiLU as SiLU
from timm.models.decomposition import LayerNormDecomposed as LayerNorm
from timm.models.decomposition import GELUDecomposed as GELU

import timm.models.decomposition as decomposition
from timm.models.decomposition import (
    DecompositionConfig,
    ic,
    print_diag,
    ic_colorize2,
    detach_maybe,
    decompose_p_v2,
    check_attributions,
    check_attributions_v2,
    sum_attributions,
    simple_obj,
    simple_obj_update,
    LayerNormDecomposed,
    AttentionDecomposed,
    attribute_patches_aggregate,
    MLPDecomposed,
    LayerScaleDecomposed,
    LinearDecomposed,
    SoftmaxDecomposed,
    residual_decomposed,
    nondecomposed_forward,
    nondecomposed_features,
    config_from_inputs,
    dynamic_config_contexts,
    decomposed_inputs_p,
    mixer_transpose,
    FairGluMlp,
    GELUDecomposed,
)

from pynight.common_obj import delattr_force
from pynight.common_torch import (
    no_grad_maybe,
    torch_shape_get,
    torch_gpu_remove_all,
    torch_gpu_empty_cache,
    torch_gpu_memory_stats,
    mask_to_zero,
)

from pynight.common_debugging import (
    stacktrace_get,
    stacktrace_caller_line,
)

from pynight.common_dynamic import (
    DynamicVariables,
    DynamicObject,
    dynamic_set,
    dynamic_get,
)


__all__ = ['MixerBlock', 'MlpMixer']  # model_registry will add each entrypoint fn to this


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self,
            dim,
            seq_len,
            mlp_ratio=(0.5, 4.0),
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=GELU,
            drop=0.,
            drop_path=0.,
    ):
        super().__init__()
        self.prefix = "unset."

        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)
        
        
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.prefix = prefix
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    
    def forward(self, inputs):
        ## Original Implementation:
        #     x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        #     x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        #     return x
        ##
        print_diag(f'entering: {self.prefix}', group='progress')

        decomposition_config = config_from_inputs(inputs)

        inputs = attribute_patches_aggregate(
            inputs,
            prefix=self.prefix,
            mode=decomposition_config.attributions_aggregation_strategy,
        )

        residuals = nondecomposed_features(inputs)
        residual_attributions_v = nondecomposed_features(inputs, name='attributions_v', default=None)

        inputs = self.norm1(inputs)

        token_mixer_decompose_p = decompose_p_v2(
            decomposition_config.token_mixer_decompose_p,
            attributions_v=nondecomposed_features(inputs, name='attributions_v', default=None,),
        )
        with DynamicVariables(
            decomposition.dynamic_obj,
            decompose_p=token_mixer_decompose_p,
        ):
            inputs = mixer_transpose(
                inputs,
                prefix=self.prefix,
            )
            inputs = self.mlp_tokens(inputs)
            inputs = mixer_transpose(
                inputs,
                prefix=self.prefix,
            )

        inputs = nondecomposed_forward(self.drop_path, inputs)

        inputs = residual_decomposed(
            inputs=inputs,
            residuals=residuals,
            residual_attributions_v=residual_attributions_v,
            decompose_p=decomposition_config.residual1_decompose_p,
            prefix=f"{self.prefix}residual1.",
        )


        residuals = nondecomposed_features(inputs)
        residual_attributions_v = nondecomposed_features(inputs, name='attributions_v', default=None)

        inputs = self.norm2(inputs)
        inputs = self.mlp_channels(inputs)
        inputs = nondecomposed_forward(self.drop_path, inputs)
        inputs = residual_decomposed(
            inputs=inputs,
            residuals=residuals,
            residual_attributions_v=residual_attributions_v,
            decompose_p=decomposition_config.residual2_decompose_p,
            prefix=f"{self.prefix}residual2.",
        )

        output = nondecomposed_features(inputs)

        if decomposition_config.save_intermediate_p:
            inputs = simple_obj_update(
                inputs,
                f'{self.prefix}attributions_v',
                inputs.attributions_v.detach().cpu(),
                f'{self.prefix}output',
                output.detach().cpu(),
            )
            
        delattr_force(self, "stored_output")
        if decomposition.dynamic_obj.block_output_store_p:
            self.stored_output = output.detach().cpu()

        delattr_force(self, "stored_output_grad")
        if decomposition.dynamic_obj.block_output_grad_store_p:
            def store_output_grad(grad):
                # ic(torch_shape_get(grad))
                self.stored_output_grad = grad.detach().cpu()

            output.requires_grad_(True)
            output.register_hook(store_output_grad)

        return inputs


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))

    def forward(self, x):
        return torch.addcmul(self.beta, self.alpha, x)


class ResBlock(nn.Module):
    """ Residual MLP block w/ LayerScale and Affine 'norm'

    Based on: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    def __init__(
            self,
            dim,
            seq_len,
            mlp_ratio=4,
            mlp_layer=Mlp,
            norm_layer=Affine,
            act_layer=GELU,
            init_values=1e-4,
            drop=0.,
            drop_path=0.,
    ):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.linear_tokens = nn.Linear(seq_len, seq_len)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, drop=drop)
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        return x


class SpatialGatingUnit(nn.Module):
    """ Spatial Gating Unit

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(self, dim, seq_len, norm_layer=nn.LayerNorm):
        super().__init__()
        gate_dim = dim // 2
        self.norm = norm_layer(gate_dim)
        self.proj = nn.Linear(seq_len, seq_len)

    def init_weights(self):
        # special init for the projection gate, called as override by base model init
        nn.init.normal_(self.proj.weight, std=1e-6)
        nn.init.ones_(self.proj.bias)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.proj(v.transpose(-1, -2))
        return u * v.transpose(-1, -2)


class SpatialGatingBlock(nn.Module):
    """ Residual Block w/ Spatial Gating

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(
            self,
            dim,
            seq_len,
            mlp_ratio=4,
            mlp_layer=GatedMlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=GELU,
            drop=0.,
            drop_path=0.,
    ):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm = norm_layer(dim)
        sgu = partial(SpatialGatingUnit, seq_len=seq_len)
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, gate_layer=sgu, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        return x


class MlpMixer(nn.Module):
    # def __str__(self):
    #     #: return only the class name when printed
    #     return self.__class__.__name__
    
    def __repr__(self):
        #: return only the class name when printed
        return self.__class__.__name__
 
    def __init__(
            self,
            num_classes=1000,
            img_size=224,
            in_chans=3,
            patch_size=16,
            num_blocks=8,
            embed_dim=512,
            mlp_ratio=(0.5, 4.0),
            block_layer=MixerBlock,
            # mlp_layer=Mlp,
            mlp_layer=MLPDecomposed,
            # norm_layer=partial(nn.LayerNorm, eps=1e-6),
            norm_layer=None,
            act_layer=GELU,
            drop_rate=0.,
            proj_drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,
            stem_norm=False,
            global_pool='avg',
            linear_layer=LinearDecomposed, #: @NightMachinery
    ):
        super().__init__()

        # self.fair_p = True
        self.droppable_tokens_p = False

        self.num_prefix_tokens = 0 #: no class token

        self.linear_layer = linear_layer
        
        if norm_layer is None:
            norm_layer = partial(LayerNormDecomposed, eps=1e-6)
        elif isinstance(norm_layer, nn.LayerNorm.__class__):
            norm_layer = partial(LayerNormDecomposed,)
        else:
            # ic(norm_layer, norm_layer((2,)).extra_repr())
            raise NotImplementedError(f"only the default layer norm is supported at this time: {repr(norm_layer)}")
        
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.grad_checkpointing = False

        self.stem = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if stem_norm else None,
        )
        reduction = self.stem.feat_ratio() if hasattr(self.stem, 'feat_ratio') else patch_size
        # FIXME drop_path (stochastic depth scaling rule or all the same?)
        self.blocks = nn.Sequential(*[
            block_layer(
                embed_dim,
                self.stem.num_patches,
                mlp_ratio,
                mlp_layer=mlp_layer,
                norm_layer=norm_layer,
                act_layer=act_layer,
                drop=proj_drop_rate,
                drop_path=drop_path_rate,
            )
            for _ in range(num_blocks)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction) for i in range(num_blocks)]
        self.norm = norm_layer(embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = self.linear_layer(embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(nlhb=nlhb)

    @torch.jit.ignore
    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = self.linear_layer(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def pixels2patches(
        self,
        x,
    ):
        x = self.stem(x)

        return x

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        B, _, height, width = x.shape
        x = self.stem(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            x = blk(x)
            if i in take_indices:
                # normalize intermediates with final norm layer if enabled
                intermediates.append(self.norm(x) if norm else x)

        # process intermediates
        if reshape:
            # reshape to BCHW output format
            H, W = self.stem.dynamic_feat_size((height, width))
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices


    def forward_features(
            self,
            x,
            *,
            decomposition_config=None,
            patchify_p=True,
            decompose_p=False,
            keep_indices=None,
    ):
        if patchify_p:
            #: mixer_b16_224.goog_in21k_ft_in1k:
            #: (stem): PatchEmbed(
            #:     (proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
            #:     (norm): Identity()
            #: )
            x = self.pixels2patches(x)

        if decompose_p:
            assert decomposition_config is not None

            assert (self.training) == False, "decomposition not supported in train mode"
            
            inputs = simple_obj(
                features=x,
                decomposition_config=decomposition_config,
            )

            inputs = attribute_patches_aggregate(
                inputs,
                prefix="forward_features.start.",
                mode=decomposition_config.attributions_aggregation_strategy,
                # mode='reset',
                start_p=True,
            )
        else:
            inputs = x

        if self.grad_checkpointing and not torch.jit.is_scripting():
            inputs = checkpoint_seq(self.blocks, inputs)
        else:
            inputs = self.blocks(inputs)
        inputs = self.norm(inputs)
        return inputs

    def forward_head(
            self,
            x,
            pre_logits: bool = False,
            *,
            # decomposition_config=None,
            decompose_p=False,
    ):
        ## Original Implementation
        # if self.global_pool == 'avg':
        #     x = x.mean(dim=1)
        # x = self.head_drop(x)
        # return x if pre_logits else self.head(x)
        ##
        inputs = x
        if decomposed_inputs_p(inputs):
            features = inputs.features
            attributions_v = inputs.attributions_v
            decompose_p = decompose_p_v2(
                decompose_p,
                attributions_v=inputs.attributions_v,
            )
        else:
            features = inputs
            attributions_v = None
            decompose_p = False

        if self.global_pool:
            if self.global_pool == 'avg':
                features = features[:, self.num_prefix_tokens :].mean(dim=1)

                if decompose_p:
                    attributions_v = attributions_v[:, self.num_prefix_tokens :].mean(dim=1)
                    #: batch, seq, patch_attribution, pixel_attribution, hidden_dim
            else:
                features = features[:, 0]
                if decompose_p:
                    attributions_v = attributions_v[:, 0]

        if decompose_p:
            print_diag(f"head pooling: attributions_v.shape: {(attributions_v.shape)}", group="shape")

        inputs = simple_obj_update(
            inputs,
            features=features,
            attributions_v=attributions_v,
            del_p='can',
        )

        # inputs = self.fc_norm(inputs)

        inputs = nondecomposed_forward(self.head_drop, inputs)

        if pre_logits:
            return inputs
        else:
            return self.head(inputs)


    def forward(
            self,
            x,
            *,
            decomposition_config=None,
            patchify_p=True,
            decompose_p=False,
            return_features_p=False,
            keep_indices=None, #: dummy
    ):
        inputs = x
        
        inputs = self.forward_features(
            inputs,
            patchify_p=patchify_p,
            decomposition_config=decomposition_config,
            decompose_p=decompose_p,
            keep_indices=keep_indices,
        )
        
        if return_features_p:
            features = simple_obj_update(
                inputs,
                del_p=False,
                _print_p=False,
            )
        else:
            features = None

        inputs = self.forward_head(
            inputs,
            decompose_p=decompose_p,
        )

        if features:
            return simple_obj(
                features=features,
                outputs=inputs,
            )
        else:
            return inputs


    def forward_patch_level(self, *args, **kwargs):
        return self.forward(*args, patchify_p=False, **kwargs,)


    def forward_decomposed(self, *args, **kwargs):
        return self.forward(
            *args,
            decompose_p=True,
            **kwargs,
        )


    def forward_patch_level_decomposed(self, *args, **kwargs):
        return self.forward_decomposed(*args, patchify_p=False, **kwargs,)


def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax=False):
    """ Mixer weight initialization (trying to match Flax defaults)
    """
    if isinstance(module, nn.Linear): #: should work with LinearDecomposed, too
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # like MLP init in vit (my original init)
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        # NOTE if a parent module contains init_weights method, it can override the init of the
        # child modules as this will be called in depth-first order.
        module.init_weights()


def checkpoint_filter_fn(state_dict, model):
    """ Remap checkpoints if needed """
    if 'patch_embed.proj.weight' in state_dict:
        # Remap FB ResMlp models -> timm
        out_dict = {}
        for k, v in state_dict.items():
            k = k.replace('patch_embed.', 'stem.')
            k = k.replace('attn.', 'linear_tokens.')
            k = k.replace('mlp.', 'mlp_channels.')
            k = k.replace('gamma_', 'ls')
            if k.endswith('.alpha') or k.endswith('.beta'):
                v = v.reshape(1, 1, -1)
            out_dict[k] = v
        return out_dict
    return state_dict


def _create_mixer(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', 3)
    model = build_model_with_cfg(
        MlpMixer,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 0.875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'stem.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'mixer_s32_224.untrained': _cfg(),
    'mixer_s16_224.untrained': _cfg(),
    'mixer_b32_224.untrained': _cfg(),
    'mixer_b16_224.goog_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224-76587d61.pth',
    ),
    'mixer_b16_224.goog_in21k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224_in21k-617b3de2.pth',
        num_classes=21843
    ),
    'mixer_l32_224.untrained': _cfg(),
    'mixer_l16_224.goog_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224-92f9adc4.pth',
    ),
    'mixer_l16_224.goog_in21k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224_in21k-846aa33c.pth',
        num_classes=21843
    ),

    # Mixer ImageNet-21K-P pretraining
    'mixer_b16_224.miil_in21k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mixer_b16_224_miil_in21k-2a558a71.pth',
        mean=(0., 0., 0.), std=(1., 1., 1.), crop_pct=0.875, interpolation='bilinear', num_classes=11221,
    ),
    'mixer_b16_224.miil_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mixer_b16_224_miil-9229a591.pth',
        mean=(0., 0., 0.), std=(1., 1., 1.), crop_pct=0.875, interpolation='bilinear',
    ),

    'gmixer_12_224.untrained': _cfg(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'gmixer_24_224.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gmixer_24_224_raa-7daf7ae6.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    'resmlp_12_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlp_12_no_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'resmlp_24_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlp_24_no_dist.pth',
        #url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resmlp_24_224_raa-a8256759.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'resmlp_36_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlp_36_no_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'resmlp_big_24_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlpB_24_no_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    'resmlp_12_224.fb_distilled_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlp_12_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'resmlp_24_224.fb_distilled_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlp_24_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'resmlp_36_224.fb_distilled_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlp_36_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'resmlp_big_24_224.fb_distilled_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlpB_24_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    'resmlp_big_24_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlpB_24_22k.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    'resmlp_12_224.fb_dino': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlp_12_dino.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'resmlp_24_224.fb_dino': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlp_24_dino.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    'gmlp_ti16_224.untrained': _cfg(),
    'gmlp_s16_224.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gmlp_s16_224_raa-10536d42.pth',
    ),
    'gmlp_b16_224.untrained': _cfg(),
})


@register_model
def mixer_s32_224(pretrained=False, **kwargs) -> MlpMixer:
    """ Mixer-S/32 224x224
    Paper: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=32, num_blocks=8, embed_dim=512, **kwargs)
    model = _create_mixer('mixer_s32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_s16_224(pretrained=False, **kwargs) -> MlpMixer:
    """ Mixer-S/16 224x224
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512, **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_b32_224(pretrained=False, **kwargs) -> MlpMixer:
    """ Mixer-B/32 224x224
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=32, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer('mixer_b32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_b16_224(pretrained=False, **kwargs) -> MlpMixer:
    """ Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer('mixer_b16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_l32_224(pretrained=False, **kwargs) -> MlpMixer:
    """ Mixer-L/32 224x224.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=32, num_blocks=24, embed_dim=1024, **kwargs)
    model = _create_mixer('mixer_l32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_l16_224(pretrained=False, **kwargs) -> MlpMixer:
    """ Mixer-L/16 224x224. ImageNet-1k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=24, embed_dim=1024, **kwargs)
    model = _create_mixer('mixer_l16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def gmixer_12_224(pretrained=False, **kwargs) -> MlpMixer:
    """ Glu-Mixer-12 224x224
    Experiment by Ross Wightman, adding SwiGLU to MLP-Mixer
    """
    model_args = dict(
        patch_size=16, num_blocks=12, embed_dim=384, mlp_ratio=(1.0, 4.0),
        mlp_layer=GluMlp, act_layer=SiLU, **kwargs)
    model = _create_mixer('gmixer_12_224', pretrained=pretrained, **model_args)
    return model


@register_model
def gmixer_24_224(pretrained=False, **kwargs) -> MlpMixer:
    """ Glu-Mixer-24 224x224
    Experiment by Ross Wightman, adding SwiGLU to MLP-Mixer
    """
    model_args = dict(
        patch_size=16, num_blocks=24, embed_dim=384, mlp_ratio=(1.0, 4.0),
        mlp_layer=GluMlp, act_layer=SiLU, **kwargs)
    model = _create_mixer('gmixer_24_224', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_12_224(pretrained=False, **kwargs) -> MlpMixer:
    """ ResMLP-12
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=16, num_blocks=12, embed_dim=384, mlp_ratio=4, block_layer=ResBlock, norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_12_224', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_24_224(pretrained=False, **kwargs) -> MlpMixer:
    """ ResMLP-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=16, num_blocks=24, embed_dim=384, mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-5), norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_24_224', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_36_224(pretrained=False, **kwargs) -> MlpMixer:
    """ ResMLP-36
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=16, num_blocks=36, embed_dim=384, mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-6), norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_36_224', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_big_24_224(pretrained=False, **kwargs) -> MlpMixer:
    """ ResMLP-B-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=8, num_blocks=24, embed_dim=768, mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-6), norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_big_24_224', pretrained=pretrained, **model_args)
    return model


@register_model
def gmlp_ti16_224(pretrained=False, **kwargs) -> MlpMixer:
    """ gMLP-Tiny
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=128, mlp_ratio=6, block_layer=SpatialGatingBlock,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_ti16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def gmlp_s16_224(pretrained=False, **kwargs) -> MlpMixer:
    """ gMLP-Small
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=256, mlp_ratio=6, block_layer=SpatialGatingBlock,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_s16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def gmlp_b16_224(pretrained=False, **kwargs) -> MlpMixer:
    """ gMLP-Base
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=512, mlp_ratio=6, block_layer=SpatialGatingBlock,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_b16_224', pretrained=pretrained, **model_args)
    return model


register_model_deprecations(__name__, {
    'mixer_b16_224_in21k': 'mixer_b16_224.goog_in21k_ft_in1k',
    'mixer_l16_224_in21k': 'mixer_l16_224.goog_in21k_ft_in1k',
    'mixer_b16_224_miil': 'mixer_b16_224.miil_in21k_ft_in1k',
    'mixer_b16_224_miil_in21k': 'mixer_b16_224.miil_in21k',
    'resmlp_12_distilled_224': 'resmlp_12_224.fb_distilled_in1k',
    'resmlp_24_distilled_224': 'resmlp_24_224.fb_distilled_in1k',
    'resmlp_36_distilled_224': 'resmlp_36_224.fb_distilled_in1k',
    'resmlp_big_24_distilled_224': 'resmlp_big_24_224.fb_distilled_in1k',
    'resmlp_big_24_224_in22ft1k': 'resmlp_big_24_224.fb_in22k_ft_in1k',
    'resmlp_12_224_dino': 'resmlp_12_224',
    'resmlp_24_224_dino': 'resmlp_24_224',
})
