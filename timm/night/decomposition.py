from IPython import embed
import tempfile
import sys
import hashlib
import pickle
import skimage
from typing import Iterable
import os
from os import getenv
import uuid
import datetime
import pytz
import gc
import re
import copy
import traceback
from functools import partial
import uuid
import urllib.request
import io
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import types
from types import SimpleNamespace
from contextlib import (
    nullcontext,
    ExitStack,
)
from pathlib import Path

import PIL
from PIL import Image, ImageDraw
import blend_modes
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm

import numpy as np

import einops
from einops import rearrange

import kornia.filters

import torch
from torch.jit import Final
import torch.nn.functional as F
from torch import Tensor
import torch.utils.checkpoint
import torchvision.transforms
import torch.nn

nn = torch.nn

from pynight.common_json import (
    json_save,
)
from pynight.common_regex import (
    rget,
    float_pattern,
    rget_percent,
    re_maybe,
)
from pynight.common_obj import delattr_force
from pynight.common_str import print_to_file
from pynight.common_benchmark import timed
from pynight.common_hash import hash_url
from pynight.common_datetime import datetime_dir_name
from pynight.common_functional import fn_name
import pynight.common_telegram as common_telegram
from pynight.common_plt import (
    colormap_get,
)

from pynight.common_numpy import image_url2np
from pynight.common_clip import (
    open_clip_sep,
)
from pynight.common_model_name import (
    model_name_clip_p,
    model_name_eva2_p,
    model_needs_MLP_DU_p,
)
from pynight.common_timm import (
    model_name_get,
    model_transform_get,
    image_batch_from_urls,
    image_from_url,
    patch_info_from_name,
)

from pynight.common_torch import (
    mod_copy_params_buffers,
    mod_init_from,
    module_mapper,
    tensor_register_hook,
    decomposed_inputs_p,
    store_tensor_with_grad,
    grey_tensor_to_pil,
    torch_prefix_autoset,
    get_compact_gbrand,
    flatten_and_move_to_last,
    swap_backward,
    quantile_sum,
    no_grad_maybe,
    torch_shape_get,
    model_device_get,
    img_tensor_show,
    hash_tensor,
    torch_memory_tensor,
    tensorify_scalars,
    rank_tensor,
    scale_patch_to_pixel,
)
from pynight.common_attr import (
    normalize_map,
)

from pynight.common_debugging import (
    stacktrace_get,
    stacktrace_caller_line,
)

from pynight.common_files import (
    mkdir,
)

from pynight.common_iterable import (
    to_iterable,
    lst_filter_out,
    list_get,
)

import timm

from timm.layers.mlp import (
    SwiGLU,
    GluMlp,
)

from timm.layers.helpers import (
    to_2tuple,
)


from pynight.common_dict import (
    # SimpleObject,
    simple_obj,
    dict_filter_out,
)

import pynight.common_jax  #: registers SimpleObject as a pytree

from pynight.common_dynamic import (
    DynamicVariables,
    DynamicObject,
    dynamic_set,
    dynamic_get,
)

dynamic_vars = dict()
dynamic_obj = DynamicObject(dynamic_vars, default_to_none_p=True)


def dynamic_obj_to_metadata_dict():
    d = dict(dynamic_obj)
    d = dict_filter_out(
        d,
        exclude_keys=[
            "print_diag_enabled_groups",
        ],
    )
    return d


dynamic_obj.cuda_gc_mode = getenv(
    "DECOMPV_CUDA_GC_MODE",
    default="y",
)

dynamic_obj.backward_softmax_p = False
##
PLOT_IMAGE_SIZE = 600
##
dynamic_obj.decompose_p = "static"  #: use the static setting from the config


def decompose_p_v2(
    decompose_p,
    *,
    attributions_v,
):
    if attributions_v is None:
        return False

    if dynamic_obj.decompose_p == "static":
        return decompose_p
    else:
        return dynamic_obj.decompose_p


##
def simple_obj_update(obj, *args, _print_p=False, del_p=False, **kwargs):
    if decomposed_inputs_p(obj):
        assert (len(args) % 2) == 0, "key-values should be balanced!"

        if del_p == "can":
            #: @perf/memory
            del_p = True
            # del_p = False
        else:
            print_diag(f"del_p is false at: {stacktrace_caller_line()}", group="memory")

        d = vars(obj)
        d = dict(d)  #: copies the dict, otherwise it will mutate the obj
        d.update(kwargs)

        if _print_p:
            msg = ""
            if (
                "attributions_v" in kwargs.keys()
                and kwargs["attributions_v"] is not None
            ):
                msg = f"attributions_v mem: {torch_memory_tensor(kwargs['attributions_v'])}GB"

            if "features" in kwargs.keys():
                msg += f", features mem: {torch_memory_tensor(kwargs['features'])}"

            if msg:
                print_diag(msg, group="memory")
                pass

        for i in range(0, len(args), 2):
            value = args[i + 1]
            key = args[i]
            key = key.replace(".", "__")

            d[key] = value

        updated_obj = simple_obj(**d)

        #: @perf/memory
        if del_p:
            if hasattr(obj, "features"):
                del obj.features

            if hasattr(obj, "attributions_v"):
                del obj.attributions_v

            if dynamic_obj.cuda_gc_mode == "y":
                #: @perf/time These are useful! They allowed me to run a model when it otherwise would not run. It might make things slow though.
                # gc.collect() #: The model ran without this line, but it might help anyway?
                torch.cuda.empty_cache()

        return updated_obj
    else:
        if "features" in kwargs:
            return kwargs["features"]
        else:
            #: Sometimes we are calling simple_obj_update solely to store some additional data (e.g., the raw attention weights). We cannot do so when not using a simple_obj in the first place, so we should just return the obj which is probably the features tensor.
            return obj


def decomposed_module_p(obj):
    return hasattr(obj, "DECOMPOSED_P") and obj.DECOMPOSED_P


def nondecomposed_features(inputs, name="features", default="MAGIC_SAME"):
    if decomposed_inputs_p(inputs):
        return inputs[name]
    elif default == "MAGIC_SAME":
        return inputs
    else:
        return default


def nondecomposed_forward(noncompatible_callable, inputs, del_p="can", *args, **kwargs):
    if decomposed_inputs_p(inputs):
        features = inputs.features
        features = noncompatible_callable(features, *args, **kwargs)
        return simple_obj_update(
            inputs,
            features=features,
            del_p=del_p,
        )
    else:
        features = inputs
        return noncompatible_callable(features, *args, **kwargs)


def config_from_inputs(inputs):
    if decomposed_inputs_p(inputs):
        return inputs.decomposition_config
    else:
        return DecompositionConfig(device=None)


##
def residual_decomposed(
    *,
    inputs,
    residuals,
    residual_attributions_v,
    decompose_p=True,
    prefix="",
    warn_nondecomposed=False,
    del_p="can",
):
    if decomposed_inputs_p(inputs):
        decomposition_config = inputs.decomposition_config

        features = inputs.features
        features = residuals + features

        if decompose_p_v2(decompose_p, attributions_v=inputs.attributions_v):
            with no_grad_maybe(decomposition_config.detach_p):
                attributions_v = residual_attributions_v + inputs.attributions_v
        else:
            attributions_v = inputs.attributions_v

        inputs = simple_obj_update(
            inputs,
            #: Storing these will fill up the RAM, perhaps with detach?
            # f'{prefix}attnblk.preres.attrv',
            # inputs.attributions_v,
            # f'{prefix}attnblk.postres.attrv',
            # attributions_v,
            features=features,
            attributions_v=attributions_v,
            del_p=del_p,
        )

        check_attributions_v2(
            inputs,
            prefix=prefix,
            assert_p=decomposition_config.assertion_check_attributions_p,
        )

        return inputs
    else:
        if warn_nondecomposed:
            print_diag(
                f"residual_decomposed: nondecomposed inputs at `{stacktrace_caller_line()}`",
                group="warning.nondecomposed",
            )

        features = inputs
        return features + residuals


##
from pynight.common_icecream import ic, ic_colorize, ic_colorize2


dynamic_obj.print_diag_enabled_groups = [
    "impo",
    "patch_embed_verbose",
    "default",
    "config",
    "sep",  #: separator
    "stats",
    "shape",
    # "LN.shape",
    "indicator",
    "warning",
    "warning.nondecomposed",
    # "check_attributions",
    "check_attributions.end",
    "fullgrad_completeness_check",
    # "fullgrad_completeness_check_success",
    # "progress",
    # "memory",
    # "telegram_send",
    "gradient_mode",
]


def print_diag(
    *args,
    group="default",
    name=None,
    force=False,
    **kwargs,
):
    name = name or __name__

    if "file" in kwargs:
        file = kwargs.pop("file")
    else:
        file = dynamic_get(dynamic_vars, "print_diag_file", default=None)

    if file == "log":
        file = str(Path.home() / f"logs/{name}/all.log")
        # file = str(Path.home() / f"logs/{name}/{group}.log")

    with ExitStack() as stack:
        if isinstance(file, str):
            mkdir(os.path.dirname(file))
            file = stack.enter_context(open(file, "a"))

        if force or (group in dynamic_obj.print_diag_enabled_groups):
            print(*args, file=file, flush=True, **kwargs)

            # if force and file is not None:
            if file is not None:
                print(*args, flush=True, **kwargs)


def print_diag_sep(c="-"):
    return print_diag(c * 40, group="sep")


##
def detach_maybe(tensor, detach_p):
    if detach_p:
        return tensor.detach()
    else:
        tensor


##
def DS_get_from_gbrand(gbrand):
    pattern = f"\\bDS({float_pattern})"
    return rget_percent(gbrand, pattern)


def XSC_get_from_gbrand(gbrand):
    pattern = f"\\bXSC({float_pattern})"
    return rget_percent(gbrand, pattern) or 1


def configure_gradient_modes(
    compact_gbrand=None,
    ig_steps=None,
    gradient_mode_brand="NG",
    patchifier_gbrand=None,
    linear_ds_gbrand=None,
    qkv_ds_gbrand=None,
    mlp_ds_gbrand=None,
    mlp_mul_gbrand=None,
    softmax_mode="S0",
    dynamic_config=None,
    model_name=None,
    normalize_to_unit_vector_gbrand=None,
):
    if dynamic_config is None:
        dynamic_config = dynamic_obj

    dynamic_config.ig_steps = ig_steps

    dynamic_config.normalize_to_unit_vector_gradient_mode = (
        normalize_to_unit_vector_gbrand
    )
    #: used by CLIP

    ##
    #: For TokenTM:
    dynamic_config.post_attn_store_p = True

    #: @duplicateCode/83cf7ddcbd7f6b5ea1b6bf051ff6de14
    dynamic_config.block_output_store_p = True
    dynamic_config.block_output_grad_store_p = True

    dynamic_config.layer_norm_attribute_bias_p = True
    dynamic_config.linear_attribute_bias_p = True

    dynamic_config.raw_attention_store_p = True
    dynamic_config.raw_attention_grad_store_p = True

    # dynamic_config.value_store_p = True
    # dynamic_config.value_grad_store_p = True

    # dynamic_config.mha_store_p = True
    # dynamic_config.mha_grad_store_p = True
    ##

    dynamic_config.fullgrad_completeness_check_p = False

    dynamic_config.softmax_swap_check_p = False

    # dynamic_config.gbrand_forward_checks_p = True
    dynamic_config.gbrand_forward_checks_p = False
    #: whether to check if our gradient-brand modifications have NOT changed the forward values, which they shouldn't

    dynamic_config.attention_gradient_mode = None
    dynamic_config.attention_softmax_gradient_mode = None
    dynamic_config.attention_softmax_competition_scale = 1
    dynamic_config.attention_elementwise_mul_gradient_mode = None

    dynamic_config.layer_norm_gradient_mode = None

    dynamic_config.gelu_gradient_mode = None
    dynamic_config.gelu_attribute_bias_p = False
    #: GELU has no real bias. We add the bias in LineX's Taylor Linearization.

    dynamic_config.linear_dissent_suppression = None
    dynamic_config.conv2d_dissent_suppression = None

    dynamic_config.mlp_linear_dissent_suppression = None
    dynamic_config.mlp_elementwise_mul_gradient_mode = mlp_mul_gbrand
    #: @unused yet, for SwiGlu etc.

    dynamic_config.softmax_gradient_mode = None
    dynamic_config.softmax_competition_scale = 1

    dynamic_config.head_softmax_gradient_mode = None
    dynamic_config.head_softmax_competition_scale = 1

    if compact_gbrand is None:
        compact_gbrand = get_compact_gbrand(
            ig_steps=ig_steps,
            gradient_mode_brand=gradient_mode_brand,
            patchifier_gbrand=patchifier_gbrand,
            linear_ds_gbrand=linear_ds_gbrand,
            qkv_ds_gbrand=qkv_ds_gbrand,
            mlp_ds_gbrand=mlp_ds_gbrand,
            mlp_mul_gbrand=mlp_mul_gbrand,
            softmax_mode=softmax_mode,
            normalize_to_unit_vector_gbrand=normalize_to_unit_vector_gbrand,
        )
    else:
        raise NotImplementedError(
            "compact_gbrand is not None, auto extracting from a compact gbrand has been disabled."
        )

        gradient_mode_brands = compact_gbrand.split(",")
        gradient_mode_brand = gradient_mode_brands[0]
        patchifier_gbrand = list_get(gradient_mode_brands, 1, None)

    ##: Dissent Suppression
    def ds_set(gbrand, target_name):
        if gbrand in ["NormalGrad", "NG", None, "DS0"]:
            pass

        elif gbrand.startswith("DS"):
            dynamic_set(
                dynamic_config,
                target_name,
                DS_get_from_gbrand(gbrand),
            )

        else:
            raise ValueError(f"Unknown ds_gbrand for {target_name}: {gbrand}")

    ds_set(
        gbrand=linear_ds_gbrand,
        target_name="linear_dissent_suppression",
    )

    ds_set(
        gbrand=qkv_ds_gbrand,
        target_name="qkv_dissent_suppression",
    )

    ds_set(
        gbrand=mlp_ds_gbrand,
        target_name="mlp_linear_dissent_suppression",
    )

    ds_set(
        gbrand=patchifier_gbrand,
        target_name="conv2d_dissent_suppression",
    )

    ##

    fgrad_maybe_complete_p = True

    if model_needs_MLP_DU_p(model_name):
        fgrad_maybe_complete_p = mlp_mul_gbrand in ["DU"]

    if model_name_clip_p(model_name):
        fgrad_maybe_complete_p = (
            fgrad_maybe_complete_p
            and normalize_to_unit_vector_gbrand
            in [
                "LineX_1",
                "LX1",
            ]
        )

    if gradient_mode_brand in ["NormalGrad", "NG", None]:
        pass

    elif re.search(r"^NG(-D.)$", gradient_mode_brand):
        #: NG with scaled Attention
        dynamic_config.attention_gradient_mode = None
        dynamic_config.attention_softmax_gradient_mode = None
        dynamic_config.attention_elementwise_mul_gradient_mode = rget(
            gradient_mode_brand, r"NG-(D.)$"
        )

    elif gradient_mode_brand in ["LineX1", "LX1"]:
        dynamic_config.attention_gradient_mode = "LineX_1"
        dynamic_config.layer_norm_gradient_mode = "LineX_1"
        dynamic_config.gelu_gradient_mode = "LineX_ZO"

        dynamic_config.fullgrad_completeness_check_p = fgrad_maybe_complete_p

    elif gradient_mode_brand in ["LX-N_LN"]:
        dynamic_config.attention_gradient_mode = "LineX_1"
        dynamic_config.layer_norm_gradient_mode = None
        dynamic_config.gelu_gradient_mode = "LineX_ZO"

    elif gradient_mode_brand in ["LX-NAB"]:
        #: LineX Without Activation (GELU) but With Bias
        dynamic_config.attention_gradient_mode = "LineX_1"
        dynamic_config.layer_norm_gradient_mode = "LineX_1"
        dynamic_config.gelu_gradient_mode = None
        dynamic_config.gelu_attribute_bias_p = True

        dynamic_config.fullgrad_completeness_check_p = True

    elif gradient_mode_brand in ["LX-NA"]:
        #: LineX Without Activation (GELU)
        dynamic_config.attention_gradient_mode = "LineX_1"
        dynamic_config.layer_norm_gradient_mode = "LineX_1"
        dynamic_config.gelu_gradient_mode = None
        dynamic_config.gelu_attribute_bias_p = False

        dynamic_config.fullgrad_completeness_check_p = True
        dynamic_config.completeness_layers = 0

    elif re.search(r"^LX-AZO(-D.)?$", gradient_mode_brand):
        #: LineX Attention ZO
        dynamic_config.attention_gradient_mode = None
        dynamic_config.attention_softmax_gradient_mode = "ZO"
        dynamic_config.attention_elementwise_mul_gradient_mode = rget(
            gradient_mode_brand, r"AZO-(D.)$"
        )

        dynamic_config.layer_norm_gradient_mode = "LineX_1"
        dynamic_config.gelu_gradient_mode = "LineX_ZO"

        dynamic_config.fullgrad_completeness_check_p = True
        dynamic_config.completeness_layers = 0
        dynamic_config.completeness_atol = 1e-02
        dynamic_config.completeness_rtol = 1e-02
        #: ViT Small: LX-AZO-D2
        #: ic| fullgrad_i_name: 'blocks__9__FGrad_s:sum'
        #: torch.abs(fullgrad_i_sum - grad_target_batch).max(): tensor(0.6233)
        #: ViT Base patch8:
        #: ic| fullgrad_i_name: 'blocks__1__FGrad_s:sum'
        #: torch.abs(fullgrad_i_sum - grad_target_batch).max(): tensor(0.3790)
        # ic| fullgrad_i_name: 'blocks__0__FGrad_s:sum'
        # torch.abs(fullgrad_i_sum - grad_target_batch).max(): tensor(3.1417)
        #: batch 2:
        #: ic| fullgrad_i_name: 'blocks__4__FGrad_s:sum'
        #: torch.abs(fullgrad_i_sum - grad_target_batch).max(): tensor(0.4548)

    elif re.search(r"^LX-AZR(-D.)?$", gradient_mode_brand):
        #: LineX Attention ZR
        dynamic_config.attention_gradient_mode = None
        dynamic_config.attention_softmax_gradient_mode = "ZR"
        dynamic_config.attention_elementwise_mul_gradient_mode = rget(
            gradient_mode_brand, r"AZR-(D.)$"
        )

        dynamic_config.layer_norm_gradient_mode = "LineX_1"
        dynamic_config.gelu_gradient_mode = "LineX_ZO"

        # dynamic_config.fullgrad_completeness_check_p = False
        dynamic_config.fullgrad_completeness_check_p = True
        dynamic_config.completeness_layers = 0

    elif re.search(r"^LX-AD(-D.)?$", gradient_mode_brand):
        #: LineX Attention Detach Denom
        dynamic_config.attention_gradient_mode = None
        dynamic_config.attention_softmax_gradient_mode = "detach_denom_nb"
        dynamic_config.attention_elementwise_mul_gradient_mode = rget(
            gradient_mode_brand, r"AD-(D.)$"
        )

        dynamic_config.layer_norm_gradient_mode = "LineX_1"
        dynamic_config.gelu_gradient_mode = "LineX_ZO"

        # dynamic_config.fullgrad_completeness_check_p = False
        dynamic_config.fullgrad_completeness_check_p = True
        dynamic_config.completeness_layers = 0

    elif re.search(r"^LX-ADB(-D.)?$", gradient_mode_brand):
        #: LineX Attention Detach Denom (With Bias)
        dynamic_config.attention_gradient_mode = None
        dynamic_config.attention_softmax_gradient_mode = "detach_denom"
        dynamic_config.attention_elementwise_mul_gradient_mode = rget(
            gradient_mode_brand, r"ADB-(D.)$"
        )

        dynamic_config.layer_norm_gradient_mode = "LineX_1"
        dynamic_config.gelu_gradient_mode = "LineX_ZO"

        # dynamic_config.fullgrad_completeness_check_p = False
        dynamic_config.fullgrad_completeness_check_p = True
        dynamic_config.completeness_layers = 0
        # dynamic_config.completeness_atol = 1e-02
        # dynamic_config.completeness_rtol = 1e-02

        # FullGrad is complete:
        # ic| fullgrad_i_name: 'blocks__0__FGrad_s:sum'
        # torch.abs(fullgrad_i_sum - grad_target_batch).max(): tensor(2.0027e-05)
        #: @old/biasScalingBug Even though we expected this to be complete, it isn't. Perhaps because there are too many bias terms.
        #: ic| fullgrad_i_name: 'blocks__11__FGrad_s:sum'
        #: torch.abs(fullgrad_i_sum - grad_target_batch).max(): tensor(1.0792)
        #: ic| bam_name: 'BAM_VisionTransformer.blocks.2.attn.attn_softmax_s:raw'
        #: raw.shape: torch.Size([5, 785, 9420])
        #: channel_mixed.shape: torch.Size([5, 785])

    elif gradient_mode_brand in ["LX-AAG"]:
        dynamic_config.attention_gradient_mode = "AAG"
        dynamic_config.layer_norm_gradient_mode = "LineX_1"
        dynamic_config.gelu_gradient_mode = "LineX_ZO"

    elif gradient_mode_brand in ["LX-Q"]:
        dynamic_config.attention_gradient_mode = "Q"
        dynamic_config.layer_norm_gradient_mode = "LineX_1"
        dynamic_config.gelu_gradient_mode = "LineX_ZO"

    elif gradient_mode_brand in ["LX-K"]:
        dynamic_config.attention_gradient_mode = "K"
        dynamic_config.layer_norm_gradient_mode = "LineX_1"
        dynamic_config.gelu_gradient_mode = "LineX_ZO"

    elif gradient_mode_brand in ["LX-NoAct-Q"]:
        dynamic_config.attention_gradient_mode = "Q"
        dynamic_config.layer_norm_gradient_mode = "LineX_1"
        dynamic_config.gelu_gradient_mode = None
        dynamic_config.gelu_attribute_bias_p = True

    elif gradient_mode_brand in ["LX-NoAct-K"]:
        dynamic_config.attention_gradient_mode = "K"
        dynamic_config.layer_norm_gradient_mode = "LineX_1"
        dynamic_config.gelu_gradient_mode = None
        dynamic_config.gelu_attribute_bias_p = True

    elif gradient_mode_brand in ["LX-Q-Only"]:
        dynamic_config.attention_gradient_mode = "Q"
        dynamic_config.layer_norm_gradient_mode = None
        dynamic_config.gelu_gradient_mode = None

    elif gradient_mode_brand in ["LX-K-Only"]:
        dynamic_config.attention_gradient_mode = "K"
        dynamic_config.layer_norm_gradient_mode = None
        dynamic_config.gelu_gradient_mode = None

    elif gradient_mode_brand in ["LXA"]:
        #: LineX Attn-Only
        dynamic_config.attention_gradient_mode = "LineX_1"
        dynamic_config.layer_norm_gradient_mode = None
        dynamic_config.gelu_gradient_mode = None
        dynamic_config.gelu_attribute_bias_p = False

    # elif gradient_mode_brand in ["LX-GA"]:
    #     dynamic_config.attention_gradient_mode = None
    #     dynamic_config.layer_norm_gradient_mode = "LineX_1"
    #     dynamic_config.gelu_gradient_mode = "LineX_ZO"
    elif re.search(r"^LX-GA(-D.)?$", gradient_mode_brand):
        #: LineX, NG Attention
        dynamic_config.attention_gradient_mode = None
        dynamic_config.attention_softmax_gradient_mode = None
        dynamic_config.attention_elementwise_mul_gradient_mode = rget(
            gradient_mode_brand, r"GA-(D.)$"
        )

        dynamic_config.layer_norm_gradient_mode = "LineX_1"

        dynamic_config.gelu_gradient_mode = "LineX_ZO"

        # dynamic_config.fullgrad_completeness_check_p = False
        dynamic_config.fullgrad_completeness_check_p = True
        dynamic_config.completeness_layers = 0

    #: Allen is the same as Gate
    # elif re.search(r"^Allen(-D.)?$", gradient_mode_brand):
    #     #: LineX, GA2 Attention, No LN
    #     dynamic_config.attention_gradient_mode = None
    #     dynamic_config.attention_softmax_gradient_mode = None
    #     dynamic_config.attention_elementwise_mul_gradient_mode = rget(
    #         gradient_mode_brand, r"Allen-(D.)$"
    #     )
    #     # assert dynamic_config.attention_elementwise_mul_gradient_mode

    #     dynamic_config.layer_norm_gradient_mode = None

    #     dynamic_config.gelu_gradient_mode = "LineX_ZO"

    #     # dynamic_config.fullgrad_completeness_check_p = False
    #     dynamic_config.fullgrad_completeness_check_p = True
    #     dynamic_config.completeness_layers = 0

    elif re.search(r"^Gate(-D.)?$", gradient_mode_brand):
        #: Only detach the activation gate
        dynamic_config.attention_gradient_mode = None
        dynamic_config.attention_softmax_gradient_mode = None
        dynamic_config.attention_elementwise_mul_gradient_mode = rget(
            gradient_mode_brand, r"Gate-(D.)$"
        )

        dynamic_config.layer_norm_gradient_mode = None

        dynamic_config.gelu_gradient_mode = "LineX_ZO"

    elif re.search(rf"^Gate-XSC{re_maybe(float_pattern)}(-D.)?$", gradient_mode_brand):
        #: Detach the gate, and use XSC gradient for the attention.
        dynamic_config.attention_gradient_mode = None
        dynamic_config.attention_softmax_gradient_mode = "XSC"
        dynamic_config.attention_softmax_competition_scale = XSC_get_from_gbrand(
            gradient_mode_brand
        )
        dynamic_config.attention_elementwise_mul_gradient_mode = rget(
            gradient_mode_brand, rf"XSC{re_maybe(float_pattern)}-(D.)$"
        )
        # assert dynamic_config.attention_elementwise_mul_gradient_mode

        dynamic_config.layer_norm_gradient_mode = None

        dynamic_config.gelu_gradient_mode = "LineX_ZO"

    # elif gradient_mode_brand in ["LX-LN"]:
    #     #: LineX LayerNorm-Only
    #     dynamic_config.attention_gradient_mode = None
    #     dynamic_config.layer_norm_gradient_mode = "LineX_1"
    #     dynamic_config.gelu_gradient_mode = None
    #     dynamic_config.gelu_attribute_bias_p = False
    #
    elif re.search(r"^LN(-D.)?$", gradient_mode_brand):
        #: Only the LayerNorm
        dynamic_config.attention_gradient_mode = None
        dynamic_config.attention_softmax_gradient_mode = None
        dynamic_config.attention_elementwise_mul_gradient_mode = rget(
            gradient_mode_brand, r"Gate-(D.)$"
        )

        dynamic_config.layer_norm_gradient_mode = "LineX_1"

        dynamic_config.gelu_gradient_mode = None

        dynamic_config.fullgrad_completeness_check_p = False
        # dynamic_config.fullgrad_completeness_check_p = True
        # dynamic_config.completeness_layers = 0

    elif re.search(rf"^LX-XSC{re_maybe(float_pattern)}(-D.)?$", gradient_mode_brand):
        #: LineX, XSC Attention
        dynamic_config.attention_gradient_mode = None
        dynamic_config.attention_softmax_gradient_mode = "XSC"
        dynamic_config.attention_softmax_competition_scale = XSC_get_from_gbrand(
            gradient_mode_brand
        )
        dynamic_config.attention_elementwise_mul_gradient_mode = rget(
            gradient_mode_brand, rf"XSC{re_maybe(float_pattern)}-(D.)$"
        )

        dynamic_config.layer_norm_gradient_mode = "LineX_1"

        dynamic_config.gelu_gradient_mode = "LineX_ZO"

        # dynamic_config.fullgrad_completeness_check_p = False
        dynamic_config.fullgrad_completeness_check_p = True
        dynamic_config.completeness_layers = 0

    elif gradient_mode_brand in ["GlobALTI"]:
        assert softmax_mode == "S0"
        assert patchifier_gbrand is None
        assert linear_ds_gbrand is None
        assert mlp_ds_gbrand is None
        assert mlp_mul_gbrand is None

        ##: @duplicateCode/83cf7ddcbd7f6b5ea1b6bf051ff6de14
        dynamic_config.block_output_store_p = True

        dynamic_config.raw_attention_store_p = False
        dynamic_config.raw_attention_grad_store_p = False
        dynamic_config.block_output_grad_store_p = False
        dynamic_config.layer_norm_attribute_bias_p = False
        dynamic_config.linear_attribute_bias_p = False
        dynamic_config.gelu_attribute_bias_p = False
        ##

    else:
        raise ValueError(f"Unknown gradient_mode_brand: {gradient_mode_brand}")

    ##
    dynamic_config.head_contrastive_n = None
    dynamic_config.head_contrastive_scale = None

    if softmax_mode in ["S1"]:
        dynamic_config.backward_softmax_p = True

        if dynamic_config.completeness_layers != 0:
            #: 0 means just printing the non-completeness info.

            dynamic_config.fullgrad_completeness_check_p = False

    elif softmax_mode in ["S0"]:
        dynamic_config.backward_softmax_p = False

    elif re.search(rf"^XSC{re_maybe(float_pattern)}$", softmax_mode or ""):
        dynamic_config.backward_softmax_p = True
        dynamic_config.head_softmax_gradient_mode = "XSC"
        dynamic_config.head_softmax_competition_scale = XSC_get_from_gbrand(
            softmax_mode
        )

        if dynamic_config.completeness_layers != 0:
            #: 0 means just printing the non-completeness info.

            dynamic_config.fullgrad_completeness_check_p = False

    elif re.search(rf"""^C\d+_{float_pattern}$""", softmax_mode or ""):
        dynamic_config.backward_softmax_p = False
        dynamic_config.head_contrastive_n = int(rget(softmax_mode, rf"""^C(\d+)"""))
        dynamic_config.head_contrastive_scale = float(
            rget(softmax_mode, rf"""^C\d+_({float_pattern})$""")
        )

    else:
        raise ValueError(f"Unknown softmax_mode: {softmax_mode}")
    ##

    ##
    from decompv.early_boot import (
        run_check_completeness_mode_p,
        run_compute_completeness_mode_p,
    )

    dynamic_config.completeness_layers = 0
    #: disable the checks, use run_check_completeness_mode_p for sanity checks instead

    if run_check_completeness_mode_p or run_compute_completeness_mode_p:
        dynamic_config.fullgrad_completeness_check_p = True

        if run_compute_completeness_mode_p:
            dynamic_config.completeness_layers = 0

        else:
            dynamic_config.completeness_layers = None

    ##

    return simple_obj(
        compact_gbrand=compact_gbrand,
        ig_steps=ig_steps,
        gradient_mode_brand=gradient_mode_brand,
        patchifier_gbrand=patchifier_gbrand,
        softmax_mode=softmax_mode,
        linear_ds_gbrand=linear_ds_gbrand,
        qkv_ds_gbrand=qkv_ds_gbrand,
        mlp_ds_gbrand=mlp_ds_gbrand,
        mlp_mul_gbrand=mlp_mul_gbrand,
        normalize_to_unit_vector_gbrand=normalize_to_unit_vector_gbrand,
    )


##
def mean_normalize_last_dim(tensor, positive_only=False):
    if positive_only:
        tensor_f = F.relu(tensor, inplace=False)
    else:
        tensor_f = tensor

    sum_last_dim = tensor_f.sum(dim=-1, keepdim=True)

    normalized_tensor = tensor / sum_last_dim

    return normalized_tensor


def multiply_attributions(attributions_v):
    return torch.prod(torch.prod(attributions_v, dim=-2), dim=-2)


def check_attributions_v2(
    inputs,
    prefix="",
    assert_p=False,
    group="check_attributions",
    **kwargs,
):
    if not decomposed_inputs_p(inputs) or inputs.attributions_v is None:
        return None

    if assert_p == "skip_on_GPU":
        device = inputs.decomposition_config.device
        assert_p = device == "cpu" or (hasattr(device, "type") and device.type == "cpu")
        #: @hallucination? =device.type=

    if assert_p:
        group = "impo"

    if (not assert_p) and (group not in dynamic_obj.print_diag_enabled_groups):
        return

    with torch.no_grad():
        check_res = check_attributions(
            inputs.features,
            inputs.attributions_v,
            **kwargs,
        )
        allclose = check_res.allclose
        err = check_res.err
        reconstructed = check_res.reconstructed

        originals = inputs.features
        originals_abs = torch.abs(originals)
        originals_abs_mean = torch.mean(originals_abs)
        originals_abs_max = torch.max(originals_abs)

        err_abs = torch.abs(err)
        err_abs_mean = torch.mean(err_abs)
        err_abs_max = torch.max(err_abs)

        cosine_sim = F.cosine_similarity(originals, reconstructed, dim=-1)
        cosine_sim_mean = torch.mean(cosine_sim)
        cosine_sim_max = torch.max(cosine_sim)
        cosine_sim_min = torch.min(cosine_sim)

        msg = f"""
{prefix}check_attributions:
    allclose = {allclose}

    originals.shape = {originals.shape}
    reconstructed.shape = {reconstructed.shape}
    err.shape = {err.shape}

    cosine_sim.shape = {cosine_sim.shape}
    cosine_sim_mean = {cosine_sim_mean}
    cosine_sim_max = {cosine_sim_max}
    cosine_sim_min = {cosine_sim_min}

    err_abs_mean = {err_abs_mean}
    err_abs_max = {err_abs_max}

    originals_abs_mean = {originals_abs_mean}
    originals_abs_max = {originals_abs_max}
"""

        print_diag(msg, group=group)

        if assert_p:
            assert allclose, msg


def check_attributions(
    originals,
    attributions_v,
    reconstructor=None,
    atol=5e-2,
    print_all=False,
    **kwargs,
):
    reconstructor = reconstructor or sum_attributions

    reconstructed = reconstructor(attributions_v)

    if print_all:
        print_diag(f"originals: {originals}")
        print_diag(f"reconstructed: {reconstructed}")

    err = reconstructed - originals

    return simple_obj(
        allclose=torch.allclose(originals, reconstructed, atol=atol, **kwargs),
        err=err.detach(),
        reconstructed=reconstructed,
    )


def attributions_n_get(attributions_v, index, zero_out_error=False):
    if zero_out_error:
        attributions_v = attributions_v.clone()
        attributions_v[..., -1, 0, :] = 0  #: error token
        attributions_v[..., 0, 0, :] = 0  #: CLS

    attr_n = attributions_normify(attributions_v)
    attr_n_i = attr_n[:, index, :]
    return attr_n_i


def attribute_patches_aggregate(
    inputs, mode="reset", start_p=False, prefix="", warn_nondecomposed=False
):
    if decomposed_inputs_p(inputs):
        decomposition_config = inputs.decomposition_config

        with no_grad_maybe(decomposition_config.detach_p):
            block_i = rget(prefix, "^blocks\.(\d+)")
            if block_i is not None:
                block_i = int(block_i)

            mode_orig = mode
            mode_i = rget(mode, "^vector_f(\d+)")
            if mode_i is None:
                if start_p:
                    mode = "reset"
            else:
                if start_p:
                    mode = "none"
                else:
                    mode_i = int(mode_i)
                    assert block_i is not None

                    if block_i > mode_i:
                        mode = "vector"
                    elif block_i == mode_i:
                        mode = "reset"
                    else:
                        # mode = 'reset'
                        mode = "none"

            print_diag(
                f"{prefix}attribute_patches_aggregate: mode={mode}, mode_orig={mode_orig}",
                group="config",
                # force=True,
            )

            if mode == "vector":
                return inputs
            elif mode == "none":
                return simple_obj_update(
                    inputs,
                    attributions_v=None,
                    del_p="can",
                )
            elif mode == "reset":
                #: @testWith
                #: features = torch.randn(1, 3, 4)
                #: features = torch.randn(2, 3, 2)
                ##
                features = inputs.features

                if True:
                    #: This version is more efficient and takes ~20ms.
                    ##
                    attributions_v_shape = list(features.shape)
                    #: (batch, token, hidden)

                    token_count = attributions_v_shape[-2]
                    attribution_token_count = token_count + 1
                    #: plus one for the error/bias token

                    attributions_v_shape = (
                        attributions_v_shape[:-1]
                        + [
                            attribution_token_count,
                            1,
                        ]
                        + attributions_v_shape[-1:]
                    )
                    #: (batch, token_to, token_from+1, 1, hidden)

                    tmp = torch.zeros(
                        attributions_v_shape,
                        device=decomposition_config.device,
                    )

                    tmp[
                        ..., torch.arange(token_count), torch.arange(token_count), 0, :
                    ] = features
                    #: PyTorch: Does assigning a slice to a slice of another tensor copy it?
                    #: My tests show that, yes, a copy happens.

                    attributions_v = tmp
                else:
                    #: This version took ~1.75s in base_224_patch16 for GlobEnc!
                    ##
                    attributions_v = features.clone()

                    attributions_v_shape = torch.tensor(attributions_v.shape)

                    token_count = attributions_v_shape[-2]
                    attribution_token_count = token_count + 1
                    #: plus one for the error/bias token

                    attributions_v_shape = torch.cat(
                        (
                            attributions_v_shape[:-1],
                            torch.tensor(
                                [
                                    attribution_token_count,
                                    1,
                                ]
                            ),
                            attributions_v_shape[-1:],
                        ),
                        dim=0,
                    )
                    attributions_v_shape = attributions_v_shape.tolist()
                    # print_diag(f"attributions_v_shape: {ic_colorize2(attributions_v_shape)}", group="shape")

                    #: ... token, token_attribution, pixel_attribution, hidden dim
                    tmp = torch.zeros(
                        attributions_v_shape,
                        device=decomposition_config.device,
                    )
                    for i in range(token_count):
                        tmp[..., i, i, 0, :] = attributions_v[..., i, :]
                    attributions_v = tmp

                # print_diag(f"attributions_v.shape: {ic_colorize2(attributions_v.shape)}", group="shape")

                inputs = simple_obj_update(
                    inputs,
                    attributions_v=attributions_v,
                    del_p="can",
                )

                check_attributions_v2(
                    inputs,
                    prefix=f"{prefix}attribute_patches_aggregate.",
                    assert_p=decomposition_config.assertion_check_attributions_p,
                )

                return inputs
            else:
                raise ValueError(f"Mode not recognized: {mode}")
    else:
        if warn_nondecomposed:
            print_diag(
                f"attribute_patches_aggregate: nondecomposed inputs at `{stacktrace_caller_line()}`",
                group="warning.nondecomposed",
            )

        return inputs


##
@dataclass
class DecompositionConfig:
    device: str

    name: str = "DecompV"

    pre_norm_include_p: bool = False  #: @useless?

    detach_p: bool = True

    assertion_check_attributions_p: bool = False
    # assertion_check_attributions_p: bool = True

    assertion_check_attributions_end_p: bool = False
    # assertion_check_attributions_end_p: bool = True
    # assertion_check_attributions_end_p: str = 'skip_on_GPU'

    assertion_check_attention_value_p: bool = False

    bias_decomposition_mode: str = "error_token"
    gelu_bias_decomposition_mode: str = "error_token"
    gelu_decompose_p: bool = True

    softmax_decompose_p: bool = True

    layer_norm_decompose_p: bool = True

    linear_decompose_p: bool = True
    mlp_decompose_p: bool = True

    layerscale_decompose_p: bool = True

    attention_decompose_p: bool = True

    residual1_decompose_p: bool = True
    residual2_decompose_p: bool = True

    attributions_aggregation_strategy: str = "vector"
    # attributions_aggregation_strategy: str = 'reset'

    # GELU_decompose_mode: str = 'taylor'
    GELU_decompose_mode: str = "zo"

    save_intermediate_p: bool = False

    token_mixer_decompose_p: bool = True


##
def store_grad_and_bias_attributions(
    grad,
    *,
    store_in,
    bias,
    layer_name,
    sum_dim=None,
):
    # ic(bias.shape, torch_shape_get(grad))

    store_bias_attributions_backcompat_p = True
    #: The sum-channel-mix of the raw version can be computed elsewhere, no need to compute and store it here. But to preserve backcompat, we have retained this feature.

    # if 'softmax' in layer_name:
    #     ic(layer_name, sum_dim, grad.shape, bias.shape)

    if sum_dim is not None:
        grad = flatten_and_move_to_last(grad, dim=sum_dim)
        bias = flatten_and_move_to_last(bias, dim=sum_dim)

    # if 'softmax' in layer_name:
    #     ic(layer_name, sum_dim, grad.shape, bias.shape)

    store_in.stored_out_grad = grad
    # store_in.stored_out_bias = bias.detach().cpu()

    if bias is not None:
        store_in.stored_bias_attributions_raw = torch.mul(bias, grad)

        if store_bias_attributions_backcompat_p:
            store_in.stored_bias_attributions = (
                store_in.stored_bias_attributions_raw.sum(dim=-1)
            )
            # store_in.stored_bias_attributions = torch.einsum(
            #     "...d,...d->...",
            #     bias,
            #     grad,
            # )

    if dynamic_obj.get(
        f"{layer_name}_stored_detach_p",
        default=True,
    ):
        store_in.stored_out_grad = store_in.stored_out_grad.detach().cpu()

        if bias is not None:
            # store_in.stored_out_bias = store_in.stored_out_bias.detach().cpu()

            store_in.stored_bias_attributions_raw = (
                store_in.stored_bias_attributions_raw.detach().cpu()
            )

            if store_bias_attributions_backcompat_p:
                store_in.stored_bias_attributions = (
                    store_in.stored_bias_attributions.detach().cpu()
                )


##
def ln_decomposer(
    *,
    inputs,
    weight,
    bias,
    eps,
):
    ##
    #: @warning Does NOT respect layer_norm_decompose_p! It always decomposes. LayerNormDecomposed itself won't call this if not layer_norm_decompose_p.
    ##
    # LN: features.shape: torch.Size([1, 197, 768])
    # LN: attributions_v.shape: torch.Size([1, 197, 198, 1, 768])
    # LN: mean.shape: torch.Size([1, 197, 1])
    # LN: var.shape: torch.Size([1, 197, 1, 1, 1])
    # LN: each_mean.shape: torch.Size([1, 197, 198, 1, 1])
    ##
    decomposition_config = inputs.decomposition_config

    with no_grad_maybe(decomposition_config.detach_p):
        attributions_v = inputs.attributions_v
        features = inputs.features
        bias_decomposition_mode = decomposition_config.bias_decomposition_mode

        print_diag(
            f"LN: features.shape: {ic_colorize2(features.shape)}", group="LN.shape"
        )
        print_diag(
            f"LN: attributions_v.shape: {ic_colorize2(attributions_v.shape)}",
            group="LN.shape",
        )

        mean = features.mean(-1, keepdim=True)
        # (batch, seq_len, 1) m(y=Σy_j)
        print_diag(f"LN: mean.shape: {ic_colorize2(mean.shape)}", group="LN.shape")

        var = (
            (features - mean)
            .pow(2)
            .mean(-1, keepdim=True)
            .unsqueeze(dim=2)
            .unsqueeze(dim=2)
        )
        # (batch, seq_len, 1, 1, 1)  s(y)
        print_diag(f"LN: var.shape: {ic_colorize2(var.shape)}", group="LN.shape")

        each_mean = attributions_v.mean(-1, keepdim=True)
        # (batch, seq_len, seq_len, pixel_len, 1) m(y_j)
        print_diag(
            f"LN: each_mean.shape: {ic_colorize2(each_mean.shape)}", group="LN.shape"
        )

        normalized_layer = torch.div(
            attributions_v - each_mean, (var + eps) ** (1 / 2)
        )  # (batch, seq_len, seq_len, pixel_len, all_head_size)

        attributions_v = torch.einsum(
            "...d,d->...d", normalized_layer, weight
        )  # (batch, seq_len, seq_len, pixel_len, all_head_size)

        attributions_v = bias_decomposer(
            attributions_v,
            bias,
            bias_decomposition_mode=bias_decomposition_mode,
        )

        return attributions_v


def bias_decomposer(attributions_v, bias, bias_decomposition_mode):
    """bias_decomposition_mode has priority over decomposition_config"""
    ##
    if bias_decomposition_mode == "error_token":
        # attributions_v = attributions_v.clone()
        #: @perf can disable the above to save on memory

        attributions_v[..., -1, 0, :] += bias
        #: @question Is this inplace operation bad?

        return attributions_v
        # return simple_obj_update(inputs, attributions_v=post_ln_layer)

    elif bias_decomposition_mode == "ignore":
        return attributions_v
    else:
        raise NotImplementedError(f"bias_decomposition_mode: {bias_decomposition_mode}")


class LayerNormDecomposed(nn.LayerNorm):
    def __init__(
        self,
        *args,
        **kwargs,
        # normalized_shape,
        # eps: float = 1e-5,
        # elementwise_affine: bool = True,
        # device=None,
        # dtype=None,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
            # normalized_shape=normalized_shape,
            # eps=eps,
            # elementwise_affine=elementwise_affine,
            # device=device,
            # dtype=dtype,
        )
        # self.prefix = 'unset.'
        self.prefix = f"unset.{self.__class__.__name__}"

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.prefix = prefix
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, inputs, warn_nondecomposed=False):
        # print_diag('LayerNormDecomposed forward', group='indicator')

        gradient_mode = dynamic_obj.layer_norm_gradient_mode
        print_diag(
            f"{self.prefix}: gradient_mode: {gradient_mode}", group="gradient_mode"
        )

        if decomposed_inputs_p(inputs):
            decomposition_config = inputs.decomposition_config

            if decompose_p_v2(
                decomposition_config.layer_norm_decompose_p,
                attributions_v=inputs.attributions_v,
            ):
                attributions_v = ln_decomposer(
                    inputs=inputs, weight=self.weight, bias=self.bias, eps=self.eps
                )
            else:
                attributions_v = inputs.attributions_v

            features = super().forward(inputs.features)

            inputs = simple_obj_update(
                inputs,
                features=features,
                attributions_v=attributions_v,
                del_p="can",
            )

            check_attributions_v2(
                inputs,
                prefix=self.prefix,
                assert_p=decomposition_config.assertion_check_attributions_p,
            )

            return inputs
        else:
            if warn_nondecomposed:
                print_diag(
                    f"LayerNormDecomposed: nondecomposed inputs at `{stacktrace_caller_line()}`",
                    group="warning.nondecomposed",
                )

            features = inputs

            ##
            mean = features.mean(-1, keepdim=True)
            features_mean = features - mean
            var = (features_mean).pow(2).mean(-1, keepdim=True)

            if gradient_mode == "LineX_1":
                var = var.detach()

            normalized_layer = torch.div(features_mean, (var + self.eps) ** (1 / 2))

            result = torch.einsum("...d,d->...d", normalized_layer, self.weight)
            result += self.bias
            ##

            return_value = None
            if True:
                #: We had small divergence between =result= and =result_orig=, e.g.,:
                #:   torch.max(torch.abs(result_orig - result)).item(): 4.76837158203125e-07
                #:
                #: So I have added the following to absolutely make sure that we do NOT change the network in the process of attributing it.
                ##
                result_orig = super().forward(features)

                if gradient_mode:
                    result_orig_correct_grad = result_orig.detach() + (
                        result - result.detach()
                    )
                    #: This has the same data as =result_orig= but with the gradient information of =result=. It's a way to manipulate the gradients without affecting the forward pass values.

                    # assert torch.allclose(result_orig_correct_grad, result_orig)
                else:
                    result_orig_correct_grad = result_orig
                    #: We have not manipulated the gradients, so there is no need to do anything here.

                return_value = result_orig_correct_grad
            else:
                debug_p = False
                if debug_p or (
                    dynamic_obj.layer_norm_check_nondecomposed_forward_p
                    and dynamic_obj.gbrand_forward_checks_p
                ):
                    result_orig = super().forward(features)

                    # ic(result_orig.shape, result.shape)

                    layer_norm_allclose_p = torch.allclose(result_orig, result)

                    # ic(layer_norm_allclose_p)

                    if not layer_norm_allclose_p:
                        ic(
                            layer_norm_allclose_p,
                            torch.max(torch.abs(result_orig - result)).item(),
                            torch.max(torch.abs(result_orig - result))
                            / torch.max(torch.abs(result_orig)).item(),
                            torch.max(torch.abs(result_orig - result))
                            / torch.max(torch.abs(result)).item(),
                        )

                    assert layer_norm_allclose_p

                return_value = result

            ##
            layer_name = "layer_norm"
            #: @duplicateCode/6b32d3a832249d36819faf9531ae5ed1
            delattr_force(self, "stored_out_grad")
            delattr_force(self, "stored_bias_attributions")
            delattr_force(self, "stored_bias_attributions_raw")
            if dynamic_obj.get(f"{layer_name}_attribute_bias_p"):
                tensor_register_hook(
                    return_value,
                    partial(
                        store_grad_and_bias_attributions,
                        bias=self.bias,
                        store_in=self,
                        layer_name=layer_name,
                    ),
                )
            ##

            return return_value


class FairOCLN(LayerNormDecomposed):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype

        ##
        x = super().forward(x)
        # x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        ##

        return x.to(orig_type)


##
def attributions_scalarify_v2(
    *args,
    **kwargs,
):
    if "sum_dim" not in kwargs:
        kwargs = dict(kwargs)

        kwargs["sum_dim"] = None

    return attributions_scalarify(*args, **kwargs)


def attributions_scalarify(
    attributions,
    *,
    sum_dim=(-2,),
    mode=None,
    dim=-1,
    ord=2,
):
    """
    Aggregates the vector attributions of tokens into scalar attributions
    """
    mode = mode or "norm"

    if sum_dim is not None:
        attributions = sum_attributions(attributions, sum_dim=sum_dim)

        print_diag(
            f"attributions_scalarify (after summing): attributions.shape: {ic_colorize2(attributions.shape)}",
            group="shape.attributions_scalarify",
        )

    if mode == "norm":
        return torch.linalg.vector_norm(attributions, dim=dim, ord=ord, keepdim=False)
    elif mode == "L1":
        return torch.linalg.vector_norm(attributions, dim=dim, ord=1, keepdim=False)
    elif mode == "L2":
        return torch.linalg.vector_norm(attributions, dim=dim, ord=2, keepdim=False)
    elif mode == "L4":
        return torch.linalg.vector_norm(attributions, dim=dim, ord=4, keepdim=False)
    elif mode == "LInf":
        return torch.linalg.vector_norm(
            attributions, dim=dim, ord=math.inf, keepdim=False
        )
    elif mode == "sum":
        return torch.sum(attributions, dim=dim, keepdim=False)
    elif mode == "RS":
        return torch.sum(F.relu(attributions), dim=dim, keepdim=False)
    elif mode == "NS":
        #: ReLU-Negative Sum
        return torch.sum(F.relu(-1 * attributions), dim=dim, keepdim=False)
    elif mode in [
        "QS25",
        "QS50",
        "QS75",
    ]:
        quantile_map = {
            "QS25": 0.25,
            "QS50": 0.5,
            "QS75": 0.75,
        }

        quantile = quantile_map[mode]
        return quantile_sum(
            attributions,
            quantile=quantile,
            dim=dim,
            keepdim=False,
            greater_than_p=True,
        )
    elif mode == "identity":
        return attributions
    else:
        raise ValueError(f"Unsupported mode: {mode}")


attributions_normify = attributions_scalarify


def sum_attributions(attributions_v, sum_dim=(-2, -3)):
    # Check if all dimensions to be summed are of size 1
    if all(attributions_v.size(dim) == 1 for dim in sum_dim):
        return torch.squeeze(attributions_v, dim=sum_dim)

    #: Our attributions have two dimensions for the sources, (token, extra_source).
    return torch.sum(attributions_v, dim=sum_dim, keepdim=False)


def sum_attributions_extra(attributions_v):
    #: Our attributions have two dimensions for the sources, (token, extra_source). This sums over all extra sources:
    return sum_attributions(attributions_v, sum_dim=(-2,))


##
@tensorify_scalars(argnums=(0,))
def h_sigmoid1(transparency_value, s=2):
    # Scaling the input to amplify the effect near zero
    new_value = F.sigmoid(transparency_value * (2 * s) - s)
    return new_value


def h_exponent(transparency_value, exponent=0.5):
    new_value = pow(transparency_value, exponent)
    return new_value


# result_np = (np.array(result_image)).transpose(2, 0, 1).astype("float32") / 255
# return result_np


def vis_normatt_vs_rawatt(
    image,
    inputs,
    block_index=11,
    block_what="__attnblk__preres__attrv",
    zero_out_cls=True,
    **kwargs,
):
    cls_attr_n_unnormalized = attributions_n_get(
        inputs[f"blocks__{block_index}{block_what}"],
        0,
        zero_out_error=True,
    ).squeeze()

    rawatt = inputs[f"blocks__{block_index}__attn__rawattn"][0, :, 0, :].squeeze()
    rawatt = torch.mean(rawatt, dim=0)

    # ic('after', rawatt.shape)
    # rawatt.shape: torch.Size([12, 197])
    # 'after', rawatt.shape: torch.Size([197])
    #: batch, head, seq, attention_weights

    return vis_attr(
        image=image,
        first_attr=cls_attr_n_unnormalized,
        second_attr=rawatt,
        **kwargs,
    )


def vis_attr(
    image,
    first_attr,
    second_attr=None,
    blend_mode="minus",
    zero_out_cls=True,
    plot_first_alone_p=False,
    first_export_dir=None,
    title="",
    first_title="",
    export_dir=None,
    **kwargs,
):
    first_attr = first_attr.cpu()
    first_attr = first_attr.squeeze()

    if zero_out_cls:
        first_attr[0] = 0

    # ic(torch.sum(first_attr))

    if plot_first_alone_p and second_attr is not None:
        overlay_colored_grid(
            image,
            first_attr,
            title=first_title,
            export_dir=first_export_dir,
            **kwargs,
        )

    if second_attr is not None:
        second_attr = second_attr.cpu()
        second_attr = second_attr.squeeze()

        if zero_out_cls:
            second_attr[0] = 0

        # ic(blend_mode)
        if blend_mode == "minus":
            ##
            #: normalizing is quite complex; logarithm-scale values sum to a negative value, so we cannot divide by their sum (this will change their sign!)
            #
            # ic(first_attr[:20], second_attr[:20])

            first_attr[(len(second_attr)) :] = 0.0
            first_attr[: (len(second_attr))] = mean_normalize_last_dim(
                first_attr[: (len(second_attr))], positive_only=True
            )

            assert (
                torch.min(second_attr) >= 0
            )  #: if negative, needs some other normalization strategy
            second_attr = mean_normalize_last_dim(second_attr)

            # ic(first_attr[:20], second_attr[:20])
            ##

            first_attr = first_attr[: (len(second_attr))] - second_attr
        elif blend_mode == "second":
            first_attr[: (len(second_attr))] = second_attr
            first_attr[(len(second_attr)) :] = 0.0
        else:
            raise ValueError(f"blend_mode not supported: {blend_mode}")

    # ic(torch.sum(first_attr))

    return overlay_colored_grid(
        image,
        first_attr,
        title=title,
        export_dir=export_dir,
        **kwargs,
    )


###
def fair_model_get(model):
    from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

    #: [[https://github.com/pytorch/pytorch/blob/9df4bc6a0dc72caccee142555d1668fad1621206/torch/nn/modules/linear.py#L136][pytorch/torch/nn/modules/linear.py at main · pytorch/pytorch]]

    import open_clip.transformer

    from timm.models.decomposition import (
        LinearDecomposed,
        LayerNormDecomposed,
        NightConv2d,
        GELUDecomposed,
        NightSoftmax,
        FairOCLN,
        FairOCAttention,
        FairMultiheadAttention,
        FairSiLU,
        FairSwiGLU,
        FairSwiGLUPacked,
        FairGluMlp,
    )

    module_mapping = {
        nn.Linear: LinearDecomposed,
        NonDynamicallyQuantizableLinear: LinearDecomposed,
        timm.layers.norm.LayerNorm: LayerNormDecomposed,
        nn.LayerNorm: LayerNormDecomposed,
        open_clip.transformer.LayerNorm: FairOCLN,
        # open_clip.transformer.Attention: FairOCAttention,
        nn.MultiheadAttention: FairMultiheadAttention,
        nn.Conv2d: NightConv2d,
        nn.GELU: GELUDecomposed,
        nn.SiLU: FairSiLU,
        nn.Softmax: NightSoftmax,
        # timm.layers.GluMlp: FairGluMlp,
        # timm.layers.SwiGLU: FairSwiGLU,
    }

    result = module_mapper(model, module_mapping)
    return result


def create_model(
    device,
    model_name="vit_base_patch16_224.augreg_in21k_ft_in1k",
    **kwargs,
):
    ic(model_name)

    dino2_1layer_head_checkpoint_urls = {
        "vit_small_patch14_dinov2.lvd142m": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_linear_head.pth",
        "vit_base_patch14_dinov2.lvd142m": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_linear_head.pth",
        "vit_large_patch14_dinov2.lvd142m": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear_head.pth",
        "vit_giant_patch14_dinov2.lvd142m": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_linear_head.pth",
    }

    model_patch_info = None

    if open_clip_sep in model_name:
        #: OpenClip
        import open_clip

        arch, arch_detail = model_name.split(open_clip_sep, maxsplit=2)
        model_name_canonized = f"{arch}.{arch_detail}"

        if (
            model_name
            == f"vit_base_patch16_224{open_clip_sep}BiomedCLIP-PubMedBERT_256"
        ):
            #: [[id:8a00b945-4f6d-4988-b4ce-91a4e5a15786][microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 · Hugging Face]]

            model, preprocess = open_clip.create_model_from_pretrained(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            #: `preprocess` is unused

            tokenizer = open_clip.get_tokenizer(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )

        else:
            model, _, preprocess_val = open_clip.create_model_and_transforms(
                arch, pretrained=arch_detail
            )
            #: `preprocess_val` is unused by us

            tokenizer = open_clip.get_tokenizer(arch)

        model.name = model_name
        # model.name = model_name_canonized
        #: When we save qual examples, having `open_clip_sep` in the name is good for us.

        model.clip_p = True

        model_patch_info = patch_info_from_name(
            arch,
            # model_name_canonized,
            bias_token_p=False,
        )

        model.tokenizer = tokenizer

        for m in [
            ##
            #: @methods
            "pixels2patches",
            ##
            "blocks",
        ]:
            setattr(model, m, getattr(model.visual.trunk, m))

        setattr(
            model,
            "has_class_token",
            getattr(model.visual.trunk, "has_class_token", False),
        )
        setattr(
            model,
            "num_prefix_tokens",
            getattr(model.visual.trunk, "num_prefix_tokens", 0),
        )

        #: Define the forward_patch_level function
        def forward_patch_level(self, image, normalize: bool = False, **kwargs):
            features = self.visual.trunk.forward_patch_level(image, **kwargs)
            features = self.visual.head(features)
            return F.normalize(features, dim=-1) if normalize else features

        #: Bind the function to the model object
        model.forward_patch_level = types.MethodType(forward_patch_level, model)

    elif model_name in list(dino2_1layer_head_checkpoint_urls.keys()):
        if "giant" in model_name:
            use_checkpoint = False
            # use_checkpoint = True
            #: Checkpointing seems to need some kind of side-effect free compilable code? It did not work, and gave a nonsensical exception.
        else:
            use_checkpoint = False

        model = timm.create_model(
            model_name,
            pretrained=True,
            **kwargs,
        ).eval()

        # Get the embedding dimension from the model
        embed_dim = model.embed_dim  # Should be 384 for vit_small

        num_classes = 1000  # Set to desired number of classes
        model.head = LinearDecomposed(2 * embed_dim, num_classes)

        # Patch the forward_head method
        def forward_head(self, x, pre_logits=False):
            # x: [batch_size, num_tokens, embed_dim]
            cls_token = x[:, 0, :]  # [batch_size, embed_dim]
            patch_tokens_mean = x[:, 1:, :].mean(dim=1)  # [batch_size, embed_dim]
            linear_input = torch.cat(
                [cls_token, patch_tokens_mean], dim=1
            )  # [batch_size, 2*embed_dim]

            # ic(x.shape, cls_token.shape, patch_tokens_mean.shape, linear_input.shape, self.head)
            x = self.head_drop(linear_input)

            return x if pre_logits else self.head(x)

        # Bind the new forward_head method to the model
        model.forward_head = types.MethodType(forward_head, model)

        # Load the linoear head weights
        head_checkpoint_url = dino2_1layer_head_checkpoint_urls[model_name]
        state_dict = torch.hub.load_state_dict_from_url(
            head_checkpoint_url, map_location="cpu"
        )

        # Load the head weights
        model.head.load_state_dict(state_dict)

        model = torch_prefix_autoset(model)

        if use_checkpoint:
            model.set_grad_checkpointing(True)

    else:
        model = timm.create_model(model_name, pretrained=True, **kwargs).eval()

    if True or not getattr(model, "fair_p", False):
        model = fair_model_get(model).new_model

    if model_patch_info is None:
        model_patch_info = patch_info_from_name(
            model_name,
            bias_token_p=False,
        )

    model.model_patch_info = model_patch_info

    model.to(device)
    model.eval()

    return model


def upscale_image_and_patch(image, patch_size=0, target_size=PLOT_IMAGE_SIZE):
    if isinstance(image, PIL.Image.Image):
        image_pil = image

    elif isinstance(image, torch.Tensor):
        to_pil = torchvision.transforms.ToPILImage()
        image_pil = to_pil(image)

    else:
        image_pil = Image.fromarray(image)

    original_size = image_pil.size[0]  #: the image should be square

    if original_size < target_size:
        scale_factor = (target_size // original_size) + 1
        new_size = scale_factor * original_size
        image_pil = image_pil.resize((new_size, new_size), Image.LANCZOS)
        patch_size *= scale_factor

    return simple_obj(
        image_pil=image_pil,
        patch_size=patch_size,
    )


def blend_invert(
    background: np.ndarray, overlay: np.ndarray, opacity: float
) -> np.ndarray:
    # The alpha channel is the last channel in the image
    alpha = overlay[..., -1] / 255.0

    # The overlay opacity scales the effect
    alpha *= opacity

    # Invert the color channels
    inverted = 255.0 - background[..., :3]

    # Blend inverted and original images
    return (
        background[..., :3] * (1.0 - alpha)[:, :, None] + inverted * alpha[:, :, None]
    )


def sRGBtoLin(colorChannel: np.ndarray) -> np.ndarray:
    return np.where(
        colorChannel <= 0.04045,
        colorChannel / 12.92,
        ((colorChannel + 0.055) / 1.055) ** 2.4,
    )


def YtoLstar(Y: np.ndarray) -> np.ndarray:
    return np.where(Y <= (216 / 24389), Y * (24389 / 27), Y ** (1 / 3) * 116 - 16)


def blend_whiteburn(
    background: np.ndarray,
    overlay: np.ndarray,
    opacity: float,
    brightness_detection_mode="perceptual_lightness",
    threshold=None,
) -> np.ndarray:
    # The alpha channel is the last channel in the image
    alpha = overlay[..., -1] / 255.0

    # The overlay opacity scales the effect
    alpha *= opacity

    # Compute brightness per pixel
    #: https://stackoverflow.com/a/56678483/1410221
    if brightness_detection_mode == "luminance":
        # Convert to linear RGB
        linear_rgb = sRGBtoLin(background / 255.0)

        # Calculate luminance
        brightness = (
            0.2126 * linear_rgb[..., 0]
            + 0.7152 * linear_rgb[..., 1]
            + 0.0722 * linear_rgb[..., 2]
        )
        #: I don't know what range this is.

        if threshold is None:
            threshold = 0.5
    elif brightness_detection_mode == "perceptual_lightness":
        # Convert to linear RGB
        linear_rgb = sRGBtoLin(background / 255.0)

        # Calculate luminance
        luminance = (
            0.2126 * linear_rgb[..., 0]
            + 0.7152 * linear_rgb[..., 1]
            + 0.0722 * linear_rgb[..., 2]
        )

        # Convert luminance to perceptual lightness
        brightness = YtoLstar(luminance) / 100.0
        #: L* is a value from 0 (black) to 100 (white) where 50 is the perceptual "middle grey". L* = 50 is the equivalent of Y = 18.4, or in other words an 18% grey card, representing the middle of a photographic exposure (Ansel Adams zone V).

        if threshold is None:
            threshold = 0.82
    else:
        raise ValueError(
            f"Unknown brightness_detection_mode: {brightness_detection_mode}"
        )

    # Compute mask for whiteburn effect: black for bright pixels, white for dark pixels
    whiteburn = np.where(brightness > threshold, 0, 255)
    # ic(brightness.shape, brightness[0, 0], brightness[100, 100])

    # Apply the whiteburn mask, considering the opacity
    return (
        background[..., :3] * (1.0 - alpha)[:, :, None]
        + whiteburn[:, :, None] * alpha[:, :, None]
    )


# @timed
def blend_images(
    background: PIL.Image.Image,
    overlay: PIL.Image.Image,
    blends=None,
) -> PIL.Image.Image:
    blends = blends or dict(mode=blend_modes.hard_light, opacity=1.0)

    #: Inputs to blend_modes need to be numpy arrays.
    background = np.array(background).astype(float)
    overlay = np.array(overlay).astype(float)

    # Blend images
    blended = background
    for b in blends:
        blender = b["mode"]
        if blender == "invert":
            blender = blend_invert
        elif blender == "whiteburn":
            blender = blend_whiteburn

        blended = blender(blended, overlay, b["opacity"])

    # Convert back to uint8 PIL image
    return Image.fromarray(np.uint8(blended))


def overlay_colored_grid(
    image_natural_tensor,
    attributions_n,
    metadata=None,
    plot_output_p=True,
    # normalize="max_attr",
    normalize=None,
    # outlier_quantile=0.07
    outlier_quantile=0.1,
    # outlier_quantile=0.0,
    # intensifier=h_exponent,
    intensifier=None,
    scale=None,
    title="",
    label=None,
    label_natural=None,
    export_name=None,
    # export_name_postfix="",
    export_name_postfix=None,
    export_dir=None,
    export_each_p=True,
    export_format="jpeg",
    lock_key=None,
    # export_each_p=False,
    export_log=None,
    export_tlg_id=None,
    tlg_autobatch=True,
    image_concats=None,
    image_concats_right=None,
    overlay_alone_p=True,
    overlay_nonscaled_p=True,
    rank_mode="overlay_alone",
    figure_size_scale=3,
    color_negative=(255, 0, 0),
    # color_positive=(0, 255, 0),
    color_positive="viridis",
    to_pixel_colormap=None,
    colormap_split_p=False,
    rank_color="white",
    rank_font_size=24,
    bar_thickness=20,
    model_patch_info=None,
    pixel_p=False,
    # to_pixel=False,
    to_pixel="bicubic",
    export_pixel_attr_alone_p=True,
    # patch_size=16,
    # bias_token_p=True,
    blend_alpha=0.6,
):
    if metadata is None:
        metadata = dict()

    if export_dir is None:
        export_dir = tempfile.mkdtemp()
        #: Without saving the figures, the results will become corrupted, idk why.

    if pixel_p:
        bias_token_p = False
        num_prefix_tokens = 0
        patch_size = 1
        rank_mode = "none"

        attributions_n = attributions_n.flatten()

    else:
        bias_token_p = model_patch_info.get("bias_token_p", False)
        num_prefix_tokens = model_patch_info.num_prefix_tokens
        patch_size = model_patch_info.patch_resolution

    if label is not None:
        metadata["label"] = label

    if label_natural is not None:
        metadata["label_natural"] = label_natural

    metadata["pixel_p"] = pixel_p
    metadata["rank_mode"] = rank_mode
    metadata["normalize"] = normalize
    metadata["outlier_quantile"] = outlier_quantile
    metadata["scale"] = scale
    metadata["intensifier"] = intensifier

    patch_size_orig = patch_size

    attributions_n = attributions_n.cpu().float()

    if image_concats is None:
        image_concats = []
    # image_concats = copy.copy(image_concats)

    #: Convert tensors to PIL images if necessary
    to_pil = torchvision.transforms.ToPILImage()

    image_concats = [upscale_image_and_patch(img).image_pil for img in image_concats]

    if image_concats_right is None:
        image_concats_right = []

    image_concats_right = [
        upscale_image_and_patch(img).image_pil for img in image_concats_right
    ]

    #: Convert the image tensor to a PIL Image
    upscale_result = upscale_image_and_patch(image_natural_tensor, patch_size)
    image_natural_pil, patch_size = upscale_result.image_pil, upscale_result.patch_size
    image_natural_pil.export_name = "image_natural"

    image_concats_right.append(image_natural_pil)

    normalized_obj = normalize_map(
        attributions=attributions_n,
        normalize=normalize,
        outlier_quantile=outlier_quantile,
        num_prefix_tokens=num_prefix_tokens,
        bias_token_p=bias_token_p,
    )
    attributions_normalized, attributions_skipped = (
        normalized_obj.attributions_normalized,
        normalized_obj.attributions_skipped,
    )

    # ic(torch_shape_get(attributions_normalized), attributions_normalized.min(), attributions_normalized.max())
    ###

    if not rank_mode == "none":
        ranks_top_is_1 = rank_tensor(attributions_skipped)
        #: The most important patch is turned into =1=, and the least important patch is turned into =len(attr)=.

        overlay_image_rank = Image.new("RGBA", image_natural_pil.size, (0, 0, 0, 0))
        draw_rank = ImageDraw.Draw(overlay_image_rank)
        font = PIL.ImageFont.truetype("DejaVuSansMono-Bold", size=rank_font_size)
        idx = 0
        for i in range(0, image_natural_pil.height, patch_size):
            for j in range(0, image_natural_pil.width, patch_size):
                #: Get the rank for the current patch
                rank = ranks_top_is_1[idx].item()

                #: Calculate the center of the patch
                center_x = j + patch_size // 2
                center_y = i + patch_size // 2

                #: Draw the text on the image
                draw_rank.text(
                    (center_x, center_y),
                    str(rank),
                    fill=rank_color,
                    font=font,
                    anchor="mm",
                )

                idx += 1
    ###
    if to_pixel:
        if scale is not None:
            msg = "to_pixel does not yet support scaling."
            if scale != 1:
                # raise NotImplementedError(msg)

                print(msg, file=sys.stderr, flush=True)

        to_pixel = to_iterable(to_pixel)
        for interpolation in to_pixel:
            attributions_scaled = scale_patch_to_pixel(
                attributions_normalized,
                output_channel_dim_p=False,
                # output_width=model_patch_info.image_resolution,
                # output_height=model_patch_info.image_resolution,
                # output_height=PLOT_IMAGE_SIZE,
                # output_width=PLOT_IMAGE_SIZE,
                output_height=image_natural_pil.height,
                output_width=image_natural_pil.width,
                interpolate_mode=interpolation,
            )
            attributions_scaled = attributions_scaled[0]  #: remove batch dimension
            attributions_scaled = attributions_scaled.cpu()

            if to_pixel_colormap is None:
                if isinstance(color_positive, str):
                    to_pixel_colormap = color_positive

                else:
                    to_pixel_colormap = "viridis"

            # ic(torch_shape_get(attributions_scaled), attributions_scaled.min(), attributions_scaled.max())
            #: Even with outlier_quantile=0, the upscaling changes the range of the data.
            # ic| torch_shape_get(attributions_normalized): (torch.float32, torch.Size([576]), device(type='cpu'))
            # attributions_normalized.min(): tensor(0.0008)
            # attributions_normalized.max(): tensor(1.)
            # ic| torch_shape_get(attributions_scaled): (torch.float32, torch.Size([672, 672]), device(type='cpu'))
            # attributions_scaled.min(): tensor(-0.1094)
            # attributions_scaled.max(): tensor(1.1106)

            attributions_scaled = attributions_scaled.clamp(min=0, max=1)
            #: if we don't clamp, =outlier_quantile= will be useless.

            attr_pixel_level = grey_tensor_to_pil(
                attributions_scaled,
                colormap=to_pixel_colormap,
                # normalize_p=False,
                normalize_p=True,
            ).image_pil
            # attr_pixel_level = to_pil(attributions_scaled)

            #: no need, as we are already upscaling to PLOT_IMAGE_SIZE.
            # attr_pixel_level = upscale_image_and_patch(attr_pixel_level, patch_size).image_pil

            if export_pixel_attr_alone_p:
                attr_pixel_level.export_name = f"{interpolation}_attr"
                image_concats.append(attr_pixel_level)

            if True:
                if attr_pixel_level.mode != "RGBA":
                    attr_pixel_level = attr_pixel_level.convert("RGBA")

                if image_natural_pil.mode != "RGBA":
                    image_natural_pil = image_natural_pil.convert("RGBA")

                alpha = blend_alpha
                # ic(blend_alpha)

                attr_pixel_level_blended = Image.blend(
                    image_natural_pil, attr_pixel_level, alpha
                )
                #: image1 * (1.0 - alpha) + image2 * alpha
                #: [[https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.blend]]

                attr_pixel_level_blended.export_name = f"{interpolation}_blended"
                image_concats.append(attr_pixel_level_blended)

    ###: Draw colored patch_sizeXpatch_size rectangles on the empty image
    colormaps = dict()
    if isinstance(color_positive, str):
        if colormap_split_p:
            colormaps["negative"] = colormap_get(
                color_positive,
                start=0.0,
                end=0.5,
                reverse_p=True,
            )
            colormaps["positive"] = colormap_get(
                color_positive,
                start=0.5,
                end=1.0,
            )

            if color_negative:
                print(
                    f"Warning: colormap_split_p but color_negative still supplied: {color_negative}"
                )
        else:
            colormaps["positive"] = colormap_get(
                color_positive,
            )

    if isinstance(color_negative, str):
        if "negative" not in colormaps:
            colormaps["negative"] = colormap_get(
                color_negative,
            )

    if not pixel_p and overlay_nonscaled_p:
        #: Create an empty image with the same size as the original image and a transparent background
        overlay_image = Image.new("RGBA", image_natural_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_image)
        idx = 0
        for i in range(0, image_natural_pil.height, patch_size):
            for j in range(0, image_natural_pil.width, patch_size):
                if (
                    idx < attributions_n.shape[0]
                ):  #: Check if there is a transparency value for the current patch
                    attribution_unprocessed = attributions_skipped[idx].item()
                    attribution = attributions_normalized[idx].item()
                    # ic("before", attribution_unprocessed, attribution)

                    neg = attribution < 0.0
                    attribution = abs(attribution)

                    if intensifier:
                        attribution = intensifier(attribution)

                    if scale is not None:
                        scale_current = scale
                    elif isinstance(color_positive, tuple):
                        scale_current = 2.5
                    else:
                        scale_current = 1.0
                    attribution = attribution * scale_current

                    # ic("after", attribution_unprocessed, attribution)
                    attribution = max(attribution, 0.0)
                    attribution = min(attribution, 1.0)

                    ##
                    if neg:
                        if "negative" in colormaps:
                            colormap = colormaps["negative"]
                        else:
                            color = color_negative
                            colormap = None
                    else:
                        if "positive" in colormaps:
                            colormap = colormaps["positive"]
                        else:
                            color = color_positive
                            colormap = None

                    if colormap:
                        color = colormap(attribution)[:3]
                        color = tuple(int(c * 255) for c in color)
                        alpha = blend_alpha

                    else:
                        # alpha = 1.0
                        alpha = attribution
                    ##

                    # ic(alpha, colormap)
                    alpha = int(alpha * 255)
                    #: Convert the transparency value to [0, 255] range

                    draw.rectangle(
                        [j, i, j + patch_size, i + patch_size], fill=(*color, alpha)
                    )
                    idx += 1
        ###

        if overlay_alone_p:
            overlay_image_white = Image.alpha_composite(
                Image.new("RGBA", overlay_image.size, (255, 255, 255, 255)),
                overlay_image,
            )

            # ic(patch_size_orig)
            if patch_size_orig <= 8:
                rank_mode = "none"

            if rank_mode == "overlay_alone":
                # overlay_image_white = Image.alpha_composite(
                #     overlay_image_white,
                #     overlay_image_rank,
                # )

                #: Time taken by timm.models.decomposition.blend_images: 0.06037282943725586 seconds
                overlay_image_white = blend_images(
                    background=overlay_image_white,
                    overlay=overlay_image_rank,
                    blends=[
                        # dict(mode=blend_modes.difference, opacity=0.1),
                        # dict(mode=blend_modes.overlay, opacity=1.0),
                        # dict(mode=blend_modes.overlay, opacity=1.0),
                        ##
                        # dict(mode=blend_modes.divide, opacity=1.0),
                        # dict(mode=blend_modes.soft_light, opacity=1.0),
                        # dict(mode=blend_modes.hard_light, opacity=1.0),
                        # dict(mode='invert', opacity=1.0),
                        dict(mode="whiteburn", opacity=1.0),
                    ],
                )
            elif rank_mode == "none":
                pass
            else:
                raise ValueError(f"Unsupported rank_mode: {rank_mode}")

            overlay_image_white = overlay_image_white.convert("RGB")
            overlay_image_white.export_name = "overlay_white"
            image_concats.append(overlay_image_white)

        #: Overlay the empty image with colored rectangles onto the original image
        result_image = Image.alpha_composite(
            image_natural_pil.convert("RGBA"), overlay_image
        ).convert("RGB")
        result_image.export_name = "overlay"
        image_concats.append(result_image)

    ###
    image_concats += image_concats_right

    concat_height = max(img.height for img in image_concats)

    white_bar = Image.new("RGB", (bar_thickness, concat_height), (255, 255, 255))

    total_concat_width = (
        sum(img.width for img in image_concats)
        + (len(image_concats) - 1) * white_bar.width
    )

    concatenated_image = Image.new("RGB", (total_concat_width, concat_height))

    current_x = 0
    for i, image_concat in enumerate(image_concats):
        concatenated_image.paste(image_concat, (current_x, 0))
        current_x += image_concat.width

        if i != (len(image_concats) - 1):
            concatenated_image.paste(white_bar, (current_x, 0))
            current_x += white_bar.width

    result_image = concatenated_image

    plt.figure(figsize=(figure_size_scale * 6.4, figure_size_scale * 6.4))
    #: tight layout seems to make meaningless? Still probably the larger the better.
    #: default=(6.4, 4.8)

    plt.imshow(result_image)
    plt.axis("off")

    # title += f"\nfinal CLS={attributions_n[0]:.3f}, err={attributions_n[-1]:.3f}"

    plt.title(title)
    plt.tight_layout()

    dest = None
    if export_dir:
        mkdir(export_dir)

        if export_name is None:
            export_name = title.replace("\n", "/")
            # export_name = title.replace("\n", "; ")

        if export_name_postfix is None:
            if export_each_p:
                export_name_postfix = "/"

        if export_name_postfix:
            export_name += export_name_postfix

        if metadata:
            dest = f"{export_dir}/{export_name}_md.json"
            mkdir(dest, do_dirname=True)
            json_save(
                metadata,
                file=dest,
                exists="ignore",
            )

        if export_each_p:
            i = 0
            for img in image_concats:
                img_name = getattr(img, "export_name", None)
                if img_name is None:
                    img_name = str(i)
                    i += 1

                dest = f"{export_dir}/{export_name}_{img_name}.{export_format}"
                mkdir(dest, do_dirname=True)

                if export_format.upper() in [
                    "JPEG",
                    "JPG",
                ]:
                    try:
                        img = img.convert("RGB")

                    except:
                        ic(img)

                        raise

                img.save(dest, export_format.upper())

        if True or export_each_p:
            dest = f"{export_dir}/{export_name}_all.{export_format}"

        else:
            dest = f"{export_dir}/{export_name}.{export_format}"

        mkdir(dest, do_dirname=True)

        plt.savefig(
            dest,
            # dpi=300,
            bbox_inches="tight",
        )

        if export_log:
            print_to_file(dest, file=export_log)

    if export_tlg_id:
        print_diag(
            f"sending to Telegram (id={export_tlg_id}) ...", group="telegram_send"
        )

        lock_path = None

        # wait_p = True
        wait_p = False

        #: After using thread pools for 'send':
        #: Time taken by pynight.common_telegram.send: 5.435943603515625e-05 seconds
        common_telegram.send(
            files=(dest if dest else plt.gcf()),
            chat_id=export_tlg_id,
            wait_p=wait_p,
            # lock_path=lock_path,
            # lock_key='decompv',
            lock_key=lock_key,
            autobatch=tlg_autobatch,
            savefig_opts=dict(
                bbox_inches="tight",
            ),
        )

    if plot_output_p:
        #: Without saving the figure first, the results can be out of layout, idk why; so do export the figures if you want to show them.
        plt.show()
        print("")  #: create empty line for org-mode scrolling bug
    elif dest:
        print(dest)

    plt.close()

    return None


##
def rollout_extract_number(s):
    match = re.search(r"rollout_([^_]+)", s)
    if match:
        return float(match.group(1))
    else:
        return 0.0


def attributions_distribute_biascls(
    attributions_v, bias_decomposition_mode="equal", inplace=True
):
    if not bias_decomposition_mode:
        return attributions_v

    if not inplace:
        attributions_v = attributions_v.clone()

    bias = attributions_v[..., -1, :, :].clone()
    attributions_v[..., -1, :, :] = 0
    # bias = torch.mean(bias, dim=-2)

    cls = attributions_v[..., 0, :, :].clone()
    attributions_v[..., 0, :, :] = 0
    # cls = torch.mean(cls, dim=-2)

    all = bias + cls
    all = torch.sum(all, dim=-2)  #: sum pixel attributions
    # ic(all.shape, attributions_v.shape)

    attributions_v[..., 1:-1, :, :] = bias_distributer(
        attributions_v=attributions_v[..., 1:-1, :, :],
        bias=all,
        bias_decomposition_mode=bias_decomposition_mode,
    )

    # all = all.unsqueeze(-3)
    # tokens_len = attributions_v.shape[-3] - 2
    # all /= tokens_len
    # ic(tokens_len, attributions_v.shape, all.shape)
    # attributions_v[..., 1:-1, :, :] += all

    return attributions_v


def bias_distributer(bias, attributions_v, bias_decomposition_mode="absdot"):
    if not bias_decomposition_mode:
        return attributions_v

    device = attributions_v.device

    eps = 1e-12

    if bias_decomposition_mode in ("dot", "absdot"):
        weights = einops.einsum(
            attributions_v,
            bias,
            "... patch_attr pixel_attr d, ... d -> ... patch_attr pixel_attr",
        )

        if bias_decomposition_mode == "absdot":
            weights = torch.abs(weights)
    elif bias_decomposition_mode in ("sim", "abssim"):
        ##
        # weights = torch.nn.functional.cosine_similarity(attributions_v, bias, dim=-1)
        #: OutOfMemoryError: CUDA out of memory. Tried to allocate 548.65 GiB
        ##
        weights = einops.einsum(
            attributions_v,
            bias,
            "... patch_attr pixel_attr d, ... d -> ... patch_attr pixel_attr",
        )
        weights /= torch.norm(attributions_v, dim=-1) + eps
        ##

        weights = (torch.norm(attributions_v, dim=-1) > eps) * weights
        #: Zero vectors have undefined cosine sim.

        if bias_decomposition_mode == "abssim":
            weights = torch.abs(weights)
    elif bias_decomposition_mode == "norm":
        weights = torch.norm(attributions_v, dim=-1)
    elif bias_decomposition_mode == "equal":
        # weights = (torch.norm(attributions_v, dim=-1) != 0) * 1.0
        weights = torch.ones(attributions_v.shape[:-1]).to(device)
    elif bias_decomposition_mode == "cls":
        raise NotImplementedError()

        weights = torch.zeros(attributions_v.shape[:-1], device=attributions_v.device)
        weights[:, :, 0] = 1.0
    else:
        raise ValueError(
            f"bias_decomposition_mode not supported: {bias_decomposition_mode}"
        )

    weights = weights / (weights.sum(dim=(-1, -2), keepdim=True) + eps)

    ic(attributions_v.shape, weights.shape, bias.shape)
    weighted_bias = einops.einsum(
        weights,
        bias,
        "... patch_attr pixel_attr , ... d -> ... patch_attr pixel_attr d",
    )
    return attributions_v + weighted_bias


def threshold_filter(attributions_n, *, top_k=None, top_ratio=None, top_fill=1):
    assert top_k is None or top_ratio is None

    unbatched = len(attributions_n.shape) == 1
    if unbatched:
        dim = 0
    else:
        dim = 1

    token_count = attributions_n.shape[dim]
    attributions_n2 = torch.zeros_like(attributions_n)

    if top_k is None:
        if top_ratio is not None:
            top_k = int(top_ratio * token_count)
        else:
            top_k = 0

    if top_k > 0:
        top_k_indices = attributions_n.topk(
            top_k, dim=dim, largest=True, sorted=False
        ).indices
        # ic(top_k_indices)

        if unbatched:
            attributions_n2[top_k_indices] = top_fill
        else:
            attributions_n2[
                torch.arange(attributions_n2.size(0)).unsqueeze(-1), top_k_indices
            ] = top_fill

    return attributions_n2


def lowpass_filter(attributions_n):
    unbatched = len(attributions_n.shape) == 1

    height = int(math.sqrt(attributions_n.shape[-1]))
    width = height
    attributions_2D = rearrange(
        attributions_n,
        "... (height width) -> ... 1 height width",
        height=height,
        width=width,
    )
    if unbatched:
        attributions_2D = attributions_2D.unsqueeze(0)
        #: adding the batch dim

    ic(attributions_2D.shape, attributions_n.shape)

    attributions_2D = kornia.filters.gaussian_blur2d(
        attributions_2D, kernel_size=(3, 3), sigma=(1.5, 1.5)
    )
    #: input: (B,C,H,W)

    attributions_n = rearrange(
        attributions_2D, "... 1 height width -> ... (height width)"
    )
    if unbatched:
        attributions_n = attributions_n.squeeze(0)

    return attributions_n


def named_experiment(
    *,
    title,
    config,
    distribute_biascls="NA",
):
    name = ""

    if distribute_biascls != "NA":
        name += f"bcls_{distribute_biascls}/"

    return simple_obj(
        title=title,
        name=name,
    )


def named_experiment2(
    *,
    title,
    config,
    distribute_biascls="NA",
):
    """
    Creates names for storing attributions into a dataset.
    """

    name = title

    if distribute_biascls != "NA":
        name += f"_bcls_{distribute_biascls}"

    return simple_obj(
        title=title,
        name=name,
    )


def entrypoint_decompv(
    model,
    url,
    name=None,
    plot_first_alone_p=True,
    block_indices=[6, 7, 8, 9, 10, 11],
    mode="CLS_n",
    target=None,
    target2name=None,
    blend_modes=["second"],
    second_attr_scale=1.0,
    distribute_biascls="equal",
    lowpass_filter_fn=None,
    lowpass_filter_second_p=True,
    top_name="unnamed",
    **kwargs,
):
    #: @deprecated
    ##
    model_name = model_name_get(model)

    try:
        image_all = image_from_url(model=model, url=url)
    except:
        print(traceback.format_exc(), file=sys.stderr)
        return

    if name is None:
        if isinstance(url, str):
            name = hash_url(url)
        else:
            name = hash_tensor(image_all.image_natural)
            # name = str(uuid.uuid4().hex)

    device = model_device_get(model)

    config_1 = DecompositionConfig(
        device=device,
        attributions_aggregation_strategy="vector",
    )

    title_attr = ""
    zero_out_cls = True
    if mode == "CLS_n":
        decomposition_config = config_1
        inputs = model.forward_features_decomposed(
            image_all.image_dv,
            decomposition_config=decomposition_config,
        )

        attributions_n = attributions_n_get(
            inputs[f"attributions_v"],
            0,
            zero_out_error=True,
        ).squeeze()

        ic(
            attributions_n.shape,
            # torch.sum(attributions_n),
            # attributions_n,
        )

        attributions_n = mean_normalize_last_dim(attributions_n)
        # ic(attributions_n.shape, torch.sum(attributions_n), attributions_n,)

        attributions_dict = dict()
        attributions_dict[
            named_experiment(title=title_attr, config=decomposition_config)
        ] = attributions_n

    elif mode == "sm":
        assert target is not None

        decomposition_config = config_1

        zero_out_cls = False

        second_attr_scale *= 10

        if True:
            inputs_features = model.forward_features_decomposed(
                image_all.image_dv,
                decomposition_config=decomposition_config,
            )

            attributions_dict = dict()

            attributions_v_orig = inputs_features.attributions_v
            features_orig = inputs_features.features
            for distribute_biascls_i in distribute_biascls:
                attributions_v = attributions_v_orig.clone()
                attributions_v = attributions_distribute_biascls(
                    attributions_v,
                    bias_decomposition_mode=distribute_biascls_i,
                    inplace=True,
                )

                inputs = simple_obj_update(
                    inputs_features,
                    attributions_v=attributions_v,
                    features=features_orig,
                )
                inputs_head = model.forward_head_decomposed(inputs)
                softmax = SoftmaxDecomposed(dim=-1)
                inputs_sm = softmax.forward(inputs_head)
                attributions_n = inputs_sm.attributions_v

                attributions_n = attributions_n[0, :, :, target]

                attributions_n = torch.sum(attributions_n, dim=-1)

                title_attr = f"distribute_biascls=before_head,{distribute_biascls_i}, CLS={attributions_n[0]:.3f}, err={attributions_n[-1]:.3f}, label_prob={inputs_sm.features[0, target]:.6f}\n"
                attributions_dict[
                    named_experiment(
                        title=title_attr,
                        config=decomposition_config,
                        distribute_biascls=distribute_biascls_i,
                    )
                ] = attributions_n
        else:
            inputs = model.forward_decomposed(
                image_all.image_dv,
                decomposition_config=decomposition_config,
            )

            softmax = SoftmaxDecomposed(dim=-1)
            inputs_sm = softmax.forward(inputs)
            attributions_v = inputs_sm.attributions_v
            attributions_dict = dict()
            for distribute_biascls_i in distribute_biascls:
                attributions_n = attributions_distribute_biascls(
                    attributions_v,
                    bias_decomposition_mode=distribute_biascls_i,
                    inplace=False,
                )

                attributions_n = attributions_n[0, :, :, target]

                # ic(attributions_n.shape)
                # ic(attributions_n.shape, attributions_n)
                attributions_n = torch.sum(attributions_n, dim=-1)

                title_attr = f"distribute_biascls=after_head,{distribute_biascls_i}, CLS={attributions_n[0]:.3f}, err={attributions_n[-1]:.3f}, label_prob={inputs_sm.features[0, target]:.6f}\n"
                attributions_dict[
                    named_experiment(
                        title=title_attr,
                        config=decomposition_config,
                        distribute_biascls=distribute_biascls_i,
                    )
                ] = attributions_n

                ic(
                    distribute_biascls_i,
                    torch.max(attributions_n[1:-1]),
                    torch.min(attributions_n[1:-1]),
                    torch.mean(attributions_n[1:-1]),
                )
                # ic(attributions_n)

                # print_diag(f"Softmax attr final shape:  {ic_colorize2(attributions_n.shape)}", group="shape")
    else:
        raise ValueError(f"unknown mode: {mode}")

    image_concats = [
        image_all.image_natural,
    ]
    image_concats_right = [
        image_all.image_cpu_squeezed,
    ]
    # img_tensor_show(image_all.image_natural)
    # img_tensor_show(image_all.image_cpu_squeezed)

    if target:
        target_name = str(target)
        if target2name:
            target_name = target2name[target]

        name += f"_trg{target}"

    if lowpass_filter_fn:
        name += f"_{fn_name(lowpass_filter_fn)}"
    else:
        name += "_nolp"

    block_indices = [None] + block_indices
    for i, block_index in enumerate(block_indices):
        rawatt = None
        if isinstance(block_index, str) and block_index.startswith("rollout"):
            residual_strength = rollout_extract_number(block_index)
            # ic(residual_strength)

            for block_i in range(0, len(model.blocks)):
                rawatt_i = torch.mean(
                    inputs[f"blocks__{block_i}__attn__rawattn"], dim=1
                )

                #: adding the residual connections as well to the rollout
                #: this didn't improve the results when done on raw attentions
                if residual_strength:
                    rawatt_i = (
                        1 - residual_strength
                    ) * rawatt_i + residual_strength * torch.eye(
                        rawatt_i.shape[-1]
                    ).unsqueeze(
                        0
                    )

                if block_i == 0:
                    rawatt = rawatt_i
                else:
                    rawatt = einops.einsum(
                        rawatt_i,
                        rawatt,
                        "batch attr_to middleman, batch middleman attr_from -> batch attr_to attr_from",
                    )

            rawatt = rawatt[0, 0, :].squeeze()

            ic(rawatt.shape)
            rawatt[0] = 0.0  #: zero out the CLS attention

            rawatt = mean_normalize_last_dim(rawatt)
            # rawatt *= 2 #: @surprise
        elif block_index is not None:
            rawatt = inputs[f"blocks__{block_index}__attn__rawattn"][
                0, :, 0, :
            ].squeeze()
            rawatt = torch.mean(rawatt, dim=0)

            rawatt[0] = 0.0  #: zero out the CLS attention

            rawatt = mean_normalize_last_dim(rawatt)

        second_attr = rawatt
        if second_attr is not None:
            second_attr = second_attr * second_attr_scale

        named_experiment_i = 0
        for named_experiment_obj, attributions_n in attributions_dict.items():
            named_experiment_i += 1
            for blend_mode in blend_modes:
                if (second_attr is None and not plot_first_alone_p) or (
                    named_experiment_i > 1
                    and not plot_first_alone_p
                    and blend_mode == "second"
                ):
                    # print_diag("skipped plotting all distribute_biascls")

                    #: no need to repeat the second_attr when there is no dependence on first_attr
                    continue

                title_attr = named_experiment_obj.title
                first_title = ""
                if target_name:
                    first_title += f"label={target_name}\n"

                first_title = f"{first_title}model={model_name}\nmode={mode}, blend={blend_mode}\n"
                first_title += title_attr
                first_title += "Vector Decomposition"
                first_export_dir = f"./plots/{top_name}/{model_name}/{name}/{named_experiment_obj.name}/blend_{blend_mode}/"

                if second_attr is not None:
                    title = f"{first_title} Vs. Raw Attention (Layer {block_index})"
                    export_dir = f"{first_export_dir}vs_rawatt_{block_index}/"
                else:
                    title = first_title
                    export_dir = first_export_dir

                if lowpass_filter_fn is not None:
                    attributions_n[1:-1] = lowpass_filter_fn(attributions_n[1:-1])
                    if second_attr is not None and lowpass_filter_second_p:
                        second_attr[1:-1] = lowpass_filter_fn(second_attr[1:-1])

                vis_attr(
                    # image_all.image_cpu,
                    image_all.image_natural,
                    first_attr=attributions_n,
                    second_attr=second_attr,
                    zero_out_cls=zero_out_cls,
                    first_title=first_title,
                    title=title,
                    export_name=f"vec",
                    first_export_dir=first_export_dir,
                    export_dir=export_dir,
                    plot_first_alone_p=plot_first_alone_p,
                    image_concats=image_concats,
                    image_concats_right=image_concats_right,
                    blend_mode=blend_mode,
                    **kwargs,
                )
                plot_first_alone_p = False

    return None


def entrypoint_normatt(
    model,
    url,
    name=None,
    block_what="__attnblk__preres__attrv",
    plot_first_alone_p=False,
    **kwargs,
):
    if name is None:
        name = hash_url(url)

    image_all = image_from_url(model=model, url=url)

    device = model_device_get(model)

    res = model.forward_features_decomposed(
        image_all.image_dv,
        decomposition_config=DecompositionConfig(
            device=device,
            residual2_decompose_p=False,
            layer_norm_decompose_p=False,
            attributions_aggregation_strategy="reset",
        ),
    )

    blocks_len = len(model.blocks)

    for i in range(0, blocks_len):
        vis_normatt_vs_rawatt(
            # image_all.image_cpu,
            image_all.image_natural,
            res,
            block_index=i,
            block_what=block_what,
            zero_out_cls=True,
            first_title=f"Block {i}: vector_norm*att",
            title=f"Block {i}: vector_norm*att vs rawatt",
            export_name=f"b{i:02d}",
            export_dir=f"./plots/no_layer_norm/{block_what}/{name}/vs_rawatt",
            first_export_dir=f"./plots/no_layer_norm/{block_what}/{name}/",
            plot_first_alone_p=plot_first_alone_p,
            **kwargs,
        )
    return None


##
class LinearDecomposed(nn.Linear):
    def __init__(self, *args, **kwargs):
        # self.prefix = 'unset.'
        self.prefix = f"unset.{self.__class__.__name__}"

        return super().__init__(*args, **kwargs)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.prefix = prefix
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, inputs, warn_nondecomposed=False) -> Tensor:
        prefix = self.prefix

        if decomposed_inputs_p(inputs):
            decomposition_config = inputs.decomposition_config

            decompose_p = decompose_p_v2(
                decomposition_config.linear_decompose_p,
                attributions_v=inputs.attributions_v,
            )

            features = inputs.features
            features_out = F.linear(features, self.weight, self.bias)

            attributions_v = inputs.attributions_v
            if decompose_p:
                with no_grad_maybe(decomposition_config.detach_p):
                    attributions_v = F.linear(attributions_v, self.weight)

                    if self.bias is not None:
                        if (
                            decomposition_config.bias_decomposition_mode
                            == "error_token"
                        ):
                            attributions_v[..., -1, 0, :] += self.bias
                        else:
                            raise NotImplementedError(
                                f"bias_decomposition_mode: {bias_decomposition_mode}"
                            )

            inputs = simple_obj_update(
                inputs,
                features=features_out,
                attributions_v=attributions_v,
                del_p="can",
            )

            if decompose_p:
                #: Without propagating the decomposition, the shapes can become invalid; hence, I have conditionally disabled the check here. It's possible to check if the shapes match.
                check_attributions_v2(
                    inputs,
                    prefix=self.prefix,
                    assert_p=decomposition_config.assertion_check_attributions_p,
                )

            return inputs
        else:
            if warn_nondecomposed:
                print_diag(
                    f"LinearDecomposed: nondecomposed inputs at `{stacktrace_caller_line()}`",
                    group="warning.nondecomposed",
                )

            features = inputs
            dissent_suppression = dynamic_obj.linear_dissent_suppression

            result_orig = F.linear(features, self.weight, self.bias)
            if dissent_suppression is None or dissent_suppression == 0:
                return_value = result_orig
            else:
                print_diag(
                    f"{prefix}: dissent_suppression: {dissent_suppression}",
                    group="gradient_mode",
                )
                x = features
                W = self.weight

                y = torch.einsum("oi,...i->...o", W, x)
                #: We could ignore the bias ...
                # y = result_orig

                # ic(W.shape, self.bias.shape, x.shape, y.shape)
                #: ic| W.shape: torch.Size([3072, 768])
                #: self.bias.shape: torch.Size([3072])
                #: x.shape: torch.Size([4, 785, 768])
                #: y.shape: torch.Size([4, 785, 3072])

                W_neg = W.clamp(max=0)
                W_pos = W.clamp(min=0)

                x_neg = x.clamp(max=0)
                x_pos = x.clamp(min=0)

                y_pos = torch.einsum("oi,...i->...o", W_pos, x_pos)
                y_pos += torch.einsum("oi,...i->...o", W_neg, x_neg)

                y_neg = torch.einsum("oi,...i->...o", W_pos, x_neg)
                y_neg += torch.einsum("oi,...i->...o", W_neg, x_pos)

                y_conformists = torch.where(y >= 0, y_pos, y_neg)
                y_nonconformists = torch.where(y >= 0, y_neg, y_pos)

                if dissent_suppression < 1:
                    y_nonconformists_adjusted = y_nonconformists * (
                        1 - dissent_suppression
                    )
                    y_conformists_adjusted = (
                        y_conformists + y_nonconformists * dissent_suppression
                    )

                else:
                    assert dissent_suppression == 1

                    y_nonconformists_adjusted = 0
                    y_conformists_adjusted = y

                eps = 1e-10
                y_conformists_adjustment_multiplier = y_conformists_adjusted / (
                    y_conformists + (eps * y_conformists.sign())
                )
                y_conformists_adjustment_multiplier = (
                    y_conformists_adjustment_multiplier.detach()
                )
                assert (
                    y_conformists_adjustment_multiplier.sign() >= 0
                ).all(), f"(y_conformists_adjustment_multiplier < 0).mean: shape={y_conformists_adjustment_multiplier.shape}\n\t{y_conformists_adjustment_multiplier[y_conformists_adjustment_multiplier < 0].mean()}"

                y_correct_grad = (
                    y_conformists_adjustment_multiplier * y_conformists
                    + y_nonconformists_adjusted
                )

                # ic(y_correct_grad.shape)
                #: ic| y_correct_grad.shape: torch.Size([4, 785, 3072])
                if self.bias is not None:
                    y_correct_grad += self.bias

                return_value = swap_backward(result_orig, y_correct_grad)
                if dynamic_obj.gbrand_forward_checks_p:
                    assert torch.allclose(
                        result_orig,
                        y_correct_grad,
                        rtol=1e-03,
                        atol=1e-01,
                    ), f"{prefix}: result_orig and return_value are not close, max error: {torch.max(torch.abs(result_orig - y_correct_grad)).item()}"
                    #: blocks.10.mlp.fc1.: result_orig and return_value are not close, max error: 0.0001239776611328125

            ##
            # bias_multiplier = dynamic_obj.get("bias_multiplier", 1)
            # if True or bias_multiplier != 1:
            #     print_diag(f"{prefix}: bias_multiplier: {bias_multiplier}", group="gradient_mode")

            layer_name = "linear"
            #: @duplicateCode/6b32d3a832249d36819faf9531ae5ed1
            delattr_force(self, "stored_out_grad")
            delattr_force(self, "stored_bias_attributions")
            delattr_force(self, "stored_bias_attributions_raw")
            if dynamic_obj.get(f"{layer_name}_attribute_bias_p"):
                tensor_register_hook(
                    return_value,
                    partial(
                        store_grad_and_bias_attributions,
                        bias=self.bias,
                        # bias=self.bias * bias_multiplier,
                        store_in=self,
                        layer_name=layer_name,
                    ),
                )
            ##

            return return_value


class NightConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        dissent_suppression="from_config",
    ):
        # torch.nn.Conv2d(
        # in_channels: int,
        # out_channels: int,
        # kernel_size: Union[int, Tuple[int, int]],
        # stride: Union[int, Tuple[int, int]] = 1,
        # padding: Union[str, int, Tuple[int, int]] = 0,
        # dilation: Union[int, Tuple[int, int]] = 1,
        # groups: int = 1,
        # bias: bool = True,
        # padding_mode: str = 'zeros',
        # device=None,
        # dtype=None,
        # )
        super(NightConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.prefix = f"unset.{self.__class__.__name__}"
        self.dissent_suppression = dissent_suppression

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.prefix = prefix
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        prefix = self.prefix

        dissent_suppression = self.dissent_suppression
        if dissent_suppression == "from_config":
            dissent_suppression = dynamic_obj.conv2d_dissent_suppression

        result_orig = F.conv2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        #: conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)

        if not dissent_suppression:
            return result_orig

        print_diag(
            f"{prefix}: dissent_suppression: {dissent_suppression}",
            group="gradient_mode",
        )

        W = self.weight
        y = F.conv2d(
            x,
            W,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        W_neg = W.clamp(max=0)
        W_pos = W.clamp(min=0)

        x_neg = x.clamp(max=0)
        x_pos = x.clamp(min=0)

        y_pos = F.conv2d(
            x_pos,
            W_pos,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        y_pos += F.conv2d(
            x_neg,
            W_neg,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        y_neg = F.conv2d(
            x_neg, W_pos, None, self.stride, self.padding, self.dilation, self.groups
        )
        y_neg += F.conv2d(
            x_pos, W_neg, None, self.stride, self.padding, self.dilation, self.groups
        )

        y_conformists = torch.where(y >= 0, y_pos, y_neg)
        y_nonconformists = torch.where(y >= 0, y_neg, y_pos)

        if dissent_suppression < 1:
            y_nonconformists_adjusted = y_nonconformists * (1 - dissent_suppression)
            y_conformists_adjusted = (
                y_conformists + y_nonconformists * dissent_suppression
            )
        else:
            assert dissent_suppression == 1

            y_nonconformists_adjusted = 0
            y_conformists_adjusted = y

        eps = 1e-10
        y_conformists_adjustment_multiplier = y_conformists_adjusted / (
            y_conformists + (eps * y_conformists.sign())
        )
        y_conformists_adjustment_multiplier = (
            y_conformists_adjustment_multiplier.detach()
        )
        assert (
            y_conformists_adjustment_multiplier.sign() >= 0
        ).all(), f"(y_conformists_adjustment_multiplier < 0).mean: shape={y_conformists_adjustment_multiplier.shape}\n\t{y_conformists_adjustment_multiplier[y_conformists_adjustment_multiplier < 0].mean()}"

        # ic(y_conformists_adjustment_multiplier.mean())
        y_correct_grad = (
            y_conformists_adjustment_multiplier * y_conformists
            + y_nonconformists_adjusted
        )

        if self.bias is not None:
            # ic(self.bias.shape, y_correct_grad.shape)
            #: ic| self.bias.shape: torch.Size([6])
            #: y_correct_grad.shape: torch.Size([2, 6, 8, 8])
            #: y: ... channel h w
            y_correct_grad += self.bias.view(-1, 1, 1)

        return_value = swap_backward(result_orig, y_correct_grad)
        if dynamic_obj.get("gbrand_forward_checks_p", True):
            max_error = torch.max(torch.abs(result_orig - y_correct_grad)).item()
            print_diag(f"{prefix}: max error: {max_error}", group="patch_embed_verbose")

            assert torch.allclose(
                result_orig,
                y_correct_grad,
                rtol=1e-03,
                atol=1e-01,
            ), f"{prefix}: result_orig and return_value are not close, max error: {max_error}"

        return return_value


##
def fair_qkv(
    q,
    k,
    v,
    *,
    gradient_mode,
):
    if gradient_mode == "LineX_1":
        q = q.detach()
        k = k.detach()

    elif gradient_mode == "Q":
        q = q.detach()

    elif gradient_mode == "K":
        k = k.detach()

    elif gradient_mode == "AAG":
        #: LineX AttnAttnGrad
        q_correct_grad = q.detach() * q
        k_correct_grad = k.detach() * k
        q = q.detach() + (q_correct_grad - q_correct_grad.detach())
        del q_correct_grad
        k = k.detach() + (k_correct_grad - k_correct_grad.detach())
        del k_correct_grad

    return q, k, v


class AttentionDecomposed(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        linear_layer=LinearDecomposed,
    ):
        super().__init__()
        # self.prefix = "unset."
        self.prefix = f"unset.{self.__class__.__name__}"

        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.dim = dim
        self.qkv_bias = qkv_bias
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # self.fused_attn = use_fused_attn()
        self.fused_attn = False
        # self.fused_attn = True
        #: fused attention doesn't return attention weights
        #: [[id:d22020df-2fbe-4d90-8f81-d922cbc2390b]]

        self.qkv = linear_layer(dim, dim * 3, bias=qkv_bias)
        self.value = linear_layer(self.dim, self.dim, bias=self.qkv_bias)

        self.attn_softmax = NightSoftmax(dim=-1)

        #: Manually set the parameters of self.value from self.qkv:
        with torch.no_grad():
            #: Copy the last third of the qkv weights for the value weights
            self.value.weight.copy_(self.qkv.weight[-self.dim :])
            if self.qkv_bias:
                #: Copy the last third of the qkv biases for the value biases
                self.value.bias.copy_(self.qkv.bias[-self.dim :])

        self.q_norm = (
            norm_layer(self.head_dim, warn_nondecomposed=False)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, warn_nondecomposed=False)
            if qk_norm
            else nn.Identity()
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = linear_layer(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # prefix: 'blocks.0.attn.'
        # torch_shape_get(state_dict): {
        #     'blocks.0.attn.proj.bias': (torch.float32, torch.Size([768])),
        #     'blocks.0.attn.proj.weight': (torch.float32, torch.Size([768, 768])),
        #     'blocks.0.attn.qkv.bias': (torch.float32, torch.Size([2304])),
        #     'blocks.0.attn.qkv.weight': (torch.float32, torch.Size([2304, 768]))
        # }
        ##
        self.prefix = prefix

        # ic(prefix, torch_shape_get(state_dict))
        # ic(args, kwargs)
        qkv_weight = state_dict[f"{prefix}qkv.weight"]
        v_weight = qkv_weight[-self.dim :]
        state_dict[f"{prefix}value.weight"] = v_weight

        qkv_bias_name = f"{prefix}qkv.bias"
        if qkv_bias_name in state_dict:
            qkv_bias = state_dict[f"{prefix}qkv.bias"]
            v_bias = qkv_bias[-self.dim :]
            state_dict[f"{prefix}value.bias"] = v_bias

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        inputs,
        decompose_p="from_config",
        *,
        attn_mask=None,
    ):
        # print_diag('AttentionDecomposed forward', group='indicator')

        decomposition_config = config_from_inputs(inputs)

        gradient_mode = dynamic_obj.attention_gradient_mode
        print_diag(
            f"{self.prefix}: gradient_mode: {gradient_mode}", group="gradient_mode"
        )

        if decomposed_inputs_p(inputs):
            decompose_p = decompose_p_v2(
                decompose_p,
                attributions_v=inputs.attributions_v,
            )
            if decompose_p == "from_config":
                decompose_p = decomposition_config.attention_decompose_p

            if decompose_p:
                attributions_v = "will be set later in the code"
            else:
                attributions_v = inputs.attributions_v

        else:
            decompose_p = False
            attributions_v = None

        if attn_mask is not None:
            # Convert boolean mask to float mask if necessary
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=x.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask

        x = nondecomposed_features(inputs)
        B, N, C = x.shape

        # ic(self.qkv.weight.shape, self.qkv.bias.shape)
        #: self.qkv.weight.shape: torch.Size([2304, 768])
        #: self.qkv.bias.shape: torch.Size([2304])

        if decompose_p:
            with no_grad_maybe(decomposition_config.detach_p):
                v_my_inputs = self.value(inputs)
                v_my = v_my_inputs.features

                # ic(v_my.shape, v_my_inputs.attributions_v.shape)
                # v_my.shape: torch.Size([1, 197, 768])
                # v_my_inputs.attributions_v.shape: torch.Size([1, 197, 198, 1, 768])

                ###
                v_my = rearrange(
                    v_my,
                    "... seq (num_heads head_dim) -> ... num_heads seq head_dim",
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                )
                ##
                # v_my = rearrange(v_my, '... (num_heads head_dim) -> ... num_heads head_dim', num_heads=self.num_heads, head_dim=self.head_dim)
                # v_my = rearrange(v_my, '... seq num_heads head_dim -> ... num_heads seq head_dim')
                ##
                # v_my = v_my.reshape(B, N, self.num_heads, self.head_dim)
                # v_my = v_my.permute(0, 2, 1, 3)
                ###

                v_my_attributions_v = v_my_inputs.attributions_v
                v_my_attributions_v = rearrange(
                    v_my_attributions_v,
                    "... seq patch_attribution pixel_attribution (num_heads head_dim) -> ... num_heads seq patch_attribution pixel_attribution head_dim",
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                )

                # ic(v_my_attributions_v.shape)
                #: v_my_attributions_v.shape: torch.Size([1, 12, 197, 198, 1, 64])

        with DynamicVariables(
            dynamic_obj,
            linear_dissent_suppression=dynamic_obj.qkv_dissent_suppression,
        ):
            qkv = (
                self.qkv(x, warn_nondecomposed=False)
                .reshape(B, N, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
        q, k, v = qkv.unbind(0)
        q, k, v = fair_qkv(
            q,
            k,
            v,
            gradient_mode=gradient_mode,
        )

        if decompose_p:
            # ic(v.shape, v_my.shape)
            #: v_my.shape = v.shape: torch.Size([1, 12, 197, 64])

            if decomposition_config.assertion_check_attention_value_p:
                with torch.no_grad():
                    assert ic(torch.max(torch.abs(v - v_my)).item() == 0.0)

        q, k = self.q_norm(q), self.k_norm(k)

        attn = None

        attn_softmax_gbrand = dynamic_obj.attention_softmax_gradient_mode
        print_diag(
            f"{self.prefix}: attn_softmax_gbrand: {attn_softmax_gbrand}",
            group="gradient_mode",
        )

        attn_mul_gbrand = dynamic_obj.attention_elementwise_mul_gradient_mode
        print_diag(
            f"{self.prefix}: attn_mul_gbrand: {attn_mul_gbrand}", group="gradient_mode"
        )

        if self.fused_attn:
            assert (
                not attn_softmax_gbrand
            ), "attention_softmax_gradient_mode not yet supported on fused attention"
            assert (
                not attn_mul_gbrand
            ), "attention_elementwise_mul_gradient_mode not yet supported on fused attention"

            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p,
            )

        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn += attn_mask
                #: I am not sure if the shape of attn_mask is supposed to work here, but it should probably be standard.

            #: dim: A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
            # attn = attn.softmax(dim=-1)
            attn = self.attn_softmax(
                attn,
                gradient_mode=attn_softmax_gbrand,
                competition_scale=dynamic_obj.attention_softmax_competition_scale,
            )

            if attn_mul_gbrand == "D2":
                #: DU needs to be done below where we construct the output.
                ##
                attn = swap_backward(attn, attn / 4)
                #: attn is the result of multiplying keys and queries (bilinear multiplication?), so it needs another division by 2

                v = swap_backward(v, v / 2)

            attn = self.attn_drop(attn)

            if dynamic_obj.raw_attention_grad_store_p:
                attn.requires_grad_(True)
                #: Our =gradient_mode= shenanigans might have prevented the gradient from being computed for =attn=, so we explicitly ask for it.

            # ic(attn.shape, attn)
            #: attn.shape: torch.Size([1, 12, 197, 197])
            #: attn: (B, head_num, to_seq, from_seq)

            x = attn @ v

            if attn_mul_gbrand == "DU":
                x = swap_backward(x, x / 3)
                #: The input feeds into the output from three multiplicative sources

            # ic(x.shape)
            #: x.shape: torch.Size([1, 12, 197, 64])

        #: Value Storage
        delattr_force(self, "stored_value")
        if dynamic_obj.value_store_p:
            self.stored_value = v.detach().cpu()  # .clone()

        delattr_force(self, "stored_value_grad")
        if dynamic_obj.value_grad_store_p:

            def store_value_grad(grad):
                # ic(torch_shape_get(grad))
                self.stored_value_grad = grad.detach().cpu()

            tensor_register_hook(v, store_value_grad)
        ##

        #: MultiHeadAttention Storage
        delattr_force(self, "stored_mha")
        if dynamic_obj.mha_store_p:
            self.stored_mha = x.detach().cpu()  # .clone()

        delattr_force(self, "stored_mha_grad")
        if dynamic_obj.mha_grad_store_p:

            def store_mha_grad(grad):
                # ic(torch_shape_get(grad))
                self.stored_mha_grad = grad.detach().cpu()

            tensor_register_hook(x, store_mha_grad)
        ##

        if decompose_p:
            with no_grad_maybe(decomposition_config.detach_p):
                # ic(attn.shape, v_my_attributions_v.shape)
                #:  attn.shape: torch.Size([1, 12, 197, 197])
                #: v_my_attributions_v.shape: torch.Size([1, 12, 197, 198, 1, 64])

                v_my_attributions_v = einops.einsum(
                    attn,
                    v_my_attributions_v,
                    "... num_heads token_to token_from, ... num_heads token_from patch_attribution pixel_attribution head_dim -> ... num_heads token_to patch_attribution pixel_attribution head_dim",
                )

                # ic(v_my_attributions_v.shape)
                #: v_my_attributions_v.shape: torch.Size([1, 12, 197, 198, 1, 64])

                # ic(attn[..., :1, :1, :])
                # ic(attributions_n_get(v_my_attributions_v, 0, zero_out_error=False))

                #: Shape rearrangement is not yet supported in einsum.
                v_my_attributions_v = rearrange(
                    v_my_attributions_v,
                    "... num_heads to patch_attribution pixel_attribution head_dim -> ... to patch_attribution pixel_attribution (num_heads head_dim)",
                )

                # ic(v_my_attributions_v.shape)
                #: v_my_attributions_v.shape: torch.Size([1, 197, 198, 1, 768])

                # ic(attributions_n_get(v_my_attributions_v, 0, zero_out_error=False))
                attributions_v = v_my_attributions_v

        x = x.transpose(1, 2).reshape(B, N, C)
        inputs = simple_obj_update(
            inputs,
            features=x,
            attributions_v=attributions_v,
            del_p="can",
        )

        inputs = self.proj(inputs)

        inputs = nondecomposed_forward(self.proj_drop, inputs)

        delattr_force(self, "stored_rawattn_grad")
        if dynamic_obj.raw_attention_grad_store_p:

            def store_rawattn_grad(grad):
                # ic(torch_shape_get(grad))
                self.stored_rawattn_grad = grad.detach().cpu()

            tensor_register_hook(attn, store_rawattn_grad)

        if dynamic_obj.raw_attention_store_p:
            attn_cpu = attn.detach().cpu()  # .clone()
            self.stored_rawattn = attn_cpu
        else:
            delattr_force(self, "stored_rawattn")

        return inputs
        # return simple_obj_update(
        #     inputs,
        #     (f'{self.prefix}rawattn'),
        #     (attn_cpu),
        #     del_p='can',
        # )


##
class FairMultiheadAttention(nn.MultiheadAttention):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.prefix = f"unset.{self.__class__.__name__}"

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.prefix = prefix
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *args,
        **kwargs,
    ):
        gradient_mode = dynamic_obj.attention_gradient_mode
        print_diag(
            f"{self.prefix}: gradient_mode: {gradient_mode}", group="gradient_mode"
        )

        if not gradient_mode or gradient_mode in [
            "NG",
        ]:
            pass

        elif gradient_mode == "LineX_1":
            query = query.detach()
            key = key.detach()

        else:
            raise NotImplementedError(f"Unsupported gradient_mode: {gradient_mode}")

        return super().forward(
            query,
            key,
            value,
            *args,
            **kwargs,
        )


##
class FairOCAttention(AttentionDecomposed):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        scaled_cosine: bool = False,
        scale_heads: bool = False,
        logit_scale_max: float = math.log(1.0 / 0.01),
        batch_first: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        self.logit_scale_max = logit_scale_max
        self.batch_first = batch_first

        if scaled_cosine:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1)))
            )
        else:
            self.logit_scale = None

        if scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Handle the conversion from in_proj_weight and in_proj_bias to qkv.weight and qkv.bias
        old_prefix = prefix
        prefix = prefix + "qkv."

        if old_prefix + "in_proj_weight" in state_dict:
            in_proj_weight = state_dict.pop(old_prefix + "in_proj_weight")
            state_dict[prefix + "weight"] = in_proj_weight

        if old_prefix + "in_proj_bias" in state_dict:
            in_proj_bias = state_dict.pop(old_prefix + "in_proj_bias")
            state_dict[prefix + "bias"] = in_proj_bias

        # Handle other parameter name changes
        if old_prefix + "out_proj.weight" in state_dict:
            state_dict[old_prefix + "proj.weight"] = state_dict.pop(
                old_prefix + "out_proj.weight"
            )

        if old_prefix + "out_proj.bias" in state_dict:
            state_dict[old_prefix + "proj.bias"] = state_dict.pop(
                old_prefix + "out_proj.bias"
            )

        # Remove parameters that are not used in the new implementation
        keys_to_remove = [
            "logit_scale",
            "head_scale",
        ]
        for key in keys_to_remove:
            full_key = old_prefix + key
            if full_key in state_dict:
                state_dict.pop(full_key)
                unexpected_keys.remove(full_key)

        # Call the parent class method to handle the rest of the loading process
        super()._load_from_state_dict(
            state_dict,
            old_prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if not self.batch_first:
            x = x.transpose(0, 1)

        if self.scaled_cosine or self.scale_heads or self.logit_scale is not None:
            raise NotImplementedError(
                "scaled_cosine, scale_heads, and logit_scale are not implemented in this version"
            )

        # Call the parent class forward method
        x = super().forward(x, attn_mask=attn_mask)

        if not self.batch_first:
            x = x.transpose(0, 1)

        return x


##
def phi(x):
    return (1 + torch.erf(x / math.sqrt(2))) / 2.0


def normal_pdf(x):
    return torch.exp(-(x**2) / 2) / math.sqrt(2.0 * math.pi)


def gelu_deriv(x):
    #: equals PyTorch's output on torch.linspace(-30, 30, steps=5000)
    ##
    return phi(x) + x * normal_pdf(x)


def gelu_deriv2(x):
    #: [[https://api.semanticscholar.org/arXiv:2104.02523][An Analysis of State-of-the-art Activation Functions For Supervised Deep Neural Network]]
    #: This does not equal PyTorch's own outputs.
    ##
    deriv = 0.5 * torch.tanh(0.0356774 * x**3 + 0.797885 * x)
    deriv = deriv + 0.5
    deriv = deriv + (0.0535161 * x**3 + 0.398942 * x) * (
        torch.cosh(0.0356774 * x**3 + 0.797885 * x) ** -2
    )


def attributions_mul(attributions_v, m, test_p=False):
    ##
    # attributions_v = torch.einsum('...d,...abd->...abd', m, attributions_v)
    ##
    if test_p:
        attributions_v_v1 = torch.einsum("...d,...abd->...abd", m, attributions_v)

    #: @perf/memory @inplace
    torch.mul(
        attributions_v,
        m.unsqueeze(-2).unsqueeze(-2),
        out=attributions_v,
    )

    if test_p:
        ic(torch.equal(attributions_v_v1, attributions_v))
        ic(torch.allclose(attributions_v_v1, attributions_v))
        ic(torch.max(torch.abs(attributions_v_v1 - attributions_v)))

    return attributions_v


class GELUDecomposed(nn.GELU):
    DECOMPOSED_P = True

    def __init__(self, derivative_fn=gelu_deriv, forward_fn=F.gelu, *args, **kwargs):
        self.derivative_fn = derivative_fn
        self.forward_fn = forward_fn

        super().__init__(*args, **kwargs)
        self.prefix = f"unset.{self.__class__.__name__}"

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.prefix = prefix
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, inputs, warn_nondecomposed=False):
        if self.approximate and self.approximate != "none":
            ic(self.approximate)

            raise NotImplementedError("Approximate GELU not supported yet")
            #: We need to implement LineX for these.

        # print_diag('GELUDecomposed forward', group='indicator')

        if decomposed_inputs_p(inputs):
            decomposition_config = inputs.decomposition_config
            gelu_bias_decomposition_mode = (
                decomposition_config.gelu_bias_decomposition_mode
            )

            features = inputs.features
            features_out = self.forward_fn(features)

            attributions_v = inputs.attributions_v
            if decompose_p_v2(
                decomposition_config.gelu_decompose_p,
                attributions_v=inputs.attributions_v,
            ):
                with no_grad_maybe(decomposition_config.detach_p):
                    if decomposition_config.GELU_decompose_mode == "taylor":
                        m = self.derivative_fn(features)
                        features_out_biasless = m * features
                        bias = features_out - features_out_biasless

                        # ic(m.shape, bias.shape, attributions_v.shape)
                        #: m.shape: torch.Size([1, 197, 3072])
                        #: bias.shape: torch.Size([1, 197, 3072])
                        #: attributions_v.shape: torch.Size([1, 197, 198, 1, 3072])

                        attributions_v = attributions_mul(attributions_v, m)

                        attributions_v = bias_decomposer(
                            attributions_v,
                            bias,
                            bias_decomposition_mode=gelu_bias_decomposition_mode,
                        )
                    elif decomposition_config.GELU_decompose_mode == "zo":
                        eps = 1e-12
                        m = features_out / (features + eps)
                        features_out_biasless = m * features
                        bias = features_out - features_out_biasless
                        max_abs_bias = torch.max(torch.abs(bias)).item()
                        assert (
                            max_abs_bias < 1e-5
                        ), f"The maximum absolute value of bias in the zo method is not almost zero: {max_abs_bias}"
                        # print_diag(f"zo bias max: {max_abs_bias}",)

                        attributions_v = attributions_mul(attributions_v, m)

            return simple_obj_update(
                inputs,
                features=features_out,
                attributions_v=attributions_v,
                del_p="can",
            )
        else:
            if warn_nondecomposed:
                print_diag(
                    f"GELUDecomposed: nondecomposed inputs at `{stacktrace_caller_line()}`",
                    group="warning.nondecomposed",
                )

            gradient_mode = dynamic_obj.gelu_gradient_mode
            print_diag(
                f"{self.prefix}: gradient_mode: {gradient_mode}", group="gradient_mode"
            )

            features = inputs
            result_orig = F.gelu(features, approximate=self.approximate)

            if gradient_mode in ["LineX_1", "LineX_ZO", "ZO"]:
                #: ZO Linearization

                #: ZO_v3 [jalali:1403/03/05/22:58]
                with torch.no_grad(): #: saves memory
                    gate = 0.5 * (1.0 + torch.erf(features / torch.sqrt(torch.tensor(2.0))))
                gate = gate.detach()
                result = features * gate

                ##
                # with torch.no_grad():
                #     if True:
                #         #: [jalali:1402/12/06]
                #         #: After seeing the segmentation results of LX-NoAct, I changed the implementation to this.
                #         # print_diag(f"{self.prefix}: using GELU ZO_v2", group="gradient_mode")

                #         eps = 1e-6

                #         sign_features = torch.sign(features)
                #         sign_features[sign_features == 0] = 1
                #         features_adjusted = torch.where(
                #             torch.abs(features) <= eps,
                #             eps * sign_features,
                #             features,
                #         )
                #         # print(f"eps={eps}\nmin abs features:{torch.min(torch.abs(features))}\nmin features_adjusted: {torch.min(torch.abs(features_adjusted))}")

                #         m = torch.div(result_orig, features_adjusted)
                #         del sign_features
                #         del features_adjusted
                #     else:
                #         eps = 1e-12
                #         m = torch.div(result_orig, features + eps)

                # result = torch.mul(m, features)
                ##

                if dynamic_obj.gbrand_forward_checks_p:
                    assert torch.allclose(
                        result,
                        result_orig,
                        ##
                        #: ZO_v2, ZO_v3
                        rtol=1e-04,
                        atol=1e-04,
                        ##
                    ), f"GELU ZO is not close enough:\n\tmax error={torch.max(torch.abs(result - result_orig))}\n\tmax result_orig: {torch.max(torch.abs(result_orig))}\n\tmax result: {torch.max(torch.abs(result))}\n\tmax m: {torch.max(torch.abs(m))}"
                    #: This assertion might need a higher tolerance to be always true, as =eps= is introducing bias.

                result_orig_correct_grad = swap_backward(result_orig, result)

                my_bias = None
            else:
                #: Normal Gradients (Taylor Linearization)

                result_orig_correct_grad = result_orig

                with torch.no_grad():  #: Second-order gradients might not appreciate this =no_grad=, or they actually might.
                    m = self.derivative_fn(features)
                    features_out_biasless = m * features
                    my_bias = result_orig - features_out_biasless

            return_value = result_orig_correct_grad

            ##
            layer_name = "gelu"
            #: @duplicateCode/6b32d3a832249d36819faf9531ae5ed1
            delattr_force(self, "stored_out_grad")
            delattr_force(self, "stored_bias_attributions")
            delattr_force(self, "stored_bias_attributions_raw")
            if dynamic_obj.get(f"{layer_name}_attribute_bias_p"):
                tensor_register_hook(
                    return_value,
                    partial(
                        store_grad_and_bias_attributions,
                        bias=my_bias,
                        store_in=self,
                        layer_name=layer_name,
                    ),
                )
            ##

            return return_value


##
def quick_gelu(features):
    return features * torch.sigmoid(1.702 * features)


def quick_gelu_deriv(features):
    sigmoid_out = torch.sigmoid(1.702 * features)
    deriv = sigmoid_out + torch.mul(
        features, (1.702 * torch.mul(sigmoid_out, 1 - sigmoid_out))
    )
    return deriv


class QuickGELUDecomposed(GELUDecomposed):
    def __init__(
        self, derivative_fn=quick_gelu_deriv, forward_fn=quick_gelu, *args, **kwargs
    ):
        raise NotImplementedError(
            "QuickGELU implementation needs to be updated for gate detaching"
        )

        super().__init__(
            *args, derivative_fn=derivative_fn, forward_fn=forward_fn, **kwargs
        )


##
class MLPDecomposed(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        assert use_conv == False, "use_conv not decomposed currently"

        super().__init__()
        if act_layer == nn.GELU:
            act_layer = GELUDecomposed
        elif decomposed_module_p(act_layer):
            # ic(act_layer)
            pass
        else:
            raise NotImplementedError(f"act_layer {act_layer} not decomposed currently")

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        # linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        linear_layer = LinearDecomposed

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        # ic(fc1 hidden_features, out_features, bias[0])

        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        # ic(fc2, hidden_features, out_features, bias[0])

        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, inputs, warn_nondecomposed=False):
        # print_diag('MLPDecomposed forward', group='indicator')

        if decomposed_inputs_p(inputs):
            decomposition_config = inputs.decomposition_config
            decompose_p = decompose_p_v2(
                decomposition_config.mlp_decompose_p,
                attributions_v=inputs.attributions_v,
            )

            with DynamicVariables(
                dynamic_obj,
                decompose_p=decompose_p,
            ):
                inputs = self.fc1(inputs)
                inputs = self.act(inputs)

                features = inputs.features
                features = self.drop1(features)
                inputs = simple_obj_update(
                    inputs,
                    features=features,
                    del_p="can",
                )

                inputs = self.fc2(inputs)

                features = inputs.features
                features = self.drop2(features)
                inputs = simple_obj_update(
                    inputs,
                    features=features,
                    del_p="can",
                )

                return inputs
        else:
            if warn_nondecomposed:
                print_diag(
                    f"MLPDecomposed: nondecomposed inputs at `{stacktrace_caller_line()}`",
                    group="warning.nondecomposed",
                )

            with DynamicVariables(
                dynamic_obj,
                linear_dissent_suppression=dynamic_obj.mlp_linear_dissent_suppression,
            ):
                x = inputs
                x = self.fc1(x)
                x = self.act(x)
                x = self.drop1(x)
                x = self.fc2(x)
                x = self.drop2(x)
                return x


##
class NightSoftmax(nn.Softmax):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prefix = f"unset.{self.__class__.__name__}"

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.prefix = prefix
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        x,
        gradient_mode="from_config",
        competition_scale="from_config",
    ):
        dim = self.dim
        prefix = self.prefix

        delattr_force(self, "stored_out_grad")
        delattr_force(self, "stored_bias_attributions")
        delattr_force(self, "stored_bias_attributions_raw")

        if gradient_mode == "from_config":
            gradient_mode = dynamic_obj.softmax_gradient_mode

        if competition_scale == "from_config":
            competition_scale = dynamic_obj.softmax_competition_scale

        if gradient_mode in ["XSC"]:
            return h_softmax_fn.apply(
                x,
                dim,
                gradient_mode,
                competition_scale,
                prefix,
            )

        print_diag(f"{prefix}: gradient_mode: {gradient_mode}", group="gradient_mode")
        result_orig = super().forward(x)

        if gradient_mode in ["detach_denom", "detach_denom_nb"]:
            #: Subtract the max value from each row for numerical stability
            x_max = torch.max(x, dim=dim, keepdim=True)[0]
            x_max = x_max.detach()
            #: We do NOT want to exclude the contribution of the biggest contributor, so we need to detach this x_max.

            x_exp = torch.exp(x - x_max)

            denom = torch.sum(x_exp, dim=dim, keepdim=True)

            denom = denom.detach()

            result = x_exp / denom
            if gradient_mode == "detach_denom_nb":
                pass
            else:
                #: The grad is also =result=
                result_d = result.detach()
                my_bias = result_d - (result_d * x.detach())
                my_bias *= 2
                #: to counteract the scaling for KQ bilinearity

                layer_name = "softmax"

                if True:
                    tensor_register_hook(
                        result,
                        partial(
                            store_grad_and_bias_attributions,
                            bias=my_bias,
                            store_in=self,
                            layer_name=layer_name,
                            sum_dim=[-2, -3],
                            #: result (attn): [batch, head, to, from]
                        ),
                    )
                ##

            if dynamic_obj.softmax_swap_check_p:
                assert torch.allclose(
                    result,
                    result_orig,
                    ##
                    rtol=1e-04,
                    atol=1e-04,
                    ##
                ), f"Softmax is not close enough:\n\tmax error={torch.max(torch.abs(result - result_orig))}\n\tmax result_orig: {torch.max(torch.abs(result_orig))}\n\tmax result: {torch.max(torch.abs(result))}"

            result = swap_backward(result_orig, result)

        elif gradient_mode in ["ZO", "ZR"]:
            eps = 1e-12
            multiplier = result_orig.detach() / (x.detach() + eps)

            if gradient_mode == "ZO":
                if False:  #: uses memory we don't really have
                    print_diag(
                        f"{prefix}: mean of negative logits: {x[x < 0].mean()}",
                        group="stats",
                    )
                    print_diag(
                        f"{prefix}: mean of positive logits: {x[x > 0].mean()}",
                        group="stats",
                    )

                result_correct_grad = multiplier * x
            elif gradient_mode == "ZR":
                result_correct_grad = multiplier * F.relu(x)

            result = swap_backward(result_orig, result_correct_grad)
            del result_correct_grad

        else:
            result = result_orig

        return result


class h_softmax_fn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        #: We can't call Function.apply with kwargs, so the order of these arguments matter.
        inputs,
        dim,
        gradient_mode=None,
        competition_scale=1,
        prefix="",
        dtype=None,
    ):
        if dtype is not None:
            inputs = inputs.to(dtype)

        outputs = F.softmax(inputs, dim=dim, dtype=dtype)

        ctx.save_for_backward(inputs, outputs)
        ctx.dim = dim
        ctx.gradient_mode = gradient_mode
        ctx.competition_scale = competition_scale
        print_diag(
            f"{prefix}h_softmax: gradient_mode: {gradient_mode}, competition_scale: {competition_scale}",
            group="gradient_mode",
        )

        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        inputs, outputs = ctx.saved_tensors
        dim = ctx.dim
        gradient_mode = ctx.gradient_mode
        competition_scale = ctx.competition_scale

        if gradient_mode == "XSC":
            #: Exclude Self-Competition
            competition_multiplier = competition_scale * outputs
            competition = competition_multiplier * torch.sum(
                grad_output * outputs, dim=dim, keepdim=True
            )
            grad_input = (
                1 + competition_multiplier
            ) * outputs * grad_output - competition
        else:
            #: Normal gradients
            grad_input = outputs * (
                grad_output - torch.sum(grad_output * outputs, dim=dim, keepdim=True)
            )

        return (grad_input, None, None, None, None)


class SoftmaxDecomposed(nn.Softmax):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs, no_grad_p=True):
        # ic(inputs.features.shape, inputs.attributions_v.shape)

        if decomposed_inputs_p(inputs):
            decomposition_config = inputs.decomposition_config

            dim = self.dim

            features = inputs.features

            output_orig = super().forward(features)

            attributions_v = inputs.attributions_v

            if decompose_p_v2(
                decomposition_config.softmax_decompose_p,
                attributions_v=inputs.attributions_v,
            ):
                with no_grad_maybe(no_grad_p):
                    attribution_sources_count = torch.prod(
                        torch.tensor(attributions_v.shape[-3:-1])
                    )

                    m, _ = torch.max(features, axis=dim, keepdims=True)
                    m_expanded = m.unsqueeze(1).unsqueeze(1)

                    features_normalized = features - m
                    numerator = torch.exp(features_normalized)
                    denom = torch.sum(numerator, axis=dim, keepdims=True)

                    output = numerator / denom

                    # denom_attr = torch.log(denom) - m
                    denom_attr = torch.log(denom)
                    denom_attr /= attribution_sources_count
                    denom_attr = denom_attr.unsqueeze(1).unsqueeze(1)

                    # ic(attributions_v.shape, denom_attr.shape)
                    # ic(m, denom, denom_attr, attribution_sources_count)
                    # ic| attributions_v.shape: torch.Size([1, 198, 1, 1000])
                    #     denom_attr.shape: torch.Size([1, 1])
                    # ic| m: tensor([[13.6208]], device='cuda:0')
                    #     denom: tensor([[1.0224]], device='cuda:0')
                    #     denom_attr: tensor([[0.0001]], device='cuda:0')
                    #     attribution_sources_count: tensor(198)

                    # embed()
                    attributions_v = (
                        attributions_v - (m_expanded / attribution_sources_count)
                    ) - denom_attr

                    # ic(output_orig - output)
                    assert torch.allclose(output_orig, output)

                inputs = simple_obj_update(
                    inputs,
                    features=output_orig,
                    attributions_v=attributions_v,
                    del_p="can",
                )

                check_attributions_v2(
                    inputs=simple_obj(
                        features=torch.log(inputs.features),
                        attributions_v=inputs.attributions_v,
                    ),
                    atol=1e-4,
                    group="check_attributions.end",
                    prefix="softmaxv2.end.",
                    # print_all=True,
                )

                return inputs
        else:
            return super().forward(inputs)

    def forward_v1(self, inputs, no_grad_p=True):
        #: @broken?
        ##
        # ic(inputs.features.shape, inputs.attributions_v.shape)

        if decomposed_inputs_p(inputs):
            with no_grad_maybe(no_grad_p):
                dim = self.dim

                features = inputs.features

                output_orig = super().forward(features)

                features = features.double()
                attributions_v = inputs.attributions_v
                attributions_v = attributions_v.double()

                check_attributions_v2(
                    simple_obj(
                        features=features,
                        attributions_v=attributions_v,
                    ),
                    print_all=False,
                    assert_p=True,
                    prefix="softmax.inputs.",
                )

                # check_attributions_v2(
                #     simple_obj(
                #     features=torch.log(torch.exp(features)),
                #     attributions_v=torch.log(torch.exp(attributions_v)),
                #         ),
                #     print_all=False,
                #     assert_p=True,
                #     prefix='softmax.inputs log exp.',
                # )

                attribution_sources_count = torch.prod(
                    torch.tensor(attributions_v.shape[-3:-1])
                )

                m, _ = torch.max(features, axis=dim, keepdims=True)
                m_expanded = m.unsqueeze(1).unsqueeze(1)

                #: @notImplemented nonzero m not currently implemented
                m = 0.0
                m_expanded = 0.0

                features_normalized = features - m
                numerator = torch.exp(features_normalized)

                attributions_v_normalized = attributions_v - m_expanded
                attributions_numerator = torch.exp(attributions_v_normalized)
                attributions_numerator_log = torch.log(attributions_numerator)

                denom = torch.sum(numerator, axis=dim, keepdims=True)
                attributions_denom = (
                    torch.pow(denom, 1 / attribution_sources_count)
                    .unsqueeze(1)
                    .unsqueeze(1)
                )

                output = numerator / denom
                output = output.float()

                ic(denom, attributions_denom)
                # ic(attributions_numerator)

                # numerator_reconstructed = multiply_attributions(attributions_numerator)
                ic(attributions_numerator.shape, numerator.shape)
                numerator_log_reconstructed = sum_attributions(
                    attributions_numerator_log
                )
                numerator_log = torch.log(numerator)

                # ic(numerator_log.shape, numerator_log_reconstructed.shape)
                # ic(numerator_log, numerator_log_reconstructed)
                check_attributions_v2(
                    simple_obj(
                        features=numerator_log,
                        attributions_v=attributions_numerator_log,
                    ),
                    print_all=False,
                    assert_p=True,
                    prefix=f"softmax.inputs log exp m={m}.",
                )
                #: passes with m=0
                #: err_abs_mean=6.547570930255938e-06
                #: err_abs_max=2.348513953620568e-05

                # ic(numerator.shape, numerator_reconstructed.shape)
                # ic(numerator, numerator_reconstructed)

                attributions_v = attributions_numerator / attributions_denom
                attributions_v = attributions_v.float()

                # ic(output_orig - output)
                assert torch.allclose(output_orig, output)

                inputs = simple_obj_update(
                    inputs,
                    features=output_orig,
                    attributions_v=attributions_v,
                    del_p="can",
                )

                check_attributions_v2(
                    inputs=inputs,
                    reconstructor=multiply_attributions,
                    atol=1e-2,
                    group="check_attributions.end",
                    prefix="softmax.end.",
                    # print_all=True,
                )

                return inputs
        else:
            return super().forward(inputs)


##
class LayerScaleDecomposed(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        # self.prefix = 'unset.'
        self.prefix = f"unset.{self.__class__.__name__}"

        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.prefix = prefix
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward_nd(self, features):
        #: [[https://pytorch.org/docs/stable/generated/torch.mul.html#torch.mul][torch.mul — PyTorch 2.0 documentation]]
        #: elementwise multiplication with broadcasting

        # ic(self.inplace, self.gamma.shape, features.shape)

        return features.mul_(self.gamma) if self.inplace else features * self.gamma

    def forward(self, inputs):
        # print_diag('LayerScaleDecomposed forward', group='indicator')

        warn_nondecomposed = False

        if decomposed_inputs_p(inputs):
            decomposition_config = inputs.decomposition_config

            features = inputs.features
            features_out = self.forward_nd(features)

            attributions_v = inputs.attributions_v
            if decompose_p_v2(
                decomposition_config.layerscale_decompose_p,
                attributions_v=inputs.attributions_v,
            ):
                with no_grad_maybe(decomposition_config.detach_p):
                    attributions_v = self.forward_nd(attributions_v)

            inputs = simple_obj_update(
                inputs,
                features=features_out,
                attributions_v=attributions_v,
                del_p="can",
            )

            check_attributions_v2(
                inputs,
                prefix=self.prefix,
                assert_p=decomposition_config.assertion_check_attributions_p,
            )

            return inputs
        else:
            if warn_nondecomposed:
                print_diag(
                    f"LayerScaleDecomposed: nondecomposed inputs at `{stacktrace_caller_line()}`",
                    group="warning.nondecomposed",
                )

            features = inputs
            return self.forward_nd(features)


##
def dynamic_config_contexts(*, decomposition_config=None, decompose_p=True):
    context_managers = []

    if not decompose_p:
        context_managers.append(
            DynamicVariables(
                dynamic_vars,
                print_diag_enabled_groups=lst_filter_out(
                    dynamic_get(dynamic_vars, "print_diag_enabled_groups"),
                    [
                        "warning.nondecomposed",
                    ],
                ),
            )
        )

    return context_managers


##
def mixer_transpose(inputs, prefix=None):
    if prefix is None:
        prefix = ""
    prefix += "mixer_transpose."

    if decomposed_inputs_p(inputs):
        decomposition_config = inputs.decomposition_config

        features = inputs.features
        #: torch.Size([1, 196, 768])

        features = features.transpose(1, 2)
        #: torch.Size([1, 768, 196])

        attributions_v = inputs.attributions_v
        #: torch.Size([1, 196, 197, 1, 768])

        if decompose_p_v2(
            decomposition_config.token_mixer_decompose_p,
            attributions_v=inputs.attributions_v,
        ):
            attributions_v = attributions_v.transpose(1, -1)
            #: torch.Size([1, 768, 197, 1, 196])

        inputs = simple_obj_update(
            inputs,
            features=features,
            attributions_v=attributions_v,
            del_p="can",
        )

        check_attributions_v2(
            inputs,
            prefix=prefix,
            assert_p=(decomposition_config.assertion_check_attributions_p),
        )

        return inputs
    else:
        inputs = inputs.transpose(1, 2)
        return inputs


##
def fair_gated_mlp(
    x,
    gradient_mode="from_config",
    prefix="",
):
    if gradient_mode == "from_config":
        gradient_mode = dynamic_obj.mlp_elementwise_mul_gradient_mode

    print_diag(
        f"{prefix}mlp_elementwise_mul_gradient_mode: {gradient_mode}",
        group="gradient_mode",
    )

    if gradient_mode in ["DU"]:
        x = swap_backward(x, x / 2)

    elif gradient_mode in ["D2"]:
        raise Exception(
            "mlp_elementwise_mul_gradient_mode D2 is the same as DU and so not supported."
        )

    return x


##
class FairSwiGLU(SwiGLU):
    def __init__(self, *args, **kwargs):
        #: The mother class is not storing its init args so `mod_init_from` fails.
        super().__init__(*args, **kwargs)
        self.prefix = f"unset.{self.__class__.__name__}"

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.prefix = prefix
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x

        x = fair_gated_mlp(x, prefix=self.prefix)

        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

        ##
        # # Call the original SwiGLU forward method
        # x = super().forward(x)

        # # Apply the fair_gated_mlp
        # x = fair_gated_mlp(x, prefix=self.prefix)

        # return x


class FairGluMlp(GluMlp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = f"unset.{self.__class__.__name__}"

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.prefix = prefix
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=self.chunk_dim)

        x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        x = fair_gated_mlp(x, prefix=self.prefix)

        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

        ##
        # Call the original GluMlp forward method
        # x = super().forward(x)

        # # Apply the fair_gated_mlp
        # x = fair_gated_mlp(x, prefix=self.prefix)

        return x


##
class FairSiLU(nn.SiLU):
    def __init__(self, inplace: bool = False):
        super().__init__(inplace)
        self.prefix = f"unset.{self.__class__.__name__}"

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.prefix = prefix
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        gradient_mode = dynamic_obj.gelu_gradient_mode
        print_diag(
            f"{self.prefix}: gradient_mode: {gradient_mode}", group="gradient_mode"
        )

        sigmoid_x = F.sigmoid(input)

        if gradient_mode in ["LineX_1", "LineX_ZO", "ZO"]:
            sigmoid_x = sigmoid_x.detach()

        if self.inplace:
            return input.mul_(sigmoid_x)
        else:
            return input * sigmoid_x


FairSwiGLUPacked = partial(FairGluMlp, act_layer=FairSiLU, gate_last=False)


##
def n2u_sum_off_diagonals(
    *,
    y,
    grad_output,
    dim,
):
    # Ensure that dim is the last dimension for simpler indexing
    if dim != -1 and dim != y.dim() - 1:
        # Bring the summation dimension to the last dimension
        permute_order = list(range(y.dim()))
        permute_order[dim], permute_order[-1] = permute_order[-1], permute_order[dim]
        y = y.permute(*permute_order)
        grad_output = grad_output.permute(*permute_order)

    # Get the size of the last dimension
    hidden_size = y.size(-1)
    batch_shape = y.shape[:-1]

    # Compute the element-wise product
    prod = y * grad_output  # Shape: [batch_shape, hidden_size]

    # Expand prod to [batch_shape, hidden_size, hidden_size]
    prod_expanded = prod.unsqueeze(-2).expand(*batch_shape, hidden_size, hidden_size)

    # Create an identity matrix to zero out diagonal elements
    mask = (1 - torch.eye(hidden_size, device=y.device, dtype=y.dtype)).unsqueeze(0)
    # Adjust the mask size to match batch dimensions
    mask = mask.expand(*batch_shape, hidden_size, hidden_size)

    # Apply the mask
    prod_masked = prod_expanded * mask  # Shape: [batch_shape, hidden_size, hidden_size]

    # Sum over the last dimension (excluding diagonal elements)
    s = prod_masked.sum(dim=-1)  # Shape: [batch_shape, hidden_size]

    # If needed, permute back to the original dimension order
    if dim != -1 and dim != y.dim() - 1:
        # Invert the permutation
        inverse_permute_order = list(range(y.dim()))
        inverse_permute_order[-1], inverse_permute_order[dim] = (
            inverse_permute_order[dim],
            inverse_permute_order[-1],
        )
        s = s.permute(*inverse_permute_order)

    return s


class NormalizeToUnitVectorSelfPos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, gradient_mode, eps=1e-12):
        norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, ord=2).clamp_min(eps)

        y = x / norm  # Normalize x to unit vector y = x / ||x||
        ctx.save_for_backward(y, norm)  # Save y and norm for backward pass
        ctx.dim = dim
        ctx.gradient_mode = gradient_mode
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, norm = ctx.saved_tensors  # Retrieve saved tensors
        dim = ctx.dim
        gradient_mode = ctx.gradient_mode

        hidden_shape = y.shape[-1]

        emphasize_pos = rget(gradient_mode, rf"""^EP({float_pattern})$""")
        if emphasize_pos:
            emphasize_pos = float(emphasize_pos)
            gradient_mode = "self_pos_v1"
            ic(emphasize_pos, gradient_mode)

        if gradient_mode in [None, "self_pos_v1"]:
            s = y * grad_output
            #: [..., hidden_shape]

            s = s.sum(dim=dim, keepdim=True)
            #: [..., 1]

            if gradient_mode == "self_pos_v1":
                #: Expand s to [..., hidden_shape]:
                s = s.expand(y.size())
                #: [[https://pytorch.org/docs/stable/generated/torch.Tensor.expand_as.html][torch.Tensor.expand_as — PyTorch 2.5 documentation]]

                s = s - grad_output * y

        elif gradient_mode == "self_pos":
            s = n2u_sum_off_diagonals(y, grad_output, dim)

        if emphasize_pos:
            grad_output = grad_output * emphasize_pos

        grad_input = (grad_output - y * s) / norm

        return grad_input, None, None, None  # None for dim, gradient_mode, and eps


def normalize_to_unit_vector(
    x,
    *,
    dim=-1,
    gradient_mode="from_config",
    # gradient_mode='from_config_LN',
    eps=1e-12,  # Default value of eps as a keyword argument
    prefix="",
):
    if gradient_mode == "from_config_LN":
        gradient_mode = dynamic_obj.layer_norm_gradient_mode
    elif gradient_mode == "from_config":
        gradient_mode = dynamic_obj.normalize_to_unit_vector_gradient_mode

    print_diag(
        f"{prefix}normalize_to_unit_vector: gradient_mode: {gradient_mode}",
        group="gradient_mode",
    )

    if gradient_mode in [
        "self_pos",
        "self_pos_v1",
    ] or (gradient_mode and gradient_mode.startswith("EP")):
        return NormalizeToUnitVectorSelfPos.apply(x, dim, gradient_mode, eps)

    else:
        # Shared logic for both "LineX_1", "LX1" and the default case
        norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, ord=2).clamp_min(eps)
        #: [[https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html][torch.linalg.vector_norm — PyTorch 2.5 documentation]]

        if gradient_mode in ["LineX_1", "LX1"]:
            norm = norm.detach()  # Detach for "LineX_1", "LX1" mode

        return x / norm


##
class FairAttentionPoolLatent(nn.Module):
    """Attention pooling w/ latent query"""

    fused_attn: torch.jit.Final[bool]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.prefix = prefix
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        embed_dim: int = None,
        num_heads: int = 8,
        feat_size: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        latent_len: int = 1,
        latent_dim: int = None,
        pos_embed: str = "",
        pool_type: str = "token",
        norm_layer: Optional[nn.Module] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        self.attn_softmax = NightSoftmax(dim=-1)
        self.prefix = f"unset.{self.__class__.__name__}"

        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.feat_size = feat_size
        self.scale = self.head_dim**-0.5
        self.pool = pool_type

        # self.fused_attn = use_fused_attn()
        self.fused_attn = False

        if pos_embed == "abs":
            assert feat_size is not None
            self.pos_embed = nn.Parameter(torch.zeros(feat_size, in_features))
        else:
            self.pos_embed = None

        self.latent_dim = latent_dim or embed_dim
        self.latent_len = latent_len
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, embed_dim))

        self.q = LinearDecomposed(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = LinearDecomposed(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = LinearDecomposed(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(drop)

        self.norm = (
            norm_layer(out_features) if norm_layer is not None else nn.Identity()
        )
        self.mlp = MLPDecomposed(embed_dim, int(embed_dim * mlp_ratio))

        self.init_weights()

    def init_weights(self):
        from timm.layers import trunc_normal_tf_

        if self.pos_embed is not None:
            trunc_normal_tf_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
        trunc_normal_tf_(self.latent, std=self.latent_dim**-0.5)

    def forward(self, x):
        import timm.models.decomposition as decomposition

        ###
        #: @duplicateCode/5002416408541872e2ad6934af56e2f6
        gradient_mode = decomposition.dynamic_obj.attention_gradient_mode
        print_diag(
            f"{self.prefix}: gradient_mode: {gradient_mode}", group="gradient_mode"
        )

        attn_softmax_gbrand = decomposition.dynamic_obj.attention_softmax_gradient_mode
        print_diag(
            f"{self.prefix}: attn_softmax_gbrand: {attn_softmax_gbrand}",
            group="gradient_mode",
        )

        attn_mul_gbrand = (
            decomposition.dynamic_obj.attention_elementwise_mul_gradient_mode
        )
        print_diag(
            f"{self.prefix}: attn_mul_gbrand: {attn_mul_gbrand}", group="gradient_mode"
        )
        ###

        B, N, C = x.shape

        if self.pos_embed is not None:
            # FIXME interpolate
            x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        q_latent = self.latent.expand(B, -1, -1)
        q = (
            self.q(q_latent)
            .reshape(B, self.latent_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        kv = (
            self.kv(x)
            .reshape(B, N, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        ###
        #: @duplicateCode/ee4f9527352d286022bfd9d0c4be373b
        q, k, v = fair_qkv(
            q,
            k,
            v,
            gradient_mode=gradient_mode,
        )
        ###

        if self.fused_attn:
            ##
            #: @duplicateCode/72d91ae046cf779ac085703a0b1fed46
            #: We also need to store the attention values for GenAtt so this is a failure anyways.
            assert (
                not attn_softmax_gbrand
            ), "attention_softmax_gradient_mode not yet supported on fused attention"
            assert (
                not attn_mul_gbrand
            ), "attention_elementwise_mul_gradient_mode not yet supported on fused attention"
            ##

            x = F.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            ###
            #: @duplicateCode/001a01d6062a862216d3cfb41b7909f4
            # attn = attn.softmax(dim=-1)
            attn = self.attn_softmax(
                attn,
                gradient_mode=attn_softmax_gbrand,
                competition_scale=decomposition.dynamic_obj.attention_softmax_competition_scale,
            )

            if attn_mul_gbrand == "D2":
                #: DU needs to be done below where we construct the output.
                ##
                attn = swap_backward(attn, attn / 4)
                #: attn is the result of multiplying keys and queries (bilinear multiplication?), so it needs another division by 2

                v = swap_backward(v, v / 2)

            if decomposition.dynamic_obj.raw_attention_grad_store_p:
                attn.requires_grad_(True)
                #: Our =gradient_mode= shenanigans might have prevented the gradient from being computed for =attn=, so we explicitly ask for it.
            ###

            x = attn @ v

            ##
            #: @duplicateCode/cbbd6065c4174ab14e17cdf2e7b6038a
            if attn_mul_gbrand == "DU":
                x = swap_backward(x, x / 3)
                #: The input feeds into the output from three multiplicative sources
            ##

        if False:
            #: We are not using these yet.
            ###
            #: @duplicateCode/6d15e4c5cbac0d72058c78d532c1e17c
            #: Value Storage
            delattr_force(self, "stored_value")
            if decomposition.dynamic_obj.value_store_p:
                self.stored_value = v.detach().cpu()  # .clone()

            delattr_force(self, "stored_value_grad")
            if decomposition.dynamic_obj.value_grad_store_p:

                def store_value_grad(grad):
                    # ic(torch_shape_get(grad))
                    self.stored_value_grad = grad.detach().cpu()

                tensor_register_hook(v, store_value_grad)
            ##
            #: MultiHeadAttention Storage
            delattr_force(self, "stored_mha")
            if decomposition.dynamic_obj.mha_store_p:
                self.stored_mha = x.detach().cpu()  # .clone()

            delattr_force(self, "stored_mha_grad")
            if decomposition.dynamic_obj.mha_grad_store_p:

                def store_mha_grad(grad):
                    # ic(torch_shape_get(grad))
                    self.stored_mha_grad = grad.detach().cpu()

                tensor_register_hook(x, store_mha_grad)
            ##
            delattr_force(self, "stored_rawattn_grad")
            if decomposition.dynamic_obj.raw_attention_grad_store_p:

                def store_rawattn_grad(grad):
                    # ic(torch_shape_get(grad))
                    self.stored_rawattn_grad = grad.detach().cpu()

                tensor_register_hook(attn, store_rawattn_grad)

            if decomposition.dynamic_obj.raw_attention_store_p:
                attn_cpu = attn.detach().cpu()  # .clone()
                self.stored_rawattn = attn_cpu
            else:
                delattr_force(self, "stored_rawattn")
            ###

        x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))

        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == "token":
            x = x[:, 0]

        elif self.pool == "avg":
            assert NotImplementedError(f"See @IDLink/ede5263a574877350366c9fbf569144c (when computing FullGrad) for details.")

            x = x.mean(1)

        return x


##
