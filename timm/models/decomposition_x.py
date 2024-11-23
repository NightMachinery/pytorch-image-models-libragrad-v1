#: @DEPRECATED This whole file has been obsoleted by the package =decompv=.
##
import traceback
import sys
from functools import partial
import datasets
from icecream import ic, colorize as ic_colorize

from pynight.common_torch import (
    no_grad_maybe,
    torch_shape_get,
    model_device_get,
    img_tensor_show,
    hash_tensor,
    torch_memory_tensor,
)

from pynight.common_timm import (
    model_name_get,
)

from timm.models.decomposition import (
    named_experiment2,
    image_from_url,
    model_device_get,
    DecompositionConfig,
    attributions_n_get,
    mean_normalize_last_dim,
)


##
def model_decompose(
    model,
    url,
    mode='CLS_n',
):
    #: @retired
    #: We should move all the attribution-computing logic out of entrypoint functions, and have separate plotters for them. Keeping the comparative 'second_attrs' might still be good though.
    ##

    model_name = model_name_get(model)

    try:
        image_all = image_from_url(model=model, url=url)
    except:
        print(traceback.format_exc(), file=sys.stderr)
        return

    device = model_device_get(model)

    config_1 = DecompositionConfig(
        device=device,
        attributions_aggregation_strategy='vector',
    )

    attributions_dict = dict()
    if mode == 'CLS_n':
        decomposition_config = config_1
        inputs = model.forward_features_decomposed(
            image_all.image_dv,
            decomposition_config=decomposition_config,
        )
        # ic(torch_shape_get(vars(inputs)))

        attributions_v = inputs[f'attributions_v']

        inputs_d = vars(inputs)
        for k, v in inputs_d.items():
            if k in [
                'attributions_v',
            ] or k.startswith('blocks__'):
                attributions_dict[k] = v

        attributions_n = attributions_n_get(
            attributions_v,
            0,
            zero_out_error=True,
        ).squeeze()

        # ic(attributions_n.shape,
        #    # torch.sum(attributions_n),
        #    # attributions_n,
        #    )

        attributions_n = mean_normalize_last_dim(attributions_n)
        # ic(attributions_n.shape, torch.sum(attributions_n), attributions_n,)

        attributions_dict[
            named_experiment2(title='CLS_n_normalized', config=decomposition_config).name
        ] = attributions_n

    return attributions_dict


##
