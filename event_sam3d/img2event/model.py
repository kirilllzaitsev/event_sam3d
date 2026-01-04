import functools
from collections import defaultdict

import torch
from torch import nn


def fetch_module_by_name(net, name):
    names = name.split(".")
    module = net
    for n in names:
        module = getattr(module, n)
    return module


class TeacherStudent(nn.Module):
    def __init__(
        self,
        s,
        t,
        t_blocks_name="backbone.blocks",
        block_idxs=None,
        use_attn_hook=False,
    ):
        # s/t are Inference pipelines from sam3d
        from event_sam3d.img2event.model_utils import get_condition_embedder

        super().__init__()
        self.s = s
        self.t = t
        self.t_blocks_name = t_blocks_name

        self.use_attn_hook = use_attn_hook

        if block_idxs is None:
            block_idxs = range(
                len(
                    fetch_module_by_name(
                        get_condition_embedder(self.t, use_event=False), t_blocks_name
                    )
                )
            )
        self.block_idxs = block_idxs

        # set hooks for all outputs
        self.s_embeds = defaultdict(list)
        self.t_embeds = defaultdict(list)

        self.s_hooks = set_hooks(
            self.s, self.s_embeds, use_event=True, block_idxs=self.block_idxs
        )
        self.t_hooks = set_hooks(
            self.t, self.t_embeds, use_event=False, block_idxs=self.block_idxs
        )
        layer = self.s.condition_embedders["ss_condition_embedder"]
        hook = layer.register_forward_hook(
            get_hook(
                "t_final_rgb_tokens",
                embeds=self.s_embeds,
                fn=functools.partial(dict_fn, k="rgb_image_tokens"),
            )
        )
        self.s_hooks.append(hook)
        layer = self.t.condition_embedders["ss_condition_embedder"]
        hook = layer.register_forward_hook(
            get_hook(
                "t_final_rgb_tokens",
                embeds=self.t_embeds,
                fn=functools.partial(dict_fn, k="rgb_image_tokens"),
            )
        )
        self.t_hooks.append(hook)

    def forward(self, s_kwargs, t_kwargs=None, **kwargs):
        self.s_embeds.clear()
        self.t_embeds.clear()
        s_pred = self.s(**s_kwargs, **kwargs)
        res = {"s_pred": s_pred, "s_feats": self.s_embeds}
        if t_kwargs is not None:
            with torch.no_grad():
                t_pred = self.t(**t_kwargs, **kwargs)
            res.update({"t_pred": t_pred, "t_feats": self.t_embeds})
        return res


def dict_fn(output, k, input=None):
    return output[k]


def get_hook(name, embeds, fn=None):
    def hook(module, input, output):
        if fn is not None:
            output = fn(output)
        embeds[name].append(output)

    return hook


def get_pre_hook(name, embeds, fn=None, kwarg_name=None):
    def hook(module, input, kwargs):
        if kwarg_name is None:
            embeds[name].append(input)
        else:
            embeds[name].append(kwargs[kwarg_name][0])

    return hook


def set_hooks(net, embeds, use_event, block_idxs, t_blocks_name="backbone.blocks"):
    from event_sam3d.img2event.model_utils import get_condition_embedder

    condition_embedder = get_condition_embedder(net, use_event=use_event)
    hooks = []

    for i in block_idxs:
        layer = fetch_module_by_name(condition_embedder, f"{t_blocks_name}.{i}")

        hook = layer.register_forward_hook(get_hook(f"block_{i}", embeds=embeds))
        hooks.append(hook)
    return hooks
