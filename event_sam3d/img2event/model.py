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
        super().__init__()
        self.s = s
        self.t = t
        self.t_blocks_name = t_blocks_name

        self.use_attn_hook = use_attn_hook

        if block_idxs is None:
            block_idxs = range(len(fetch_module_by_name(self.t, t_blocks_name)))
        self.block_idxs = block_idxs

        # set hooks for all outputs
        self.s_embeds = defaultdict(list)
        self.t_embeds = defaultdict(list)

        def set_hooks(net, embeds):
            condition_embedder = [
                x
                for x in net._pipeline.condition_embedders[
                    "ss_condition_embedder"
                ].embedder_list
                if all("event" in xx[0] for xx in x[1])
            ]
            assert (
                len(condition_embedder) == 1 and len(condition_embedder[0]) == 2
            ), len(condition_embedder)
            # same encoder for full/cropped imgs
            condition_embedder = condition_embedder[0][0]
            hooks = []

            for i in self.block_idxs:
                layer = fetch_module_by_name(condition_embedder, f"{t_blocks_name}.{i}")

                def get_hook(name):
                    def hook(module, input, output):
                        embeds[name].append(output)

                    return hook

                hook = layer.register_forward_hook(get_hook(f"block_{i}"))
                hooks.append(hook)
            return hooks

        self.s_hooks = set_hooks(self.s, self.s_embeds)
        self.t_hooks = set_hooks(self.t, self.t_embeds)

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
