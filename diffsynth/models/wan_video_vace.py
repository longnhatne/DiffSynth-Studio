import torch
from .wan_video_dit import DiTBlock
from .utils import hash_state_dict_keys

class VaceWanAttentionBlock(DiTBlock):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = torch.nn.Linear(self.dim, self.dim)
        self.after_proj = torch.nn.Linear(self.dim, self.dim)

    def forward(self, c, x, context, t_mod, freqs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, context, t_mod, freqs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class VaceWanModel(torch.nn.Module):
    def __init__(
        self,
        vace_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        vace_in_dim=96,
        patch_size=(1, 2, 2),
        has_image_input=False,
        dim=1536,
        num_heads=12,
        ffn_dim=8960,
        eps=1e-6,
    ):
        super().__init__()
        self.vace_layers = vace_layers
        self.vace_in_dim = vace_in_dim
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # vace blocks
        self.vace_blocks = torch.nn.ModuleList([
            VaceWanAttentionBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i)
            for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = torch.nn.Conv3d(vace_in_dim, dim, kernel_size=patch_size, stride=patch_size)

    def forward(
        self, x, vace_context, context, t_mod, freqs,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block in self.vace_blocks:
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    c = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        c, x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                c = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    c, x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                c = block(c, x, context, t_mod, freqs)
        hints = torch.unbind(c)[:-1]
        return hints
    
    @staticmethod
    def state_dict_converter():
        return VaceWanModelDictConverter()
    

'''===========VACE Fuser==========='''
import torch
import torch.nn as nn
from .wan_video_dit import RMSNorm, AttentionModule
from .wan_video_dit import flash_attention

class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_ori = x
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(y))
        v = self.v(y)
        x = self.attn(q, k, v)
        return self.o(x) + x_ori

class VaceFuserBlock(torch.nn.Module):
    def __init__(self, num_vace_blocks: int, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.vace_hints_fuser = nn.ModuleList([CrossAttention(dim, num_heads, eps) for _ in range(num_vace_blocks)])

    def forward(self, vace_hints_list_a, vace_hints_list_b):
        fused_vace_hints = []
        for i in range(len(vace_hints_list_a)):
            x = self.vace_hints_fuser[i](vace_hints_list_a[i], vace_hints_list_b[i])
            fused_vace_hints.append(x)
        return fused_vace_hints

class VaceFuser(torch.nn.Module):
    def __init__(
        self,
        num_vace_blocks: int,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.fuser = VaceFuserBlock(num_vace_blocks, dim, num_heads, eps)

    def forward(
        self, vace_hints_list: list[torch.Tensor],
        use_gradient_checkpointing_offload: bool = False,
        use_gradient_checkpointing: bool = False,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        x = vace_hints_list[0] # global vace_context

        # Process local vace_context
        for i in range(1, len(vace_hints_list)):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.fuser),
                        x, vace_hints_list[i],
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.fuser),
                    x, vace_hints_list[i],
                    use_reentrant=False,
                )
            else:
                x = self.fuser(x, vace_hints_list[i])
        return x

class VaceWanModelDictConverter:
    def __init__(self):
        pass
    
    def from_civitai(self, state_dict):
        state_dict_ = {name: param for name, param in state_dict.items() if name.startswith("vace")}
        if hash_state_dict_keys(state_dict_) == '3b2726384e4f64837bdf216eea3f310d': # vace 14B
            config = {
                "vace_layers": (0, 5, 10, 15, 20, 25, 30, 35),
                "vace_in_dim": 96,
                "patch_size": (1, 2, 2),
                "has_image_input": False,
                "dim": 5120,
                "num_heads": 40,
                "ffn_dim": 13824,
                "eps": 1e-06,                
            }
        else:
            config = {}
        return state_dict_, config
