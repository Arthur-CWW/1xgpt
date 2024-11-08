# %%
import sys
import torch
import torch.nn as nn

from magvit2.config import VQConfig


# %%
def swish(x: torch.Tensor) -> torch.Tensor:
    # swish
    return x * torch.sigmoid(x)


if __name__ == "__main__":
    x = torch.randn(size=(2, 3, 128, 128))
    y = swish(x)
    # Plot swish activation
    import matplotlib.pyplot as plt
    import numpy as np

    x_plot = np.linspace(-6, 6, 100)

    sigmoid = 1 / (1 + np.exp(-x_plot))
    y_plot = swish(torch.tensor(x_plot))
    y_plot = y_plot.numpy()

    plt.figure(figsize=(8, 6))
    plt.plot(x_plot, y_plot, label="swish(x)")
    plt.plot(x_plot, x_plot, "--", label="x", alpha=0.5)  # Plot identity for reference
    plt.plot(x_plot, sigmoid, "--", label="sigmoid(x)", alpha=0.5)  # Plot sigmoid for comparison

    plt.grid(True)
    plt.legend()
    plt.title("Swish Activation Function")
    plt.xlabel("x")
    plt.ylabel("swish(x)")
    # plt.savefig("swish.png")
    plt.show()
    # plt.close()
    print(y.shape)

# %%


class ResBlock(nn.Module):
    def __init__(self, in_filters: int, out_filters: int, use_conv_shortcut: bool = False) -> None:
        super().__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = nn.GroupNorm(32, in_filters, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, out_filters, eps=1e-6)

        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)

        if in_filters != out_filters:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_filters, out_filters, kernel_size=(1, 1), padding=0, bias=False
                )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = x

        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return x + residual


if __name__ == "__main__":
    x = torch.randn(size=(2, 3, 128, 128))  # [B, C, H, W]
    # Demonstrate GroupNorm effects
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    # Create sample feature maps
    x = 500 * torch.randn(2, 32, 64, 64) + 100  # batch_size=2, channels=32, height=64, width=64

    # Apply GroupNorm with different numbers of groups
    gn2 = nn.GroupNorm(num_groups=2, num_channels=32, eps=1e-6)
    # gn4 = nn.GroupNorm(num_groups=4, num_channels=32, eps=1e-6)
    # gn8 = nn.GroupNorm(num_groups=8, num_channels=32, eps=1e-6)

    out2: torch.Tensor = gn2(x)
    # out4: torch.Tensor = gn4(x)
    # out8: torch.Tensor = gn8(x)

    # # Plot original and normalized feature maps
    # plt.figure(figsize=(15, 5))

    # plt.subplot(121)
    # # plt.imshow(x[0, :3].permute(1, 2, 0).detach().numpy())
    # plt.imshow(x[0, :3].permute(1, 2, 0).detach().numpy())
    # plt.subplot(122)
    ex = out2[0, :3].detach()
    ex2 = out2[0, :3].transpose(0, 2).detach()
    ex3 = out2[0, :3].permute(1, 2, 0).detach()
    import einops

    ex4 = einops.rearrange(out2, "b c h w ->b h w c")
    sl = ex4[..., 0, 0]
    slu = einops.reduce(sl, "b h w -> h w", "mean")
    print(ex.shape, ex2.shape, ex3.shape)
    assert sl
    print(ex4.shape, sl.shape)

    raise
    plt.imshow(ex, cmap="gray")
    plt.title("GroupNorm (2 groups)")
    plt.colorbar()
    for _ in range(2):
        print(f"channel {_}:", x[_, 0].mean().item(), x[_, 0].std().item())

    print(x.mean().item(), x.std().item())
    print(x.shape, x.shape)
    print(x.mean(dim=0).shape, x.std(dim=1).shape)
    print(x.mean(dim=(1, 2)).shape, x.std(dim=1).shape)
    print(out2.mean(dim=1).shape, out2.std(dim=1).shape)

    # for i in range(2):
    #     plt.subplot(142)
    #     plt.imshow(out2[i, 0].detach().numpy())
    #     plt.title("GroupNorm (2 groups)")
    #     plt.colorbar()

    # plt.subplot(143)
    # plt.imshow(out4[0, 0].detach().numpy())
    # plt.title("GroupNorm (4 groups)")
    # plt.colorbar()

    # plt.subplot(144)
    # plt.imshow(out8[0, 0].detach().numpy())
    # plt.title("GroupNorm (8 groups)")
    # plt.colorbar()

    # plt.tight_layout()
    # plt.show()

    # # Print statistics to show normalization effect
    # print("Original stats:")
    # print(f"Mean: {x.mean():.3f}, Std: {x.std():.3f}")
    # print("\nAfter GroupNorm (2 groups):")
    # print(f"Mean: {out2.mean():.3f}, Std: {out2.std():.3f}")
    # print("\nAfter GroupNorm (4 groups):")
    # print(f"Mean: {out4.mean():.3f}, Std: {out4.std():.3f}")
    # print("\nAfter GroupNorm (8 groups):")
    # print(f"Mean: {out8.mean():.3f}, Std: {out8.std():.3f}")
# resblock = ResBlock(in_filters=3, out_filters=3)
# y = resblock(x)
# print(y.shape)


class Encoder(nn.Module):
    def __init__(
        # self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4),
        self,
        config: VQConfig,
    ):
        super().__init__()

        self.in_channels = config.in_channels
        self.z_channels = config.z_channels

        self.num_res_blocks = config.num_res_blocks
        self.num_blocks = len(config.ch_mult)

        self.conv_in = nn.Conv2d(
            config.in_channels,
            config.base_channels,
            kernel_size=(3, 3),
            padding=1,
            bias=False,
        )

        ## construct the model
        self.down = nn.ModuleList()

        in_ch_mult = (1,) + tuple(config.ch_mult)
        for i_level in range(self.num_blocks):
            block = nn.ModuleList()
            block_in = config.base_channels * in_ch_mult[i_level]  # [1, 1, 2, 2, 4]
            block_out = config.base_channels * config.ch_mult[i_level]  # [1, 2, 2, 4]
            for _ in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out

            down = nn.Module()
            down.block = block
            if i_level < self.num_blocks - 1:
                down.downsample = nn.Conv2d(
                    block_out, block_out, kernel_size=(3, 3), stride=(2, 2), padding=1
                )

            self.down.append(down)

        ### mid
        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in))

        ### end
        self.norm_out = nn.GroupNorm(32, block_out, eps=1e-6)
        self.conv_out = nn.Conv2d(block_out, config.z_channels, kernel_size=(1, 1))

    def forward(self, x):
        ## down
        x = self.conv_in(x)
        for i_level in range(self.num_blocks):
            for i_block in range(self.num_res_blocks):
                x = self.down[i_level].block[i_block](x)

            if i_level < self.num_blocks - 1:
                x = self.down[i_level].downsample(x)

        ## mid
        for res in range(self.num_res_blocks):
            x = self.mid_block[res](x)

        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)

        return x


class Decoder(nn.Module):
    # def __init__(self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4)) -> None:
    def __init__(self, config: VQConfig) -> None:
        super().__init__()

        self.base_channels = config.base_channels
        self.num_blocks = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks

        block_in = self.base_channels * config.ch_mult[self.num_blocks - 1]

        self.conv_in = nn.Conv2d(
            config.z_channels, block_in, kernel_size=(3, 3), padding=1, bias=True
        )

        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in))

        self.up = nn.ModuleList()

        for i_level in reversed(range(self.num_blocks)):
            block = nn.ModuleList()
            block_out = self.base_channels * config.ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out

            up = nn.Module()
            up.block = block
            if i_level > 0:
                up.upsample = Upsampler(block_in)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)

        self.conv_out = nn.Conv2d(block_in, config.out_channels, kernel_size=(3, 3), padding=1)

    def forward(self, z):
        z = self.conv_in(z)

        ## mid
        for res in range(self.num_res_blocks):
            z = self.mid_block[res](z)

        ## upsample
        for i_level in reversed(range(self.num_blocks)):
            for i_block in range(self.num_res_blocks):
                z = self.up[i_level].block[i_block](z)

            if i_level > 0:
                z = self.up[i_level].upsample(z)

        z = self.norm_out(z)
        z = swish(z)
        z = self.conv_out(z)

        return z


def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Depth-to-Space DCR mode (depth-column-row) core implementation.

    Args:
        x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
        block_size (int): block side size
    """
    # check inputs
    if x.dim() < 3:
        raise ValueError("Expecting a channels-first (*CHW) tensor of at least 3 dimensions")
    c, h, w = x.shape[-3:]

    s = block_size**2
    if c % s != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels"
        )

    outer_dims = x.shape[:-3]

    # splitting two additional dimensions from the channel dimension
    x = x.view(-1, block_size, block_size, c // s, h, w)

    # putting the two new dimensions along H and W
    x = x.permute(0, 3, 4, 1, 5, 2)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(*outer_dims, c // s, h * block_size, w * block_size)

    return x


class Upsampler(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim * 4
        self.conv1 = nn.Conv2d(dim, dim_out, (3, 3), padding=1)
        self.depth2space = depth_to_space

    def forward(self, x):
        """
        input_image: [B C H W]
        """
        out = self.conv1(x)
        out = self.depth2space(out, block_size=2)
        return out


# if __name__ == "__main__":
#     x = torch.randn(size = (2, 3, 128, 128))
#     encoder = Encoder(ch=128, in_channels=3, num_res_blocks=2, z_channels=18, out_ch=3, resolution=128)
#     decoder = Decoder(out_ch=3, z_channels=18, num_res_blocks=2, ch=128, in_channels=3, resolution=128)
#     z = encoder(x)
#     out = decoder(z)

# %%
