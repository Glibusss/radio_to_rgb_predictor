import torch
import torch.nn as nn
import torch.nn.functional as F


def make_gn(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    groups = min(num_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_se: bool = False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = make_gn(out_ch)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = make_gn(out_ch)

        self.use_proj = in_ch != out_ch
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                make_gn(out_ch),
            )
        else:
            self.proj = nn.Identity()

        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.proj(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)

        out = out + identity
        out = self.act2(out)
        return out


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_se: bool = False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ResidualBlock(in_ch, out_ch, use_se=use_se)

    def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x


class AttentionGate(nn.Module):
    """
    gating signal g: decoder feature
    skip feature x: encoder feature
    """
    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False),
            make_gn(inter_ch),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False),
            make_gn(inter_ch),
        )

        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.psi(g1 + x1)
        return x * psi


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_se: bool = False):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.attn = AttentionGate(
            g_ch=in_ch // 2,
            x_ch=skip_ch,
            inter_ch=max(min(skip_ch, in_ch // 2) // 2, 8),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, kernel_size=1, bias=False),
            make_gn(out_ch),
            nn.ReLU(inplace=True),
        )

        self.block = ResidualBlock(out_ch, out_ch, use_se=use_se)

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(
            x,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        skip = self.attn(x, skip)
        x = torch.cat([skip, x], dim=1)
        x = self.fuse(x)
        x = self.block(x)
        return x


class ASPPLite(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        branch_ch = channels // 4

        self.b1 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, kernel_size=1, bias=False),
            make_gn(branch_ch),
            nn.ReLU(inplace=True),
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            make_gn(branch_ch),
            nn.ReLU(inplace=True),
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, kernel_size=3, padding=4, dilation=4, bias=False),
            make_gn(branch_ch),
            nn.ReLU(inplace=True),
        )

        self.b4 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, kernel_size=3, padding=6, dilation=6, bias=False),
            make_gn(branch_ch),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(branch_ch * 4, channels, kernel_size=1, bias=False),
            make_gn(channels),
            nn.ReLU(inplace=True),
        )

        self.se = SEBlock(channels)

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)

        y = torch.cat([y1, y2, y3, y4], dim=1)
        y = self.project(y)
        y = self.se(y)
        return y


class UNetColorizer(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3, base: int = 64):
        super().__init__()

        self.inc = ResidualBlock(in_channels, base, use_se=False)

        self.down1 = Down(base, base * 2, use_se=False)
        self.down2 = Down(base * 2, base * 4, use_se=False)
        self.down3 = Down(base * 4, base * 8, use_se=True)
        self.down4 = Down(base * 8, base * 16, use_se=True)

        self.bottleneck = nn.Sequential(
            ResidualBlock(base * 16, base * 16, use_se=True),
            ASPPLite(base * 16),
        )

        self.up1 = Up(base * 16, base * 8, base * 8, use_se=True)
        self.up2 = Up(base * 8, base * 4, base * 4, use_se=True)
        self.up3 = Up(base * 4, base * 2, base * 2, use_se=False)
        self.up4 = Up(base * 2, base, base, use_se=False)

        self.head = nn.Sequential(
            nn.Conv2d(base, base, kernel_size=3, padding=1, bias=False),
            make_gn(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.bottleneck(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.head(x)
        return x