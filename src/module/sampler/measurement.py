from torch import nn

from src.dataset.fastmri.data.transforms import chan_complex_to_last_dim, complex_to_chan_dim
from src.dataset.fastmri import utils


class MRISystemMatrix(nn.Module):
    def __init__(self, subsample_mask, sensmaps):
        super().__init__()
        self.subsample_mask = subsample_mask
        self.sensmaps = sensmaps

    def forward(self, x):
        return self.AT(self.A(x))

    def A(self, x):
        """
        Args:
            x: shape [batch_size, 2, height, width]

        Returns:
            x: shape [batch_size, c, height, width, 2]
        """
        x = chan_complex_to_last_dim(x) # [batch_size, 2, height, width] -> [batch_size, 1, height, width, 2]
        # x = x * self.max_value
        x = utils.complex_mul(self.sensmaps, x) # sens expand
        x = utils.fft2c(x) # image to k-space
        x = x * self.subsample_mask
        return x

    def AT(self, x):
        """
        Args:
            x: shape [batch_size, c, height, width, 2]

        Returns:
            x: shape [batch_size, 2, height, width]

        """
        x = x * self.subsample_mask
        x = utils.ifft2c(x)   # k-space to image
        x = utils.complex_mul(utils.complex_conj(self.sensmaps), x).sum(dim=-4, keepdim=True) # sens reduce
        # x = x / self.max_value
        x = complex_to_chan_dim(x) # [batch_size, 1, height, width, 2] -> [batch_size, 2, height, width]
        return x