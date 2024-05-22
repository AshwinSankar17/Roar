import torch
import torch.nn.functional as F

from roar.core.classes import Loss, typecheck
from roar.core.neural_types.elements import (
    AudioSignal,
    LengthsType,
    LossType,
    SpectrogramType,
)
from roar.core.neural_types.neural_type import NeuralType


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.view_as_real(
        torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
    )
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


class SpectralConvergenceLoss(Loss):
    """Spectral convergence loss module."""

    @property
    def input_types(self):
        return {
            "x_mag": NeuralType(("B", "T", "D"), SpectrogramType()),
            "y_mag": NeuralType(("B", "T", "D"), SpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, *, x_mag, y_mag):
        """Calculate forward propagation. It is assumed that x_mag and y_mag were padded to fit the maximum batch
        sequence length with silence, hence it is assumed that the norm of these extra padded values are 0. Therefore,
        input_lengths is not a argument unlike in LogSTFTMagnitudeLoss.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        # Mean across time and freq_bins first
        loss = torch.norm(y_mag - x_mag, p="fro", dim=(1, 2)) / torch.norm(
            y_mag, p="fro", dim=(1, 2)
        )
        # Mean across batches
        loss = torch.mean(loss)
        return loss


class LogSTFTMagnitudeLoss(Loss):
    """Log STFT magnitude loss module."""

    @property
    def input_types(self):
        return {
            "x_mag": NeuralType(("B", "T", "D"), SpectrogramType()),
            "y_mag": NeuralType(("B", "T", "D"), SpectrogramType()),
            "input_lengths": NeuralType(("B"), LengthsType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, *, x_mag, y_mag, input_lengths=None):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
            input_lengths (Tensor): Length of groundtruth sample in samples (B).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        if input_lengths is None:
            # During training, we used fixed sequence length, so just average across all dimensions
            return F.l1_loss(torch.log(y_mag), torch.log(x_mag))
        loss = F.l1_loss(torch.log(y_mag), torch.log(x_mag), reduction="none")
        # First sum and average across time and freq bins
        loss = loss / loss.shape[2]
        loss = torch.sum(loss, dim=[1, 2])
        loss = loss / input_lengths
        # Last average across batch
        return torch.sum(loss) / loss.shape[0]


class STFTLoss(Loss):
    """STFT loss module."""

    def __init__(
        self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    @property
    def input_types(self):
        return {
            "x": NeuralType(("B", "T"), AudioSignal()),
            "y": NeuralType(("B", "T"), AudioSignal()),
            "input_lengths": NeuralType(("B"), LengthsType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "sc_loss": NeuralType(elements_type=LossType()),
            "mag_loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, *, x, y, input_lengths=None):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
            input_lengths (Tensor): Length of groundtruth sample in samples (B).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        if self.window.device != x.device:
            self.window = self.window.to(x.device)
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergence_loss(x_mag=x_mag, y_mag=y_mag)
        if input_lengths is not None:
            input_lengths = torch.floor(input_lengths / float(self.shift_size)) + 1
            assert (
                max(input_lengths) == x_mag.shape[1]
            ), f"{max(input_lengths)} != {x_mag.shape[1]}"
        mag_loss = self.log_stft_magnitude_loss(
            x_mag=x_mag, y_mag=y_mag, input_lengths=input_lengths
        )

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(Loss):
    """Multi resolution STFT loss module."""

    def __init__(self, resolutions, window="hann_window"):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        # assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in resolutions:
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    @property
    def input_types(self):
        return {
            "x": NeuralType(("B", "T"), AudioSignal()),
            "y": NeuralType(("B", "T"), AudioSignal()),
            "input_lengths": NeuralType(("B"), LengthsType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "sc_loss": [NeuralType(elements_type=LossType())],
            "mag_loss": [NeuralType(elements_type=LossType())],
        }

    @typecheck()
    def forward(self, *, x, y, input_lengths=None):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
            input_lengths (Tensor): Length of groundtruth sample in samples (B).
        Returns:
            List[Tensor]: Multi resolution spectral convergence loss value.
            List[Tensor]: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = [0.0] * len(self.stft_losses)
        mag_loss = [0.0] * len(self.stft_losses)
        for i, f in enumerate(self.stft_losses):
            sc_l, mag_l = f(x=x, y=y, input_lengths=input_lengths)
            sc_loss[i] = sc_l
            mag_loss[i] = mag_l

        return sc_loss, mag_loss


# class MultiResolutionSTFTLoss(Loss):
#     """Multi resolution STFT loss module."""

#     def __init__(self, fft_sizes, hop_sizes, win_lengths, window="hann_window"):
#         """Initialize Multi resolution STFT loss module.
#         Args:
#             fft_sizes (list): List of FFT sizes.
#             hop_sizes (list): List of hop sizes.
#             win_lengths (list): List of window lengths.
#             window (str): Window function type.
#         """
#         super(MultiResolutionSTFTLoss, self).__init__()
#         assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
#         self.stft_losses = torch.nn.ModuleList()
#         for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
#             self.stft_losses += [STFTLoss(fs, ss, wl, window)]

#     @property
#     def input_types(self):
#         return {
#             "x": NeuralType(("B", "T"), AudioSignal()),
#             "y": NeuralType(("B", "T"), AudioSignal()),
#             "input_lengths": NeuralType(("B"), LengthsType(), optional=True),
#         }

#     @property
#     def output_types(self):
#         return {
#             "sc_loss": [NeuralType(elements_type=LossType())],
#             "mag_loss": [NeuralType(elements_type=LossType())],
#         }

#     @typecheck()
#     def forward(self, *, x, y, input_lengths=None):
#         """Calculate forward propagation.
#         Args:
#             x (Tensor): Predicted signal (B, T).
#             y (Tensor): Groundtruth signal (B, T).
#             input_lengths (Tensor): Length of groundtruth sample in samples (B).
#         Returns:
#             List[Tensor]: Multi resolution spectral convergence loss value.
#             List[Tensor]: Multi resolution log STFT magnitude loss value.
#         """
#         sc_loss = [0.0] * len(self.stft_losses)
#         mag_loss = [0.0] * len(self.stft_losses)
#         for i, f in enumerate(self.stft_losses):
#             sc_l, mag_l = f(x=x, y=y, input_lengths=input_lengths)
#             sc_loss[i] = sc_l
#             mag_loss[i] = mag_l

#         return sc_loss, mag_loss
