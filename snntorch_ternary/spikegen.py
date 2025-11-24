import torch
from snntorch import spikegen


def rate_ternary(
    data,
    num_steps=False,
    gain=1,
    offset=0,
    first_spike_time=0,
    time_var_input=False,
):
    """
    Ternary rate encoding.

    - Input data may be in [-1, 1].
    - |data| is interpreted as a Bernoulli rate.
    - Sign(data) controls polarity of the spike:
        +1 -> +1 spikes
        -1 -> -1 spikes
         0 -> no spikes

    Shape & arguments mirror snntorch.spikegen.rate.
    """

    # Capture sign (broadcasts across time)
    sign = torch.sign(data)  # -1, 0, +1

    # Use magnitude for actual rate
    mag = torch.abs(data)

    if time_var_input:
        # time dimension already present; spikegen.rate expects time_var_input flag
        spk = spikegen.rate(
            mag,
            num_steps=num_steps,
            gain=gain,
            offset=offset,
            first_spike_time=first_spike_time,
            time_var_input=True,
        )
        # spk, mag, sign all same shape, so simple elementwise product
        return spk * sign
    else:
        # Static input: spikegen.rate will prepend time dimension
        spk = spikegen.rate(
            mag,
            num_steps=num_steps,
            gain=gain,
            offset=offset,
            first_spike_time=first_spike_time,
            time_var_input=False,
        )
        # spk shape: [T, ...], sign shape: [...]
        # Broadcast sign over time:
        #   spk * sign (with extra leading time dimension of size 1)
        while sign.dim() < spk.dim():
            sign = sign.unsqueeze(0)
        return spk * sign