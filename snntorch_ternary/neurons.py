import torch
from torch import nn

from snntorch._neurons.neurons import LIF


class TernaryLeaky(LIF):
    """
    Ternary Leaky Integrate-and-Fire neuron.

    - Inherits all buffers/parameters from LIF (beta, threshold, spike_grad, etc.)
    - Output spikes are in {-1, 0, +1}:
        +1 if mem >  +threshold
        -1 if mem <  -neg_threshold
         0 otherwise

    Reset behaviour mirrors Leaky:
        - 'subtract': pos spikes subtract +threshold, neg spikes add neg_threshold
        - 'zero':     pos spikes zero positive state, neg spikes zero negative state
        - 'none':     no reset
    """

    def __init__(
        self,
        beta,
        threshold=1.0,
        neg_threshold=None,
        symmetric_threshold=True,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        reset_delay=True,
    ):
        # Call LIF parent to set up beta/threshold, surrogate, buffers, etc.
        super().__init__(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            surrogate_disable=surrogate_disable,
            init_hidden=init_hidden,
            inhibition=inhibition,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            reset_mechanism=reset_mechanism,
            state_quant=state_quant,
            output=output,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
        )

        self._init_mem()

        # Set ternary thresholds
        if neg_threshold is None and not symmetric_threshold:
            # default negative threshold: 0 (no negative spikes)
            neg_threshold = 0.0

        if neg_threshold is not None and symmetric_threshold:
            raise ValueError(
                "If `symmetric_threshold` is True, `neg_threshold` must be None."
            )


        if neg_threshold is None and symmetric_threshold:
            # symmetric thresholds: [-threshold, +threshold]
            neg_threshold = threshold

        if not isinstance(neg_threshold, torch.Tensor):
            neg_threshold = torch.as_tensor(neg_threshold)

        # Separate buffer for negative threshold magnitude
        # We store magnitude; sign is handled in the logic.
        self.register_buffer("neg_threshold", torch.abs(neg_threshold))

        if self.reset_mechanism_val == 0:  # reset by subtraction
            self.state_function = self._base_sub
        elif self.reset_mechanism_val == 1:  # reset to zero
            self.state_function = self._base_zero
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            self.state_function = self._base_int


        # Which update function to use for reset
        # (We don't use Leaky.state_function directly, but keep reset_mechanism_val)
        self.reset_delay = reset_delay

    def _init_mem(self):
        mem = torch.zeros(0)
        self.register_buffer("mem", mem, False)

    def _base_state_function(self, input_):
        # Same as Leaky._base_state_function, but we re-expose it here for clarity
        return self.beta.clamp(0, 1) * self.mem + input_
    
    def _base_state_function(self, input_):
        base_fn = self.beta.clamp(0, 1) * self.mem + input_
        return base_fn

    def _base_sub(self, input_):
        # changed for ternary
        pos_res = torch.clamp(self.reset, min=0)
        neg_res = torch.clamp(-self.reset, min=0)
        reset_sig = -pos_res * self.threshold + neg_res * self.neg_threshold
        return self._base_state_function(input_) + reset_sig

    def _base_zero(self, input_): # ACK TODO
        self.mem = (1 - self.reset) * self.mem
        return self._base_state_function(input_)

    def _base_int(self, input_):# ACK TODO
        return self._base_state_function(input_)
    
    def mem_reset(self, mem, p_threshold, n_threshold):
        """Generates detached reset signal if mem > threshold.
        Returns reset."""
        pos_shift = mem - p_threshold
        neg_shift = -mem - n_threshold
        reset  = self.spike_grad(pos_shift).clone().detach()
        reset -= self.spike_grad(neg_shift).clone().detach()

        return reset
    
    def ternary_fire(self, mem):
        # Compute positive and negative spike candidates
        # pos: mem > +threshold, neg: mem < -neg_threshold
        # Use surrogate gradient on both sides.

        pos_shift = self.mem - self.threshold 
        neg_shift = -self.mem - self.neg_threshold

        spk_pos = self.spike_grad(pos_shift)          # ~Heaviside(mem - thr)
        spk_neg = self.spike_grad(neg_shift)          # ~Heaviside(-mem - neg_thr)

        # Scale with graded_spikes_factor like base class
        spk_pos = spk_pos * self.graded_spikes_factor
        spk_neg = spk_neg * self.graded_spikes_factor

        # Final ternary spike: +1 for pos, -1 for neg. Only one can be nonzero at a time.
        spk = spk_pos - spk_neg

        return spk
    
    def forward(self, input_, mem=None):
        # Optional external mem override (same semantics as Leaky)
        if mem is not None:
            self.mem = mem

        if self.init_hidden and mem is not None:
            raise TypeError(
                "`mem` should not be passed as an argument while `init_hidden=True`"
            )

        # Initialize mem shape if first call
        if self.mem.shape != input_.shape:
            self.mem = torch.zeros_like(input_, device=self.mem.device)

        # Update membrane (pure LIF integration, no reset yet)
        self.reset = self.mem_reset( self.mem, self.threshold, self.neg_threshold)
        self.mem = self.state_function(input_)

        # Optional state quantization
        if self.state_quant:
            self.mem = self.state_quant(self.mem)

        spk = self.ternary_fire(self.mem)

        # Reset behaviour (optional delay)
        if not self.reset_delay:
            # Detach resets from graph (like mem_reset does)
            with torch.no_grad():
                spk_pos = torch.clamp(spk, min=0)
                spk_neg = torch.clamp(-spk, min=0)
                reset_pos = spk_pos / self.graded_spikes_factor
                reset_neg = -spk_neg / self.graded_spikes_factor

            # reset_mechanism_val: 0=subtract, 1=zero, 2=none (see SpikingNeuron.reset_dict)
            if self.reset_mechanism_val == 0:  # subtract / add
                # Positive spikes pull mem down by thr
                self.mem = self.mem - reset_pos * self.threshold
                # Negative spikes pull mem up by neg_thr
                self.mem = self.mem + reset_neg * self.neg_threshold

            elif self.reset_mechanism_val == 1:  # zero
                # Zero out positive part on positive spikes
                self.mem = self.mem * (1 - reset_pos)
                # Zero out negative part on negative spikes
                self.mem = self.mem * (1 - reset_neg)

            # If reset_mechanism_val == 2 -> "none": do nothing

        # Output semantics consistent with Leaky
        if self.output:
            return spk, self.mem
        elif self.init_hidden:
            return spk
        else:
            return spk, self.mem

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], TernaryLeaky):
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables.
        Assumes hidden states have a batch dimension already."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], TernaryLeaky):
                cls.instances[layer].mem = torch.zeros_like(
                    cls.instances[layer].mem,
                    device=cls.instances[layer].mem.device,
                )
