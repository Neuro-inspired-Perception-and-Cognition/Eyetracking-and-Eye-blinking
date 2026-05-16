import torch
import torch.nn as nn

class SurrogateExponential(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, threshold):
        ctx.save_for_backward(input_tensor, threshold)
        return (input_tensor >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, threshold = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        grad_width = 1.0
        surrogate_grad = (1.0 / grad_width) * torch.exp(-torch.abs(input_tensor - threshold) / grad_width)
        
        return grad_input * surrogate_grad, None

class PLIF(nn.Module):
    def __init__(self, channels, init_tau, threshold=1.0):
        super(PLIF, self).__init__()
        self.threshold = threshold
        
        init_decay = 1.0 - (1.0 / init_tau)
        init_w = torch.log(torch.tensor(init_decay / (1.0 - init_decay)))
        self.w = nn.Parameter(torch.full((1, channels, 1, 1), init_w))
        
        self.spike_fn = SurrogateExponential.apply
        self.v_mem = None

    def forward(self, x):
        alpha = torch.sigmoid(self.w).clamp(0.01, 0.99)
        
        if self.v_mem is None or self.v_mem.shape != x.shape:
            self.v_mem = torch.zeros_like(x)

        self.v_mem = (self.v_mem * alpha) + x
        spike = self.spike_fn(self.v_mem, torch.tensor(self.threshold, device=x.device))
        self.v_mem = self.v_mem - (spike * self.threshold)
        
        return spike

    def reset_states(self):
        self.v_mem = None

class SNNConvLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, tau_mem, threshold=1.0, stride=1, padding=None):
        super(SNNConvLayer, self).__init__()
        
        if padding is None:
            self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            self.padding = padding
            
        self.conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size, 
                              stride=stride, padding=self.padding, bias=False)
                              
        nn.init.xavier_uniform_(self.conv.weight)
        self.bn = nn.BatchNorm2d(hidden_dim, affine=True)
        self.lif = PLIF(channels=hidden_dim, init_tau=tau_mem, threshold=threshold)

    def forward(self, input_spike):
        current = self.bn(self.conv(input_spike))
        return self.lif(current)

    def reset_states(self):
        self.lif.reset_states()

class SNNDeconvLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, tau_mem, threshold=1.0, stride=2, padding=1):
        super(SNNDeconvLayer, self).__init__()
        
        self.deconv = nn.ConvTranspose2d(input_dim, hidden_dim, kernel_size=kernel_size, 
                                         stride=stride, padding=padding, bias=False)
                              
        nn.init.xavier_uniform_(self.deconv.weight)
        self.bn = nn.BatchNorm2d(hidden_dim, affine=True)
        self.lif = PLIF(channels=hidden_dim, init_tau=tau_mem, threshold=threshold)

    def forward(self, input_spike):
        current = self.bn(self.deconv(input_spike))
        return self.lif(current)

    def reset_states(self):
        self.lif.reset_states()

class SlowLeakIntegrator(nn.Module):
    """Accumulates SNN spatial features with a bounded exponential decay."""
    def __init__(self, channels, init_tau=100.0):
        super(SlowLeakIntegrator, self).__init__()
        import math
        init_alpha = 1.0 - (1.0 / init_tau)
        init_w = math.log(init_alpha / (1.0 - init_alpha))
        self.w = nn.Parameter(torch.full((1, channels, 1, 1), float(init_w)))

    def forward(self, x, v_mem=None):
        if v_mem is None:
            v_mem = torch.zeros_like(x)
            
        alpha = torch.sigmoid(self.w).clamp(0.80, 0.9999)
        
        v_mem = v_mem * alpha + x
        return v_mem
        
    def get_tau(self):
        alpha = torch.sigmoid(self.w).clamp(0.80, 0.9999).detach()
        tau = 1.0 / (1.0 - alpha)
        return {
            'min': tau.min().item(),
            'max': tau.max().item(),
            'mean': tau.mean().item()
        }

class EyeTrackingSNN(nn.Module):
    def __init__(self, input_dim=2, tau_mem=40.0, decoder_tau=5.0):
        super(EyeTrackingSNN, self).__init__()
        
        # ENCODER: Reduced channel capacity to force generalization
        self.hidden_dims = [8, 16, 16, 32, 32] 
        self.encoder_layers = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        current_dim = input_dim
        for i, h_dim in enumerate(self.hidden_dims):
            self.encoder_layers.append(SNNConvLayer(current_dim, h_dim, (3, 3), tau_mem, threshold=1.0, stride=1))
            
            # Maintain the 8x8 spatial topology constraint
            if i < 3: 
                self.pools.append(SNNConvLayer(h_dim, h_dim, (2, 2), tau_mem, threshold=1.0, stride=2, padding=0))
            else:     
                self.pools.append(SNNConvLayer(h_dim, h_dim, (3, 3), tau_mem, threshold=1.0, stride=1, padding=1))
                
            current_dim = h_dim

        # INTEGRATOR: Operates on (32, 8, 8) map
        self.integrator = SlowLeakIntegrator(channels=32, init_tau=100.0)
        self.integrator_state = None
        
        # DECODER: Fully spiking upsampling layers
        self.decoder_dims = [16, 8, 1]  # Channel progression for decoder
        self.decoder_layers = nn.ModuleList()
        
        current_dim = 32  # Start from integrator output
        for h_dim in self.decoder_dims:
            self.decoder_layers.append(
                SNNDeconvLayer(current_dim, h_dim, kernel_size=4, tau_mem=decoder_tau, 
                              threshold=1.0, stride=2, padding=1)
            )
            current_dim = h_dim

    def reset_states(self):
        for layer in self.encoder_layers:
            layer.reset_states()
        for pool in self.pools:
            pool.reset_states()
        self.integrator_state = None
        for decoder_layer in self.decoder_layers:
            decoder_layer.reset_states()

    def get_layer_info(self):
        """Returns information about layer sizes for weighted spike rate calculation."""
        # Encoder layer outputs (before pooling)
        encoder_info = [
            {'channels': 8, 'h': 64, 'w': 64, 'name': 'enc_L0'},   # Layer 0 output
            {'channels': 16, 'h': 32, 'w': 32, 'name': 'enc_L1'},  # Layer 1 output (after pool)
            {'channels': 16, 'h': 16, 'w': 16, 'name': 'enc_L2'},  # Layer 2 output (after pool)
            {'channels': 32, 'h': 8, 'w': 8, 'name': 'enc_L3'},    # Layer 3 output (after pool)
            {'channels': 32, 'h': 8, 'w': 8, 'name': 'enc_L4'},    # Layer 4 output (no pool)
        ]
        
        # Decoder layer outputs
        decoder_info = [
            {'channels': 16, 'h': 16, 'w': 16, 'name': 'dec_L0'},  # Decoder layer 1
            {'channels': 8, 'h': 32, 'w': 32, 'name': 'dec_L1'},   # Decoder layer 2
            {'channels': 1, 'h': 64, 'w': 64, 'name': 'dec_L2'},   # Decoder layer 3
        ]
        
        # Add neuron counts
        for info in encoder_info + decoder_info:
            info['neurons'] = info['channels'] * info['h'] * info['w']
        
        return encoder_info, decoder_info

    def forward(self, x, reset=True):
        batch_size, seq_len, _, _, _ = x.shape
        if reset:
            self.reset_states()
        
        outputs = []
        
        differentiable_spike_sum = torch.tensor(0.0, device=x.device)
        # Include decoder layers in total count
        total_layers = len(self.encoder_layers) + len(self.pools) + len(self.decoder_layers)
        
        # Track spike activity over evaluation window (not just last frame)
        eval_start = 30  # Should match EVAL_START from training
        encoder_spike_accum = [0.0] * len(self.encoder_layers)
        decoder_spike_accum = [0.0] * len(self.decoder_layers)
        eval_frames = 0

        for t in range(seq_len):
            current_input = x[:, t, ...]
            
            # Encoder pass
            for i in range(len(self.encoder_layers)):
                spk_out = self.encoder_layers[i](current_input)
                current_input = self.pools[i](spk_out)
                differentiable_spike_sum = differentiable_spike_sum + spk_out.mean() + current_input.mean()
                
                # Accumulate spike rates during evaluation window
                if t >= eval_start:
                    encoder_spike_accum[i] += spk_out.detach().mean().item()
            
            # Integrator
            self.integrator_state = self.integrator(current_input, self.integrator_state)
            
            # Decoder pass (spiking) - same pattern as encoder
            dec_out = self.integrator_state
            for i, decoder_layer in enumerate(self.decoder_layers):
                dec_out = decoder_layer(dec_out)
                differentiable_spike_sum = differentiable_spike_sum + dec_out.mean()
                
                # Accumulate spike rates during evaluation window
                if t >= eval_start:
                    decoder_spike_accum[i] += dec_out.detach().mean().item()
            
            # Count evaluation frames
            if t >= eval_start:
                eval_frames += 1
            
            # Read membrane potential from final decoder layer for continuous output
            heatmap = self.decoder_layers[-1].lif.v_mem
            heatmap = torch.sigmoid(heatmap)
            outputs.append(heatmap.squeeze(1))
        
        # Compute average spike rates over evaluation window
        self.current_epoch_encoder_activity = [s / max(eval_frames, 1) for s in encoder_spike_accum]
        self.current_epoch_decoder_activity = [s / max(eval_frames, 1) for s in decoder_spike_accum]
        
        # Get layer sizes (constant, so compute once from any frame)
        self.current_epoch_encoder_sizes = [
            8 * 64 * 64,   # L0
            16 * 32 * 32,  # L1
            16 * 16 * 16,  # L2
            32 * 8 * 8,    # L3
            32 * 8 * 8,    # L4
        ]
        self.current_epoch_decoder_sizes = [
            16 * 16 * 16,  # D0
            8 * 32 * 32,   # D1
            1 * 64 * 64,   # D2
        ]
            
        batch_spike_loss = differentiable_spike_sum / (seq_len * total_layers)
        return torch.stack(outputs, dim=1), batch_spike_loss

    def get_taus(self):
        # Encoder taus
        encoder_tau_stats = []
        for layer in self.encoder_layers:
            w = layer.lif.w.detach()
            alpha = torch.sigmoid(w).clamp(0.01, 0.99)
            tau = 1.0 / (1.0 - alpha)

            encoder_tau_stats.append({
                'min': tau.min().item(),
                'max': tau.max().item(),
                'mean': tau.mean().item()
            })
        
        # Decoder taus
        decoder_tau_stats = []
        for decoder_layer in self.decoder_layers:
            w = decoder_layer.lif.w.detach()
            alpha = torch.sigmoid(w).clamp(0.01, 0.99)
            tau = 1.0 / (1.0 - alpha)
            
            decoder_tau_stats.append({
                'min': tau.min().item(),
                'max': tau.max().item(),
                'mean': tau.mean().item()
            })
            
        int_tau = self.integrator.get_tau()
        return encoder_tau_stats, decoder_tau_stats, int_tau