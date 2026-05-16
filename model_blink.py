import torch
import torch.nn as nn
import torch.nn.functional as F
import sinabs.layers as sl

class SurrogateExponential(torch.autograd.Function):
    grad_width = 2.0  # Initial width
    
    @staticmethod
    def forward(ctx, input_tensor, threshold):
        ctx.save_for_backward(input_tensor, threshold)
        return (input_tensor >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, threshold = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        gw = SurrogateExponential.grad_width
        gamma = 1.0  # Locked peak amplitude
        
        surrogate_grad = gamma * torch.exp(-torch.abs(input_tensor - threshold) / gw)
        
        return grad_input * surrogate_grad, None

class PLIF(nn.Module):
    def __init__(self, channels, init_tau, threshold=1.0):
        super(PLIF, self).__init__()
        self.threshold = threshold
        
        tau_spread = init_tau * 0.5
        min_tau = max(1.5, init_tau - tau_spread)
        max_tau = init_tau + tau_spread
        
        taus = torch.empty(1, channels, 1, 1).uniform_(min_tau, max_tau)
        init_decays = 1.0 - (1.0 / taus)
        
        init_w = torch.log(init_decays / (1.0 - init_decays))
        self.w = nn.Parameter(init_w)
        
        self.spike_fn = SurrogateExponential.apply
        self.v_mem = None
    
    def forward(self, x):
        alpha = torch.sigmoid(self.w).clamp(0.01, 0.99)
        
        if x.dim() == 2:
            alpha = alpha.view(1, -1)
            
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
        self.bn = nn.GroupNorm(num_groups=1, num_channels=hidden_dim, affine=True, eps=0.01)
        self.lif = PLIF(channels=hidden_dim, init_tau=tau_mem, threshold=threshold)

    def forward(self, input_spike):
        current = self.bn(self.conv(input_spike))
        return self.lif(current)

    def reset_states(self):
        self.lif.reset_states()

class SNN_Model(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        tau_c1 = 5.0
        tau_c2 = 10.0
        tau_c3 = 15.0
        tau_c4 = 20.0
        tau_temp = 25.0
        tau_slow = 35.0
        threshold = 1.0

        self.conv1 = SNNConvLayer(2, 16, (3,3), tau_c1, threshold, stride=1)
        self.pool1 = SNNConvLayer(16, 16, (2,2), tau_c1, threshold, stride=2, padding=0)

        self.conv2 = SNNConvLayer(16, 32, (3,3), tau_c2, threshold, stride=1)
        self.pool2 = SNNConvLayer(32, 32, (2,2), tau_c2, threshold, stride=2, padding=0)

        self.conv3 = SNNConvLayer(32, 64, (3,3), tau_c3, threshold, stride=1)
        self.pool3 = SNNConvLayer(64, 64, (2,2), tau_c3, threshold, stride=2, padding=0)

        self.conv4 = SNNConvLayer(64, 128, (3,3), tau_c4, threshold, stride=1)
        self.pool4 = SNNConvLayer(128, 128, (2,2), tau_c4, threshold, stride=2, padding=0)

        self.flatten_dim = 128 * 4 * 4
        self.hidden_dim = 128

        self.fc1 = nn.Linear(self.flatten_dim, self.hidden_dim, bias=False)
        self.fc1_bn = nn.GroupNorm(num_groups=1, num_channels=self.hidden_dim, affine=True, eps=0.01)
        self.fc1_lif = PLIF(channels=self.hidden_dim, init_tau=tau_c4, threshold=threshold)

        self.temporal_conv = nn.Conv1d(in_channels=self.hidden_dim, 
                                       out_channels=self.hidden_dim, 
                                       kernel_size=3, padding=0, bias=False)
        self.temporal_bn = nn.GroupNorm(num_groups=1, num_channels=self.hidden_dim, affine=True, eps=0.01)
        self.drop_spatial = nn.Dropout1d(p=0.2)
        self.temporal_lif = PLIF(channels=self.hidden_dim, init_tau=tau_temp, threshold=threshold)
        
        self.temporal_buffer = None
        self.spk_rec = None
        
        self.fc_recurrent = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.lif_fc = PLIF(channels=self.hidden_dim, init_tau=tau_slow, threshold=threshold)
        self.fc2 = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        outputs = []
        
        if t > 1:
            self.reset_states()
            
        if self.spk_rec is None or self.spk_rec.shape[0] != b:
            self.spk_rec = torch.zeros(b, self.hidden_dim, device=x.device)
            
        if t == 1 and self.temporal_buffer is None:
            self.temporal_buffer = torch.zeros(b, self.hidden_dim, 2, device=x.device)
        
        total_spikes = 0.0
        num_spike_layers = 11 # 8 spatial + 1 bottleneck + 1 temporal + 1 recurrent

        spatial_bottleneck_spikes = []
        
        # Phase 1: Spatial Feature Extraction & Bottleneck
        for step in range(t):
            frame = x[:, step, :, :, :]

            out = self.conv1(frame)
            out = self.pool1(out)
            total_spikes += out.mean()

            out = self.conv2(out)
            out = self.pool2(out)
            total_spikes += out.mean()

            out = self.conv3(out)
            out = self.pool3(out)
            total_spikes += out.mean()

            out = self.conv4(out)
            out = self.pool4(out)
            total_spikes += out.mean()

            out_flat = out.reshape(b, -1)
            
            dense_out = self.fc1_bn(self.fc1(out_flat))
            spike_out = self.fc1_lif(dense_out)
            total_spikes += spike_out.mean()
            
            spatial_bottleneck_spikes.append(spike_out)

        # Phase 2: Temporal Convolution 
        spatial_tensor = torch.stack(spatial_bottleneck_spikes, dim=1).permute(0, 2, 1)
        spatial_tensor = self.drop_spatial(spatial_tensor)
        
        if t == 1:
            temporal_input = torch.cat([self.temporal_buffer, spatial_tensor], dim=2)
            self.temporal_buffer = temporal_input[:, :, 1:].detach()
            
            # temporal_input is 3 frames (2 buffered + 1 current). Kernel of 3 produces 1 frame.
            temporal_continuous = self.temporal_conv(temporal_input)
            temporal_continuous = self.temporal_bn(temporal_continuous)
            temporal_continuous = temporal_continuous[:, :, -1:] 
        else:
            padded_spatial = F.pad(spatial_tensor, (2, 0))
            
            # Convolution now looks at [t-2, t-1, t] to compute frame t.
            temporal_continuous = self.temporal_conv(padded_spatial)
            temporal_continuous = self.temporal_bn(temporal_continuous)
            
        temporal_continuous = temporal_continuous.permute(0, 2, 1)

        # Phase 3: Spiking Temporal Integration & Recurrence
        for step in range(t):
            temporal_spike = self.temporal_lif(temporal_continuous[:, step, :])
            total_spikes += temporal_spike.mean()
            
            feedback = self.fc_recurrent(self.spk_rec)
            dense_input = temporal_spike + feedback
            
            self.spk_rec = self.lif_fc(dense_input)
            total_spikes += self.spk_rec.mean()
            
            final = self.fc2(self.spk_rec)
            outputs.append(final)

        batch_spike_loss = total_spikes / (t * num_spike_layers)
        
        return torch.stack(outputs, dim=1), batch_spike_loss

    def reset_states(self):
        self.conv1.reset_states()
        self.pool1.reset_states()
        self.conv2.reset_states()
        self.pool2.reset_states()
        self.conv3.reset_states()
        self.pool3.reset_states()
        self.conv4.reset_states()
        self.pool4.reset_states()
        self.fc1_lif.reset_states()
        self.temporal_lif.reset_states()
        self.lif_fc.reset_states()
        self.spk_rec = None
        self.temporal_buffer = None

    def get_grad_width(self):
        return SurrogateExponential.grad_width
        
    def set_grad_width(self, width):
        SurrogateExponential.grad_width = width
        
if __name__ == "__main__":
    model = SNN_Model()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Recurrent SNN Parameters: {params:,}")