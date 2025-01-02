from dependencies import *


# 默认输入的 X 是 onehot encode 完成的，3d tensor
# 由于加法模的对称性，不需要 positional encoding
class Transformer(torch.nn.Module):
    def __init__(self, num_layers=2, p=97, seq_len=2, dropout=0.1):
        super().__init__()
        self.model = nn.Sequential(
            *[TransformerDecoder(p, dropout=dropout) for _ in range(num_layers)],
            nn.LayerNorm(p),
            nn.Linear(p, p)
        )

    def forward(self, inputs, add_noise=False, noise_std=0.0):
        if add_noise and noise_std > 0:
            # 生成噪声
            noise = torch.randn_like(inputs) * noise_std
            # 添加噪声到输入的embedding
            perturbed_inputs = inputs + noise
            return self.model(perturbed_inputs)
        return self.model(inputs)


class TransformerDecoder(torch.nn.Module):
    def __init__(self, p=97, multiplier=4, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim=p, num_heads=1, batch_first=True)
        self.attn_layer_norm = nn.LayerNorm(p)
        if dropout > 0:
            self.attn_dropout = nn.Dropout(dropout)  # 注意力层后的 Dropout
        self.fc = nn.Sequential(
            nn.Linear(p, p * multiplier),
            nn.GELU(),
            nn.Dropout(dropout) if dropout>0 else nn.Identity(),
            nn.Linear(p * multiplier, p),
        )
        self.fc_layer_norm = nn.LayerNorm(p)

    def forward(self, x):
        out1, _ = self.self_attn(x, x, x)
        out1 = self.attn_layer_norm(x + out1)
        out2 = self.fc(out1)
        out = self.fc_layer_norm(out1 + out2)
        return out



# 默认输入的 X 是 onehot encode 完成的，3d tensor
class MLP(torch.nn.Module):
    def __init__(self, layer_dims=[512, 128, 64, 64], p=97):
        """
        Args:   
            layer_dims (list[int]): List of integers defining the dimensions of each layer.
                For example, [input_dim, hidden1_dim, hidden2_dim, ..., output_dim].
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),  # 将 3D Tensor (batch_size, seq_len, p) 转换为 2D Tensor (batch_size, seq_len * p)
            nn.Linear(p*2, layer_dims[0]),
            nn.GELU(),
            *[
                nn.Sequential(
                    nn.Linear(layer_dims[i], layer_dims[i + 1]),  # Linear transformation
                    nn.GELU()
                )
                for i in range(len(layer_dims) - 1)

            ],
            nn.Linear(layer_dims[-1], p),
            # nn.LayerNorm(p),
        )

    def forward(self, inputs):
        return self.layers(inputs).unsqueeze(1)  # 将(batch_size, p)转换为(batch_size, 1, p), 为了与 Transformer 的输出维度一致
    
    def find_linear_layers(self, module, linear_layers):
        """
        递归遍历模型，找到所有 nn.Linear 层及其索引。
        Args:
            module: 模型或子模块。
            linear_layers: 用于存储找到的 nn.Linear 层的列表。
        """
        for layer in module.children():
            if isinstance(layer, nn.Linear):
                linear_layers.append(layer)
            elif isinstance(layer, nn.Sequential):
                self.find_linear_layers(layer, linear_layers)

    def masked_forward(self, inputs, masks):
        """
        使用 masks 对多层神经元进行筛选。
        支持递归处理 nn.Sequential 结构。
        
        Args:
            inputs: 模型的输入张量。
            masks: 字典，键为线性层索引，值为对应层的 mask。
        """
        x = self.layers[0](inputs)  # 处理 Flatten
        layer_idx = 0  # 记录逻辑线性层索引
        x, _ = self._apply_layers_with_masks(self.layers[1:], x, masks, layer_idx)  # 递归处理层
        return x.unsqueeze(1)  # 返回维度一致的结果

    def _apply_layers_with_masks(self, module, x, masks, layer_idx):
        """
        递归遍历模型的每一层，应用 mask 到线性层。
        
        Args:
            module: 当前模型或子模块（可能是 nn.Sequential）。
            x: 当前层的输入张量。
            masks: 字典，键为逻辑线性层索引，值为对应层的 mask。
            layer_idx: 当前逻辑线性层索引。
        
        Returns:
            处理后的输出张量。
        """
        for layer in module:
            if isinstance(layer, nn.Linear):
                # 检查是否需要对该线性层应用 mask
                if layer_idx in masks:
                    x = layer(x)  # 应用线性变换
                    mask = masks[layer_idx]
                    x = x * mask  # 应用 mask
                else:
                    x = layer(x)  # 应用线性变换但无 mask
                layer_idx += 1  # 更新逻辑线性层索引
            elif isinstance(layer, nn.Sequential):
                # 如果是嵌套 Sequential，递归进入
                x, layer_idx = self._apply_layers_with_masks(layer, x, masks, layer_idx)
            else:
                x = layer(x)  # 处理非线性激活或其他层
        return x, layer_idx


# 默认输入的 X 是 onehot encode 完成的，3d tensor
class LSTM(torch.nn.Module):
    def __init__(self, input_dim=97, hidden_dim=128, output_dim=97, num_layers=2, seq_len=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,  # 输入维度
            hidden_size=hidden_dim,  # 隐藏层维度
            num_layers=num_layers,  # LSTM 的层数
            batch_first=True  # 输入数据格式为 (batch_size, seq_len, input_dim)
        )
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)  # 对隐藏状态进行归一化
        self.fc = torch.nn.Linear(hidden_dim, output_dim)  # 输出层

    def forward(self, inputs):
        # LSTM 层
        lstm_out, _ = self.lstm(inputs)  # lstm_out: (batch_size, seq_len, hidden_dim)
        
        # 归一化 + 输出层
        norm_out = self.layer_norm(lstm_out)  # (batch_size, seq_len, hidden_dim)
        output = self.fc(norm_out)  # (batch_size, seq_len, output_dim)
        return output
