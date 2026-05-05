import numpy as np

class CNN:
    def __init__(self, layers):
        # list of layers
        self.layers = layers
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def save_weights(self, filepath):
        weights_dict = {}
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, 'kernel') and hasattr(layer, 'bias'):
                weights_dict[f'layer_{idx}_kernel'] = layer.kernel
                weights_dict[f'layer_{idx}_bias'] = layer.bias
                
        filepath = 'src/weights/' + filepath if not filepath.endswith('.npz') else filepath
        np.savez(filepath, **weights_dict)
        print(f"Bobot model berhasil disimpan ke {filepath}.npz")

    def load_weights(self, filepath):
        if not filepath.endswith('.npz'):
            filepath = 'src/weights/' + filepath + '.npz'
            
        data = np.load(filepath)
        for idx, layer in enumerate(self.layers):
            kernel_key = f'layer_{idx}_kernel'
            bias_key = f'layer_{idx}_bias'
            
            if kernel_key in data.files and bias_key in data.files:
                layer.kernel = data[kernel_key]
                layer.bias = data[bias_key]
                
        print(f"Bobot model berhasil di-load dari {filepath}.npz")

class Conv2D:
    def __init__(self, kernel_weights, bias_weights, stride=1, padding=0):
        self.kernel = kernel_weights
        self.bias = bias_weights
        self.stride = stride
        self.padding = padding

        self.kH, self.kW, self.C_in, self.C_out = self.kernel.shape
        
    def forward(self, input_tensor):
        if self.padding > 0:
            p = self.padding        
            input_tensor = np.pad(input_tensor, ((p, p), (p, p), (0, 0)), mode='constant', constant_values=0)

        H, W, C_in_input = input_tensor.shape

        assert C_in_input == self.C_in, "Jumlah channel input tidak sesuai dengan kernel"

        # hitung output dimensions
        out_H = int((H - self.kH) / self.stride) + 1
        out_W = int((W - self.kW) / self.stride) + 1
        output_tensor = np.zeros((out_H, out_W, self.C_out))

        output_tensor = np.zeros((out_H, out_W, self.C_out))

        for h in range(out_H):
            for w in range(out_W):
                # batas window/patch pada gambar
                h_start = h * self.stride
                h_end = h_start + self.kH
                w_start = w * self.stride
                w_end = w_start + self.kW
                
                patch = input_tensor[h_start:h_end, w_start:w_end, :]
                
                for k in range(self.C_out):
                    kernel_k = self.kernel[:, :, :, k]                    
                    output_tensor[h, w, k] = np.sum(patch * kernel_k) + self.bias[k]
                    
        return output_tensor
    
class LocallyConnected2D:
    def __init__(self, kernel_weights, bias_weights, kernel_size, stride=1):
        self.kernel = kernel_weights
        self.bias = bias_weights
        self.kH, self.kW = kernel_size
        self.stride = stride

    def forward(self, input_tensor):
        H, W, C_in = input_tensor.shape
        out_H = int((H - self.kH) / self.stride) + 1
        out_W = int((W - self.kW) / self.stride) + 1

        num_positions, patch_dim, C_out = self.kernel.shape
        assert num_positions == out_H * out_W, \
            f"ERROR: Bobot mengharapkan {num_positions} posisi spasial, tapi hitungan kita {out_H * out_W}"
        assert patch_dim == self.kH * self.kW * C_in, \
            "ERROR: Dimensi patch input tidak sesuai dengan arsitektur bobot kernel!"
        
        output_tensor = np.zeros((out_H, out_W, C_out))

        for h in range(out_H):
            for w in range(out_W):
                position = h * out_W + w

                # batas patch pada gambar
                h_start = h * self.stride
                h_end = h_start + self.kH
                w_start = w * self.stride
                w_end = w_start + self.kW
                
                patch = input_tensor[h_start:h_end, w_start:w_end, :]
                patch_flatten = patch.flatten()

                kernel_position = self.kernel[position]
                bias_position = self.bias[position]
                
                output_tensor[h, w, :] = np.dot(patch_flatten, kernel_position) + bias_position

        return output_tensor

class MaxAvgPooling2D:
    def __init__(self, pool_size=2, stride=2, tipe='max'):
        self.pool_size = pool_size
        self.stride = stride
        self.tipe = tipe

    def forward(self, input_tensor, tipe=None):
        H, W, C = input_tensor.shape
        out_H = int((H - self.pool_size) / self.stride) + 1
        out_W = int((W - self.pool_size) / self.stride) + 1

        output_tensor = np.zeros((out_H, out_W, C))
        for h in range(out_H):
            for w in range(out_W):
                h_start = h * self.stride
                h_end = h_start + self.pool_size
                w_start = w * self.stride
                w_end = w_start + self.pool_size
                
                patch = input_tensor[h_start:h_end, w_start:w_end, :]
                if tipe == 'max': 
                    output_tensor[h, w, :] = np.max(patch, axis=(0, 1))
                elif tipe == 'avg':
                    output_tensor[h, w, :] = np.mean(patch, axis=(0, 1))

        return output_tensor
    
class GlobalMaxAvgPooling2D:
    def __init__(self, tipe='max'):
        self.tipe = tipe
        
    def forward(self, input_tensor, tipe=None):
        if tipe is None:
            tipe = self.tipe
            
        if tipe == 'max':
            return np.max(input_tensor, axis=(0, 1))
        elif tipe == 'avg':
            return np.mean(input_tensor, axis=(0, 1))
    
class Flatten:
    def forward(self, input_tensor):
        return input_tensor.flatten()

class Dense:
    def __init__(self, weights, bias):
        self.kernel = weights  
        self.bias = bias
        
    def forward(self, input_tensor):
        expected_input_dim = self.kernel.shape[0]
        actual_input_dim = input_tensor.shape[0]
        
        assert actual_input_dim == expected_input_dim, f"ERROR DENSE: Harapannya input ukuran {expected_input_dim}, tapi dapat {actual_input_dim}!"
        
        output = np.dot(input_tensor, self.kernel) + self.bias
        return output

# fungsi aktivasi
class FungsiAktivasi:
    def __init__(self, tipe='relu'):
        self.tipe = tipe
        
    def forward(self, input_tensor, tipe=None):
        if tipe is None:
            tipe = self.tipe
            
        if tipe == 'relu':
            return np.maximum(0, input_tensor)
        elif tipe == 'sigmoid':
            return 1 / (1 + np.exp(-input_tensor))
        elif tipe == 'tanh':
            return np.tanh(input_tensor)
        elif tipe == 'softmax':
            exp_values = np.exp(input_tensor - np.max(input_tensor))
            return exp_values / np.sum(exp_values)