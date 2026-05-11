# TESTING 
import unittest
import numpy as np
import os

from layers import CNN, Conv2D, Flatten, MaxAvgPooling2D, Dense, FungsiAktivasi, LocallyConnected2D

class TestCNNModels(unittest.TestCase):

    def test_save_and_load_weights(self):        
        C_in = 3
        C_out = 2
        input_scratch = np.random.rand(5, 5, C_in).astype(np.float32)
        
        dummy_kernel = np.random.rand(3, 3, C_in, C_out)
        dummy_bias = np.random.rand(C_out)
        
        dense_input_dim = 1 * 1 * C_out
        dummy_dense_weights = np.random.rand(dense_input_dim, 10)
        dummy_dense_bias = np.random.rand(10)
        
        model_cnn = CNN(layers=[
            Conv2D(kernel_weights=dummy_kernel, bias_weights=dummy_bias, stride=1, padding=0),
            FungsiAktivasi(tipe='relu'),
            MaxAvgPooling2D(pool_size=2, stride=2, tipe='max'),
            Flatten(),
            Dense(weights=dummy_dense_weights, bias=dummy_dense_bias)
        ])
        
        hasil_awal = model_cnn.forward(input_scratch)
        
        test_filepath = "test_bobot_dummy.npz"
        model_cnn.save_weights(test_filepath)
        
        model_baru = CNN(layers=[
            Conv2D(kernel_weights=np.zeros_like(dummy_kernel), bias_weights=np.zeros_like(dummy_bias), stride=1, padding=0),
            FungsiAktivasi(tipe='relu'),
            MaxAvgPooling2D(pool_size=2, stride=2, tipe='max'),
            Flatten(),
            Dense(weights=np.zeros_like(dummy_dense_weights), bias=np.zeros_like(dummy_dense_bias))
        ])
        
        model_baru.load_weights(test_filepath)
        hasil_setelah_load = model_baru.forward(input_scratch)        
    
        np.testing.assert_allclose(
            hasil_awal, 
            hasil_setelah_load, 
            err_msg="GAGAL: Output model baru berbeda dengan model asal. Bobot tidak ter-load sempurna!"
        )
        
        file_npz = test_filepath + ".npz"
        if os.path.exists(file_npz):
            os.remove(file_npz)

    def test_forward_pass(self):
        C_in = 3
        C_out = 2
        input_scratch = np.random.rand(5, 5, C_in).astype(np.float32)
        
        dummy_kernel = np.random.rand(3, 3, C_in, C_out)
        dummy_bias = np.random.rand(C_out)
        
        dense_input_dim = 1 * 1 * C_out
        dummy_dense_weights = np.random.rand(dense_input_dim, 10)
        dummy_dense_bias = np.random.rand(10)
        
        model_cnn = CNN(layers=[
            Conv2D(kernel_weights=dummy_kernel, bias_weights=dummy_bias, stride=1, padding=0),
            FungsiAktivasi(tipe='relu'),
            MaxAvgPooling2D(pool_size=2, stride=2, tipe='max'),
            Flatten(),
            Dense(weights=dummy_dense_weights, bias=dummy_dense_bias)
        ])
        
        output = model_cnn.forward(input_scratch)
        
        self.assertEqual(output.shape, (10,), "GAGAL: Output dari forward pass tidak memiliki shape yang diharapkan (10,)")

    def test_locally_connected_layer(self):
        H, W, C_in = 5, 5, 3
        
        kH, kW = 3, 3
        stride = 1
        C_out = 2
        
        out_H = int((H - kH) / stride) + 1
        out_W = int((W - kW) / stride) + 1
        num_positions = out_H * out_W
        patch_dim = kH * kW * C_in
        
        dummy_kernel = np.random.rand(num_positions, patch_dim, C_out)
        dummy_bias = np.random.rand(num_positions, C_out)
        
        scratch_lc = LocallyConnected2D(
            kernel_weights=dummy_kernel, 
            bias_weights=dummy_bias, 
            kernel_size=(kH, kW), 
            stride=stride
        )
        
        self.assertEqual(
            scratch_lc.kernel.shape, 
            (num_positions, patch_dim, C_out), 
            "GAGAL: Bobot kernel LC dari scratch tidak sesuai dengan standar arsitektur Keras!"
        )

    def test_flatten_layer(self):
        input_tensor = np.random.rand(4, 4, 3)
        flatten_layer = Flatten()
        output = flatten_layer.forward(input_tensor)
        
        self.assertEqual(
            output.shape, 
            (48,), 
            "GAGAL: Output dari Flatten tidak memiliki shape yang diharapkan (48,)"
        )

if __name__ == '__main__':
    unittest.main()