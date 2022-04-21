"""
    File regrouping unittests concerning models implementation.
"""    
import torch

from src.models import Encoder, Decoder

def test_encoder():
    """
        Testing the encoder model. 
    """
    dim_Ls = [2,3,4]
    dim_M = 3
    batch_size = 8
    models =  ['linear']
    devices = ["cuda:0", "cpu"]

    for model in models:
        for device in devices:
            for dim_L in dim_Ls:
                enc = Encoder(model = model,
                              dim_in = dim_L,
                              dim_out = dim_M)
                sentences = torch.rand(batch_size, dim_L, dtype = torch.float32)

                enc = enc.to(device)
                sentences = sentences.to(device)

                m = enc(sentences)

                assert m.shape == torch.Size([batch_size, dim_M])

def test_decoder():
    """
        Testing the decoder model. 
    """
    dim_Ls = [2,3,4]
    dim_M = 3
    batch_size = 8
    models =  ['linear']
    devices = ["cuda:0", "cpu"]

    for model in models:
        for device in devices:
            for dim_L in dim_Ls:
                dec = Decoder(model = model,
                              dim_in = dim_M,
                              dim_out = dim_L)
                m = torch.rand(batch_size, dim_M, dtype = torch.float32)

                dec = dec.to(device)
                m = m.to(device)

                sentences = dec(m)

                assert sentences.shape == torch.Size([batch_size, dim_L])






        
        
