# -*- coding: utf-8 -*-
import block
fusion_types = ['block', 'linear_sum', 'concat_mlp', 'mlb', 'tucker', 'mutan', 'block_tucker', \
                'mfb', 'mfh', 'mcb']
def factory_fusion(config):
    fusion_type = config['type']
    if fusion_type not in fusion_types:
        raise ValueError("Wrong fusion specification, please choose one from {}".format(fusion_types))
    
    input_dims = config['input_dims']
    output_dims = config['output_dims']
    dropout_pre_lin = config['dropout_prelin']
    dropout_input = config['dropout_input']
    chunks = config['chunks']
    rank = config['rank']
    mm_dim = config['mm_dim']
    
    if fusion_type == 'block':
        return block.fusions.Block(input_dims, output_dims, dropout_input=dropout_input, \
                                   dropout_pre_lin=dropout_pre_lin, chunks=chunks, rank=rank, mm_dim=mm_dim)
    
    if fusion_type == 'linear_sum':
        return block.fusions.LinearSum(input_dims, output_dims, dropout_input=dropout_input, \
                                   dropout_pre_lin=dropout_pre_lin, chunks=chunks, rank=rank, mm_dim=mm_dim)
    
    if fusion_type == 'concat_mlp':
        return block.fusions.ConcatMLP(input_dims, output_dims)
    
    if fusion_type == 'mlb':
        return block.fusions.MLB(input_dims, output_dims, dropout_input=dropout_input, \
                                   dropout_pre_lin=dropout_pre_lin, chunks=chunks, rank=rank, mm_dim=mm_dim)
    
    if fusion_type == 'tucker':
        return block.fusions.Tucker(input_dims, output_dims, dropout_input=dropout_input, \
                                   dropout_pre_lin=dropout_pre_lin, chunks=chunks, rank=rank, mm_dim=mm_dim)
    
    if fusion_type == 'mutan':
        return block.fusions.Mutan(input_dims, output_dims, dropout_input=dropout_input, \
                                   dropout_pre_lin=dropout_pre_lin, chunks=chunks, rank=rank, mm_dim=mm_dim)
    
    if fusion_type == 'block_tucker':
        return block.fusions.BlockTucker(input_dims, output_dims, dropout_input=dropout_input, \
                                   dropout_pre_lin=dropout_pre_lin, chunks=chunks, rank=rank, mm_dim=mm_dim)
    
    if fusion_type == 'mfb':
        return block.fusions.MFB(input_dims, output_dims, dropout_input=dropout_input, \
                                   dropout_pre_lin=dropout_pre_lin, chunks=chunks, rank=rank, mm_dim=mm_dim)
    
    if fusion_type == 'mfh':
        return block.fusions.MFH(input_dims, output_dims, dropout_input=dropout_input, \
                                   dropout_pre_lin=dropout_pre_lin, chunks=chunks, rank=rank, mm_dim=mm_dim)
    
    if fusion_type == 'mcb':
        return block.fusions.MCB(input_dims, output_dims)
