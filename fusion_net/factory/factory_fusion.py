# -*- coding: utf-8 -*-
import block
fusion_types = ['block', 'linear_sum', 'concat_mlp', 'mlb', 'tucker', 'mutan', 'block_tucker', \
                'mfb', 'mfh', 'mcb']
def factory_fusion(input_dims, output_dim, fusion_type, dropout):
    if fusion_type not in fusion_types:
        raise ValueError("Wrong fusion specification, please choose one from {}".format(fusion_types))
    
    if fusion_type == 'block':
        return block.fusions.Block(input_dims, output_dim, dropout_input=dropout, \
                                   dropout_pre_lin=dropout, dropout_output=dropout)
    
    if fusion_type == 'linear_sum':
        return block.fusions.LinearSum(input_dims, output_dim, dropout_input=dropout, \
                                   dropout_pre_lin=dropout, dropout_output=dropout)
    
    if fusion_type == 'concat_mlp':
        return block.fusions.ConcatMLP(input_dims, output_dim, dropout=dropout)
    
    if fusion_type == 'mlb':
        return block.fusions.MLB(input_dims, output_dim, dropout_input=dropout, \
                                   dropout_pre_lin=dropout, dropout_output=dropout)
    
    if fusion_type == 'tucker':
        return block.fusions.Tucker(input_dims, output_dim, dropout_input=dropout, \
                                   dropout_pre_lin=dropout, dropout_output=dropout)
    
    if fusion_type == 'mutan':
        return block.fusions.Mutan(input_dims, output_dim, dropout_input=dropout, \
                                   dropout_pre_lin=dropout, dropout_output=dropout)
    
    if fusion_type == 'block_tucker':
        return block.fusions.BlockTucker(input_dims, output_dim, dropout_input=dropout, \
                                   dropout_pre_lin=dropout, dropout_output=dropout)
    
    if fusion_type == 'mfb':
        return block.fusions.MFB(input_dims, output_dim, dropout_input=dropout, \
                                   dropout_pre_lin=dropout, dropout_output=dropout)
    
    if fusion_type == 'mfh':
        return block.fusions.MFH(input_dims, output_dim, dropout_input=dropout, \
                                   dropout_pre_lin=dropout, dropout_output=dropout)
    
    if fusion_type == 'mcb':
        return block.fusions.MCB(input_dims, output_dim, dropout_output=dropout)
