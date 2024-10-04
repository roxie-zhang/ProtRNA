CONFIG = {
    'evoformer': {
        'mode' : 'primary',
        'msa_column_attention': {
            'dropout_rate': 0.0,
            'gating': True,
            'input_dim': 256,
            'key_dim': 256,
            'value_dim': 256,
            'output_dim': 256,
            'num_head': 8,
            'orientation': 'per_column',
            'shared_dropout': True,
        },

        'msa_row_attention_with_pair_bias': {
            'dropout_rate': 0.15,
            'gating': True,
            'num_head': 8,
            'input_dim': 256,
            'pair_input_dim': 128,
            'key_dim': 256,
            'value_dim': 256,
            'output_dim': 256,
            'orientation': 'per_row',
            'shared_dropout': True,
        },

        'outer_product_mean': {
            'dropout_rate': 0.0,
            'orientation': 'per_row',
            'shared_dropout': True,
            'num_outer_channel': 32,
            'num_output_channel': 128,
            'num_input_channel': 256,
        },

        'msa_transition': {
            'input_dim': 256,
            'num_intermediate_factor': 4,
            'dropout_rate': 0.,
            'orientation': 'per_row',
            'shared_dropout': True,
        },

        'pair_transition': {
            'input_dim': 128,
            'num_intermediate_factor': 4,
            'dropout_rate': 0.,
            'orientation': 'per_row',
            'shared_dropout': True,
        },

        'msa_column_global_attention': {
            'dropout_rate': 0.0,
            'gating': True,
            'input_dim': 64,
            'key_dim': 64,
            'value_dim': 64,
            'output_dim': 64,
            'num_head': 8,
            'orientation': 'per_column',
            'shared_dropout': True,
        },

        'triangle_multiplication_outgoing': {
            'dropout_rate': 0.25,
            'equation': 'ikc,jkc->ijc',
            'num_intermediate_channel': 128,
            'input_dim': 128,
            'output_channel': 128,
            'orientation': 'per_row',
            'shared_dropout': True,
            'gating': True,
        },

        'triangle_multiplication_incoming': {
            'dropout_rate': 0.25,
            'equation': 'kjc,kic->ijc',
            'num_intermediate_channel': 128,
            'input_dim': 128,
            'output_channel': 128,
            'orientation': 'per_row',
            'shared_dropout': True,
            'gating': True,
        },

        'triangle_attention_starting_node': {
            'dropout_rate': 0.25,
            'gating': True,
            'input_dim': 128,
            'key_dim': 128,
            'output_dim': 128,
            'num_head': 4,
            'orientation': 'per_row',
            'shared_dropout': True,
            'value_dim': 128,
        },

        'triangle_attention_ending_node': {
            'dropout_rate': 0.25,
            'gating': True,
            'input_dim': 128,
            'key_dim': 128,
            'output_dim': 128,
            'num_head': 4,
            'orientation': 'per_column',
            'shared_dropout': True,
            'value_dim': 128,
        },

    },
}
