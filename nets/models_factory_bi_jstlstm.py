import tensorflow as tf
from nets import trape_bi_jstlstm, inter_trape_bi_cslstm_t2s2, inter_trape_bi_cslstm_t1s3, inter_trape_bi_cslstm_t3s1, inter_trape_bi_cslstm_t4s0, inter_trape_bi_cslstm_t0s4


networks_map = {'trape_bi_jstlstm':trape_bi_jstlstm.rnn,
                'inter_trape_bi_cslstm': inter_trape_bi_cslstm_t2s2.rnn,
                'inter_trape_bi_cslstm_t1s3': inter_trape_bi_cslstm_t1s3.rnn,
                'inter_trape_bi_cslstm_t3s1': inter_trape_bi_cslstm_t3s1.rnn,
                'inter_trape_bi_cslstm_t4s0': inter_trape_bi_cslstm_t4s0.rnn,
                'inter_trape_bi_cslstm_t0s4': inter_trape_bi_cslstm_t0s4.rnn,
               }



def construct_model(name, images, images_rev, mask_true, num_layers, num_hidden,
                    filter_size, stride, seq_length, input_length, tln):
    '''Returns a sequence of generated frames
    Args:
        name: [predrnn_pp]
        mask_true: for schedualed sampling.
        num_hidden: number of units in a lstm layer.
        filter_size: for convolutions inside lstm.
        stride: for convolutions inside lstm.
        seq_length: including ins and outs.
        input_length: for inputs.
        tln: whether to apply tensor layer normalization.
    Returns:
        gen_images: a seq of frames.
        loss: [l2 / l1+l2].
    Raises:
        ValueError: If network `name` is not recognized.
    '''
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    func = networks_map[name]
    return func(images, images_rev, mask_true, num_layers, num_hidden, filter_size,
                stride, seq_length, input_length, tln)
