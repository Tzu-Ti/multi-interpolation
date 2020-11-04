from data_provider import mnist
from data_provider import ucf101
from data_provider import NCTU_MIRC_1F
from data_provider import NCTU_MIRC_1F_DoubleUp
from data_provider import rovit
from data_provider import BiTAI_base_dataset

datasets_map = {
    'mnist': mnist, 'ucf101': ucf101, 'NCTU_MIRC_1F': NCTU_MIRC_1F, 'NCTU_MIRC_1F_DoubleUp': NCTU_MIRC_1F_DoubleUp, 'rovit': rovit, 'BiTAI_base_dataset': BiTAI_base_dataset,
}

def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size,
                  img_size=[128,128,3], seq_len=11, is_training=True):
    '''Given a dataset name and returns a Dataset.
    Args:
        dataset_name: String, the name of the dataset.
        train_data_paths: List, [train_data_path1, train_data_path2...]
        valid_data_paths: List, [val_data_path1, val_data_path2...]
        batch_size: Int
        img_size: [Int height, Int width, Int channel]
        is_training: Bool
    Returns:
        if is_training:
            Two dataset instances for both training and evaluation.
        else:
            One dataset instance for evaluation.
    Raises:
        ValueError: If `dataset_name` is unknown.
    '''

    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')

    if dataset_name == 'mnist' or dataset_name == 'ucf101' or dataset_name == 'NCTU_MIRC_1F' or dataset_name == 'NCTU_MIRC_1F_DoubleUp' or dataset_name == 'rovit' or dataset_name == 'BiTAI_base_dataset':  
        test_input_param = {'paths': valid_data_list,
                            'minibatch_size': batch_size,
                            'input_data_type': 'float32',
                            'is_output_sequence': True,
                            'name': dataset_name+'test iterator',
                            'img_size': img_size,
                            'seq_len': seq_len}
        test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param)
        test_input_handle.begin(do_shuffle = False)
        if is_training:
            train_input_param = {'paths': train_data_list,
                                 'minibatch_size': batch_size,
                                 'input_data_type': 'float32',
                                 'is_output_sequence': True,
                                 'name': dataset_name+' train iterator',
                                 'img_size': img_size,
                                 'seq_len': seq_len}
            train_input_handle = datasets_map[dataset_name].InputHandle(train_input_param)
            train_input_handle.begin(do_shuffle = True)
            return train_input_handle, test_input_handle
        else:
            return test_input_handle
