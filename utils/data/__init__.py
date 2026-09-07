def get_dataset(dataset_name):
    if dataset_name == 'cornell':
        from .cornell_data import CornellDataset
        return CornellDataset
    elif dataset_name == 'jacquard':
        from .jacquard_data import JacquardDataset
        return JacquardDataset
    elif dataset_name =='multiobj':
        from .multiobj_data import MulitObjDataset
        return MulitObjDataset
    elif dataset_name == 'vmrd':
        from .vmrd_data import VMRDDataset
        return VMRDDataset
    elif dataset_name == 'graspnet':
        from .graspnet_data import GraspNetDataset
        return GraspNetDataset
    elif dataset_name == 'CBRGD':
        from .CBRGD_data import CBRGDDataset
        return CBRGDDataset
    elif dataset_name == 'real':
        from .real_data import CBRGDDataset
        return CBRGDDataset
    elif dataset_name == 'realscene':
        from .real_dataset_data import RealDataset
        return RealDataset
    else:
        raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))
