import argparse




__all__ = ['Args', 'TorchArgs']

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


class Args(argparse.ArgumentParser):
    ''' A class taht encapsulates commonly used arguments for torch training, inheriting from argparse.ArgumentParser.
    
    default arguments:
        seed: int, default 100, random seed
        batch_size: int, default 32, batch size
        lr: float, default 1e-4, learning rate
        epochs: int, default 100, epochs
        weight_decay: float, default 1e-5, weight decay
        load_model: bool, default False, whether load the exist model paramters
        load_epoch: int, default 10, The number of epochs for the trained model.
    
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_argument('--seed', type=int, default=100, help="seed of the random")
        self.add_argument('--batch_size', type=int, default=256, help="batch size")
        self.add_argument('--lr', type=float, default=1e-4, help="learning rate")
        self.add_argument('--epochs', type=int, default=100, help="epochs")
        self.add_argument('--weight_decay', type=float, default=1e-5, help="weight decay")
        self.add_argument('--load_model', type=boolean_string, default=False, help='whether load the exist trained model\'s paramters')
        self.add_argument('--load_epoch', type=int, default=10, help='The number of epochs for the trained model.')


        
        
        
        
            
            