from .toy import bbbp
from .toy import dropout
from .toy import ensemble
from .toy import evidential
from .toy import gaussian
from .toy import deterministic
from .toy.h_params import h_params

def get_correct_model(dataset, trainer):
    """ Hacky helper function to grab the right model for a given dataset and trainer. """
    dataset_loader = globals()[dataset]
    trainer_lookup = trainer.__name__.lower()
    model_pointer = dataset_loader.__dict__[trainer_lookup]
    return model_pointer