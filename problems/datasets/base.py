import pathlib

import numpy as np
import pandas as pd
from sklearn.utils import Bunch
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from ..base import ProblemRepr

from sklearn.datasets import load_diabetes

DATASETS_PATH = (pathlib.Path(__file__).parent / 'data').resolve()
CLASS_DATASETS_PATH = (pathlib.Path(__file__).parent / 'class_data').resolve()

CLASSIFICATION_TASKS = ["iris", "breastcancer", "dry_bean", "abalone", "car", "cirrhosis", "raisin", "balance-scale"]

def is_classification(taskname: str) -> bool:
    return taskname in CLASSIFICATION_TASKS

def load_dataset(filename: str, target_column: str, return_X_y: bool, as_frame: bool,
                 remove_columns: list = None) -> ProblemRepr:
    frame = pd.read_csv(DATASETS_PATH / filename, sep=',')

    data = frame.drop(columns=[target_column] + (remove_columns if remove_columns is not None else []))
    target = frame[target_column]

    if not as_frame:
        data = data.to_numpy(dtype=float)
        target = target.to_numpy(dtype=float)

    if return_X_y:
        return data, target
    elif as_frame:
        return Bunch(frame=frame, data=data, target=target)
    else:
        return Bunch(X=data, y=target)
    

def load_class_dataset(filename: str, target_column: str, return_X_y: bool, as_frame: bool,
                 remove_columns: list = None, label_to_num: bool = True, oneHotEncoding: list = None) -> ProblemRepr:
    frame = pd.read_csv(CLASS_DATASETS_PATH / filename, sep=',')
    frame = frame.dropna()
    data = frame.drop(columns=[target_column] + (remove_columns if remove_columns is not None else []))
    target = frame[target_column]

    if not as_frame:
        data = data.to_numpy()
        target = target.to_numpy()
    
    if label_to_num:
        labels = np.unique(target)
        toNum = dict(zip(labels, range(1, len(labels)+1)))
        target = [toNum[x] for x in target]
        target = np.array(target)
    
    if oneHotEncoding is not None:
        transformer = make_column_transformer(
            ((OneHotEncoder(sparse=False)), oneHotEncoding),
            remainder='passthrough')
        # transforming
        data = transformer.fit_transform(data)

    if return_X_y:
        return data, target
    elif as_frame:
        return Bunch(frame=frame, data=data, target=target)
    else:
        return Bunch(X=data, y=target)

def load_raisin(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the dataset.

    ==============   ==================
    Samples total    900
    Dimensionality   7
    Features         real, TODO: ranges
    Targets          strings, binary,
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/850/raisin.
    """
    return load_class_dataset(filename='raisin.csv', target_column='Class', return_X_y=return_X_y, as_frame=as_frame)

def load_cirrhosis(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the dataset.

    ==============   ==================
    Samples total    345
    Dimensionality   19
    Features         real, TODO: ranges
    Targets          string, 3 classes
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/dataset/878/cirrhosis+patient+survival+prediction+dataset-1.
    """
    return load_class_dataset(filename='cirrhosis.csv', target_column='Status', return_X_y=return_X_y, as_frame=as_frame, oneHotEncoding=[1,3,4,5,6,7,8])

def load_car(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the dataset.

    ==============   ==================
    Samples total    1728
    Dimensionality   6
    Features         categorical, TODO: ranges
    Targets          string, 4 classes
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/dataset/19/car+evaluation.
    """
    return load_class_dataset(filename='car.csv', target_column='class', return_X_y=return_X_y, as_frame=as_frame, oneHotEncoding=[0, 1, 2, 3, 4, 5])

def load_abalone(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the dataset.

    ==============   ==================
    Samples total    4177
    Dimensionality   8
    Features         real, TODO: ranges
    Targets          integer,  25 classes
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/dataset/1/abalone.
    """
    return load_class_dataset(filename='abalone.csv', target_column='Rings', return_X_y=return_X_y, as_frame=as_frame, oneHotEncoding=[0])

def load_iris(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the dataset.

    ==============   ==================
    Samples total    150
    Dimensionality   4
    Features         real, TODO: ranges
    Targets          string, 3 classes
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/dataset/53/iris.
    """
    return load_class_dataset(filename='iris.csv', target_column='y', return_X_y=return_X_y, as_frame=as_frame)

def load_dry_bean(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the dataset.

    ==============   ==================
    Samples total    13611
    Dimensionality   16
    Features         real, TODO: ranges
    Targets          string, multi-class, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/dataset/602/dry+bean+dataset.
    """
    return load_class_dataset(filename='dry_bean.csv', target_column='y', return_X_y=return_X_y, as_frame=as_frame)

def load_breastcancer(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the dataset.

    ==============   ==================
    Samples total    569
    Dimensionality   30
    Features         real, TODO: ranges
    Targets          binary, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic.
    """
    return load_class_dataset(filename='breastcancer.csv', target_column='Y', return_X_y=return_X_y, as_frame=as_frame)


def load_combined_cycle_power_plant(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Combined Cycle Power Plant dataset.

    ==============   ==================
    Samples total    9568
    Dimensionality   4
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant.
    """
    return load_dataset(filename='ccpp.csv', target_column='PE', return_X_y=return_X_y, as_frame=as_frame)


def load_gas_turbine(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Gas Turbine dataset.

    ==============   ==================
    Samples total    36733
    Dimensionality   10
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/Gas+Turbine+CO+and+NOx+Emission+Data+Set.
    """
    return load_dataset(filename='gas_turbine.csv', target_column='TEY', return_X_y=return_X_y, as_frame=as_frame)


def load_concrete_strength(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Concrete Strength dataset.

    ==============   ==================
    Samples total    1030
    Dimensionality   8
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength.
    """
    return load_dataset(filename='concrete.csv', target_column='CCS', return_X_y=return_X_y, as_frame=as_frame)


def load_airfoil_self_noise(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Airfoil Self-Noise dataset.

    ==============   ==================
    Samples total    1503
    Dimensionality   5
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise.
    """
    return load_dataset(filename='airfoil_self_noise.csv', target_column='SPL', return_X_y=return_X_y,
                        as_frame=as_frame)


def load_energy_heat(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Energy efficiency dataset with heating load as target.

    ==============   ==================
    Samples total    768
    Dimensionality   8
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/energy+efficiency.
    """

    return load_dataset(filename='energy.csv', target_column='Y1', return_X_y=return_X_y,
                        as_frame=as_frame, remove_columns=['Y2'])


def load_energy_cool(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Energy efficiency dataset with cooling load as target.

    ==============   ==================
    Samples total    768
    Dimensionality   8
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/energy+efficiency.
    """

    return load_dataset(filename='energy.csv', target_column='Y2', return_X_y=return_X_y,
                        as_frame=as_frame, remove_columns=['Y1'])


def load_forest_fires(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Forest Fires dataset.

    ==============   ==================
    Samples total    517
    Dimensionality   13
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/forest+fires.
    """

    return load_dataset(filename='forest_fires.csv', target_column='area', return_X_y=return_X_y,
                        as_frame=as_frame)


def load_parkinson_total(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Parkinson dataset with total UPDRS as target.

    ==============   ==================
    Samples total    5875
    Dimensionality   26
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/parkinsons+telemonitoring.
    """

    return load_dataset(filename='parkinson.csv', target_column='total_UPDRS', return_X_y=return_X_y,
                        as_frame=as_frame, remove_columns=['subject#', 'test_time', 'motor_UPDRS'])


def load_parkinson_motor(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Parkinson dataset with motor UPDRS as target.

    ==============   ==================
    Samples total    5875
    Dimensionality   26
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/parkinsons+telemonitoring.
    """

    return load_dataset(filename='parkinson.csv', target_column='motor_UPDRS', return_X_y=return_X_y,
                        as_frame=as_frame, remove_columns=['subject#', 'test_time', 'total_UPDRS'])


def load_protein_structure(return_X_y: bool = True, as_frame: bool = False, sample: bool = True):
    """ Load and return the Protein Structure dataset.
        ==============   ==================
        Samples total    45730
        Dimensionality   9
        Features         real, TODO: ranges
        Targets          real, TODO: ranges
        ==============   ==================
        Downloaded from
        https://https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure.
        """
    return load_dataset(filename='protein_structure.csv', target_column='RMSD', return_X_y=return_X_y,
                        as_frame=as_frame)
