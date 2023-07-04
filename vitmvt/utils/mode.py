from enum import Enum, unique


@unique
class OptimizeMode(Enum):

    Minimize = 'minimize'
    Maximize = 'maximize'
