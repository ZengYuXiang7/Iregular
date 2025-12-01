from abc import ABC, abstractmethod


class ErrorCompensation(ABC):

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def append(self, hidden, target, predict):
        raise NotImplementedError("Method append() not implemented")

    @abstractmethod
    def set_ready(self):
        raise NotImplementedError("Method set_ready() not implemented")

    @abstractmethod
    def clear(self):
        raise NotImplementedError("Method clear() not implemented")

    @abstractmethod
    def correction(self, hidden):
        raise NotImplementedError("Method correction() not implemented")
