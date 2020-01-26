from abc import ABCMeta, abstractmethod


class VisualizationInterface(metaclass=ABCMeta):
    
    @abstractmethod
    def visualize_mask():
        pass
    
    @abstractmethod
    def process_output():
        pass

class VisualizationSaveImages(VisualizationInterface):
    def visualize_mask():
        pass
    
    def process_output():
        pass

class VisualizationTensorboard(VisualizationInterface):
    def visualize_mask():
        pass
    
    def process_output():
        pass