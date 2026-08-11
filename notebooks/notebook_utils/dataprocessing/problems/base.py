from abc import ABC, abstractmethod
import numpy as np


class BaseProblem(ABC):
    """Base class for all physics problems."""
    
    def __init__(self, **kwargs):
        self.dof = kwargs.get('dof', 1)
        self.name = kwargs.get('name', 'base_problem')
    
    @abstractmethod
    def compute_du(self, u):
        """Compute time derivative of state vector."""
        pass
    
    @abstractmethod
    def compute_errors(self, u, ref_u):
        """Compute errors between two state vectors."""
        pass
    
    @classmethod
    def get_reference_filepaths(cls, category='default'):
        """
        Get reference trajectory filepaths for this problem.
        
        Parameters:
        -----------
        category : str, optional
            The category of reference trajectories to retrieve.
            Default is 'default'. Subclasses can define additional categories.
            
        Returns:
        --------
        dict
            Dictionary containing reference trajectory filepaths organized by
            parameter values and initial condition indices.
            Structure: {param_value: {ic_idx: {"filepath": str, "dt": float}}}
        """
        # Default implementation returns empty dict
        # Subclasses should override this method
        return {}
    
    @classmethod
    def get_available_reference_categories(cls):
        """
        Get list of available reference trajectory categories for this problem.
        
        Returns:
        --------
        list
            List of available category names for reference trajectories.
        """
        # Default implementation returns empty list
        # Subclasses should override this method
        return []