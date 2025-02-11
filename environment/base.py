from abc import ABC, abstractmethod

class AbstractEnvironment(ABC):
    @abstractmethod
    def reset(self):
        """Reset environment to initial state."""
        raise NotImplementedError
    
    @abstractmethod
    def step(self, action):
        """Take one step in the environment given an action."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def observation_space(self):
        """Return observation space."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def action_space(self):
        """Return action space."""
        raise NotImplementedError
        
        
class AbstractPuppet(ABC):
    @abstractmethod
    def substep(self, action):
        raise NotImplementedError
