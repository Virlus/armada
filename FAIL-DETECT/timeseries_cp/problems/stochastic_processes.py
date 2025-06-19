from abc import ABC, abstractmethod
import numpy as np


class AbstractStochasticProcess(ABC):
    """An abstract class for data-generating stochastic processes."""

    @abstractmethod
    def generate_data(
        self, batch_size: int, length: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Generates data and returns them in a numpy array format.
        Args:
            batch_size: Number of trajectories to be returned.
            length: Length of each trajectory.
            rng: An np.random.Generator object.

        Returns:
            A numpy array of size (batch_size, length).
        """


class RandomWalkProcess(AbstractStochasticProcess):
    """A Gaussian random walk process.

    Attributes:
        loc: mean of Gaussian noise
        scale: stdev of Gaussian noise
    """

    def __init__(self, loc: float, scale: float) -> None:
        """Initializes a RandomWalkProcess object.

        Args:
            loc: Mean of Gaussian noise.
            scale: Stdev of Gaussian noise.
        """
        self.loc = loc
        self.scale = scale

    def generate_data(
        self, batch_size: int, length: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Generates data and returns them in a numpy array format.
        Args:
            batch_size: Number of trajectories to be returned.
            length: Length of each trajectory.
            rng: An np.random.Generator object.

        Returns:
            A numpy array of size (batch_size, length).
        """
        noise_array = rng.normal(
            loc=self.loc, scale=self.scale, size=(batch_size, length)
        )
        data_array = np.cumsum(noise_array, axis=-1)
        assert data_array.shape == (batch_size, length)
        return data_array
