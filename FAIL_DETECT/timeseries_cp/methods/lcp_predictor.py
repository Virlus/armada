from enum import Enum
from typing import Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from timeseries_cp.methods.abstract_predictor import AbstractPredictor
from timeseries_cp.utils.data_utils import RegressionType, regress


class OptimizationType(Enum):
    KKT = 1
    KKTNoMaxUpperBound = 2


class LinearComplementarityPredictor(AbstractPredictor):
    """Linear Complementarity Programming method for conformal prediction based on `Conformal
    Prediction Regions for Time Series using Linear Complementarity Programming`
    (https://arxiv.org/abs/2304.01075).

    Attributes:
        optimization_type: Type of optimization.
        regression_type: Type of regression function (for point prediction).
        calibration_size_for_lcp: Number of calibration data points used for LCP optimization.
    """

    def __init__(
        self,
        optimization_type: OptimizationType,
        regression_type: RegressionType,
        calibration_size_for_lcp: int,
    ) -> None:
        """Initializes a LinearComplementarityProgrammingPredictor object.

        Args:
            optimization_type: Type of optimization.
            regression_type: Type of regression function (for point prediction).
            calibration_size_for_lcp: Number of calibration data points used for LCP optimization.
        """
        self.optimization_type = optimization_type
        self.regression_type = regression_type
        self.calibration_size_for_lcp = calibration_size_for_lcp

    def get_prediction_band(
        self, training_data: np.ndarray, calibration_data: np.ndarray, alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Obtains upper and lower prediction trajectory, which together constitutes a prediction
        band.

        Args:
            training_data: An np.ndarray object of size (train_size, length).
            calibration_data: An np.ndarray object of size (calibration_size, length).
            alpha: The significance level. The prediction set is valid with probability 1 - alpha.

        Returns:
            A tuple of (uppder_trajectory, lower_trajectory) that constitutes the prediction band,
            where each object is an np.ndarray with shape (1, length).

        Raises:
            NotImplementedError if unknown regression_type or optimization_type is given.
        """
        length = training_data.shape[-1]
        assert length == calibration_data.shape[-1]
        assert 0.0 < alpha < 1.0
        total_calibration_size = calibration_data.shape[0]
        assert total_calibration_size > self.calibration_size_for_lcp
        optimization_data = calibration_data[: self.calibration_size_for_lcp]
        calibration_data = calibration_data[self.calibration_size_for_lcp :]

        prediction_trajectory = regress(training_data, self.regression_type)
        prediction_errors_on_optimization_data = np.abs(
            optimization_data - prediction_trajectory
        )
        model = self._get_optimization_model(
            prediction_errors_on_optimization_data, alpha
        )
        model.optimize()
        weighting_coeffs = np.array(
            [v.x for v in filter(lambda v: "alphas" in v.varName, model.getVars())]
        )

        calibration_scores = [
            np.max(
                np.abs(calibration_trajectory - prediction_trajectory)
                * weighting_coeffs
            )
            for calibration_trajectory in calibration_data
        ]
        band_width = np.quantile(calibration_scores, 1.0 - alpha)
        upper_trajectory = prediction_trajectory + band_width / weighting_coeffs
        lower_trajectory = prediction_trajectory - band_width / weighting_coeffs

        assert upper_trajectory.shape == lower_trajectory.shape == (1, length)
        return (upper_trajectory, lower_trajectory)

    def get_one_sided_prediction_band(
        self,
        training_data: np.ndarray,
        calibration_data: np.ndarray,
        alpha: float,
        lower_bound: bool,
    ) -> np.ndarray:
        """Obtains either upper or lower prediction trajectory, depending on the argument
        lower_bound.

        Args:
            training_data: An np.ndarray object of size (train_size, length).
            calibration_data: An np.ndarray object of size (calibration_size, length).
            alpha: The significance level. The prediction set is valid with probability 1 - alpha.
            lower_bound: If True, lower-bound is returned. Otherwise, upper-bound is returned.

        Returns:
            An np.ndarray of shape (1, length) that represents either the lower-boudning or
            upper-bounding trajectory.

        Raises:
            NotImplementedError if unknown regression_type or optimization_type is given.
        """
        length = training_data.shape[-1]
        assert length == calibration_data.shape[-1]
        assert 0.0 < alpha < 1.0
        total_calibration_size = calibration_data.shape[0]
        assert total_calibration_size > self.calibration_size_for_lcp
        optimization_data = calibration_data[: self.calibration_size_for_lcp]
        calibration_data = calibration_data[self.calibration_size_for_lcp :]

        prediction_trajectory = regress(training_data, self.regression_type)

        if lower_bound:
            prediction_errors_on_optimization_data = (
                optimization_data - prediction_trajectory
            )
        else:
            prediction_errors_on_optimization_data = (
                prediction_trajectory - optimization_data
            )

        model = self._get_optimization_model(
            prediction_errors_on_optimization_data, alpha
        )
        model.optimize()
        weighting_coeffs = np.array(
            [v.x for v in filter(lambda v: "alphas" in v.varName, model.getVars())]
        )

        if lower_bound:
            calibration_scores = [
                np.max(
                    (calibration_trajectory - prediction_trajectory) * weighting_coeffs
                )
                for calibration_trajectory in calibration_data
            ]
        else:
            calibration_scores = [
                np.max(
                    (prediction_trajectory - calibration_trajectory) * weighting_coeffs
                )
                for calibration_trajectory in calibration_data
            ]
        band_width = np.quantile(calibration_scores, 1.0 - alpha)

        if lower_bound:
            bounding_trajectory = prediction_trajectory - band_width / weighting_coeffs
        else:
            bounding_trajectory = prediction_trajectory + band_width / weighting_coeffs
        return bounding_trajectory

    def _get_optimization_model(
        self, error_values: np.ndarray, alpha: float
    ) -> gp.Model:
        """Sets up the optimization problem in Gurobi. Reference:
        https://github.com/earnedkibbles58/timeParamCPScores/blob/master/code/gurobipyTutorial.py

        Args:
            error_values: an np.ndarray with shape (size, length) of prediction error values.
            alpha: The significance level. The prediction set is valid with probability 1 - alpha.

        Returns:
            A Gurobi model to be solved.
        """

        num_trajectories = len(error_values)
        length = len(error_values[0])

        model = gp.Model("optimization_problem")

        # declare variables we have
        q = model.addVar(lb=np.min(error_values), vtype=GRB.CONTINUOUS, name="q")
        alphas = model.addVars(length, lb=0, vtype=GRB.CONTINUOUS, name="alphas")
        Rs = model.addVars(num_trajectories, vtype=GRB.CONTINUOUS, name="Rs")
        es_plus = model.addVars(
            num_trajectories, lb=0, vtype=GRB.CONTINUOUS, name="es_plus"
        )
        es_minus = model.addVars(
            num_trajectories, lb=0, vtype=GRB.CONTINUOUS, name="es_minus"
        )
        us_plus = model.addVars(
            num_trajectories, lb=0, vtype=GRB.CONTINUOUS, name="us_plus"
        )
        us_minus = model.addVars(
            num_trajectories, lb=0, vtype=GRB.CONTINUOUS, name="us_minus"
        )
        v = model.addVars(
            num_trajectories, lb=-float("inf"), vtype=GRB.CONTINUOUS, name="v"
        )

        # create objective
        objective = gp.LinExpr(q)
        model.setObjective(objective, GRB.MINIMIZE)

        # Lowerbound constraints on Rs
        for i in range(num_trajectories):
            for t in range(length):
                model.addConstr(Rs[i] >= alphas[t] * error_values[i][t])

        # KKT constraints 1: Lagrangian stationarity
        q_gradient_constraint = gp.LinExpr()
        for i in range(num_trajectories):
            model.addConstr((1.0 - alpha) - us_plus[i] + v[i] == 0)
            model.addConstr(alpha - us_minus[i] - v[i] == 0)
            q_gradient_constraint += v[i]
        model.addConstr(q_gradient_constraint == 0)

        # KKT constraints 2: complementary slackness
        for i in range(num_trajectories):
            model.addConstr(us_plus[i] * es_plus[i] == 0)
            model.addConstr(us_minus[i] * es_minus[i] == 0)

        # KKT constraints 3: primal feasibility
        for i in range(num_trajectories):
            model.addConstr(es_plus[i] >= 0)
            model.addConstr(es_minus[i] >= 0)
            model.addConstr(es_plus[i] + q - es_minus[i] - Rs[i] == 0)

        # KKT constraints 4: dual feasibility
        for i in range(num_trajectories):
            model.addConstr(us_plus[i] >= 0)
            model.addConstr(us_minus[i] >= 0)

        # Normalization constraints on alphas
        m_constraint = gp.LinExpr()
        for t in range(length):
            m_constraint += alphas[t]
            model.addConstr(alphas[t] >= 0)
        model.addConstr(m_constraint == 1)

        if self.optimization_type == OptimizationType.KKT:
            # declare additional binary variables and their constraints
            b = model.addVars(num_trajectories, length, vtype=GRB.BINARY, name="b")
            M = np.max(error_values)
            for i in range(num_trajectories):
                b_constraint = gp.LinExpr()
                for t in range(length):
                    model.addConstr(
                        Rs[i] <= alphas[t] * error_values[i][t] + (1.0 - b[(i, t)]) * M
                    )
                    b_constraint += b[(i, t)]
                model.addConstr(b_constraint == 1)
        elif self.optimization_type == OptimizationType.KKTNoMaxUpperBound:
            model.params.NonConvex = 2
        else:
            raise (NotImplementedError)

        return model
