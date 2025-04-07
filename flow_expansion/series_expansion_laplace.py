import math
import scipy.special as scsp
import scipy.stats as scs
import numpy as np
from typing import List
from flow_expansion import inn

#Function to compare the predicted samples with the true samples in one dimension (chosen by 'direction' varible)
def compare_predictions_samples(samples: np.ndarray, samples_pred: np.ndarray, NDIM: int, direction: int = 0) -> List:
    if NDIM > 1:
        samples = samples[:, direction]
        samples_pred = samples_pred[:, direction]
    pred_mean, pred_var = np.mean(samples_pred), np.var(samples_pred)
    pred_skew = scs.skew(samples_pred)
    pred_kurt = scs.kurtosis(samples_pred, fisher=False)
    print("Mean of predicted samples: ", pred_mean, "true mean: ", np.mean(samples))
    print("Variance of predicted samples: ", pred_var, "true variance: ", np.var(samples))
    print("Skewness of predicted samples: ", pred_skew, "true skewness: ", scs.skew(samples))
    print("Kurtosis of predicted samples: ", pred_kurt, "true kurtosis: ", scs.kurtosis(samples,fisher=False))
    cumulants_norm_true = [np.mean(samples), np.var(samples), scs.skew(samples), scs.kurtosis(samples,fisher=False)]
    return cumulants_norm_true

#Moment calculation (zeroth moment is integral over posterior)
def moment_calc(n_flow, func: List[int], order_series: int, order_moment: int, ndim: int, NETWORK) -> float:
    laplacian = inn.laplacian(n_flow, func, order_series, order_moment, ndim, NETWORK)
    sum_series = 0
    for k in range(order_series+1):
        sum_series += laplacian[k] *1/(2**k * math.factorial(k))
    return sum_series

#Calculate moments up to order_moment and store in list
def moments_calc(n_flow, func: List[int], order_series: int, order_moment: int, ndim: int, NETWORK) -> List:
    moments = [moment_calc(n_flow, func, order_series, i, ndim, NETWORK) for i in range(order_moment)]
    return moments

#Calculate nth central moment
def nth_central_moment(n: int, moments: List) -> float:
    assert len(moments) > n-1, "Not enough moments calculated"
    sum = 0
    for k in range(0, n+1):
        sum += scsp.binom(n, k) * moments[k]* moments[1]**(n-k) * (-1)**(n-k)
    return sum

#Calculate central moments up to order_moment and store in list
def central_moments_calc(n_flow, func: List[int], order_series: int, order_moment: int, ndim: int, NETWORK, moments: List = None) -> List:
    if moments is not None:
        moments = moments
    else:
        moments = moments_calc(n_flow, func, order_series, order_moment, ndim, NETWORK)
    nth_central_moments = [nth_central_moment(i, moments) for i in range(order_moment)]
    return nth_central_moments

#Calculate nth standardised moment
def nth_standardized_moment(n: int, central_moments: List) -> float:
    assert len(central_moments) > n-1, "Not enough cumulants calculated"

    if n == 2:
        return 1
    else:
        return central_moments[n] / central_moments[2]**(n/2)
    
#Calculate standardised moments
def standardized_moments_calc(n_flow, func: List[int], order_series: int, order_moment: int, ndim: int, NETWORK, nth_central_moments = None) -> List:
    if nth_central_moments is not None:
        nth_central_moments = nth_central_moments
    else:
        nth_central_moments = central_moments_calc(n_flow, func, order_series, order_moment, ndim, NETWORK)
    nth_standardized_moments = [nth_standardized_moment(i, nth_central_moments) for i in range(order_moment)]
    return nth_standardized_moments

#Calculate mean, variance, skewness and kurtosis for specific case defined by func (e.g. [1,0] for first dimension, [0,1] for second dimension, [1,1] for covariance)
def mean_var_skew_kurt(n_flow, func: List[int], order_series: int, order_moment: int, ndim: int, NETWORK) -> List:
    moments = moments_calc(n_flow, func, order_series, order_moment, ndim, NETWORK)
    nth_central_moments = central_moments_calc(n_flow, func, order_series, order_moment, ndim, NETWORK, moments)
    nth_standardized_moments = standardized_moments_calc(n_flow, func, order_series, order_moment, ndim, NETWORK, nth_central_moments)
    mean = moments[1]
    var = nth_central_moments[2]
    skew = nth_standardized_moments[3]
    kurt = nth_standardized_moments[4]
    return mean, var, skew, kurt

#Calculate mean, variance, skewness and kurtosis for 2D
def mean_var_skew_kurt_2d(n_flow, order_series: int, order_moment: int, ndim: int, NETWORK):
    assert ndim == 2, "This function works only in 2D."
    mean_vec = np.ones(2)
    skew_vec = np.ones(2)
    kurt_vec = np.ones(2)
    cov = np.eye(2)

    mean_vec[0], cov[0, 0], skew_vec[0], kurt_vec[0] = mean_var_skew_kurt(n_flow, [1, 0], order_series, order_moment, ndim, NETWORK)
    mean_vec[1], cov[1, 1], skew_vec[1], kurt_vec[1] = mean_var_skew_kurt(n_flow, [0, 1], order_series, order_moment, ndim, NETWORK)
    cov_off_diag, _, _, _ = mean_var_skew_kurt(n_flow, [1, 1], order_series, order_moment, ndim, NETWORK)
    cov[0, 1] = cov_off_diag - mean_vec[0] * mean_vec[1]
    cov[1, 0] = cov[0, 1]

    print("Via flow series:")
    print("Mean:       [{:6.2f} {:6.2f}]".format(mean_vec[0], mean_vec[1]))
    print("Covariance:")
    print("           [{:6.2f} {:6.2f}]".format(cov[0, 0], cov[0, 1]))
    print("           [{:6.2f} {:6.2f}]".format(cov[1, 0], cov[1, 1]))
    print("Skewness:   [{:6.2f} {:6.2f}]".format(skew_vec[0], skew_vec[1]))
    print("Kurtosis:   [{:6.2f} {:6.2f}]".format(kurt_vec[0], kurt_vec[1]))

    return mean_vec, cov, skew_vec, kurt_vec

#Mean, variance, skewness and kurtosis for 2d Gaussian toy model
def toy_model_mean_var_skew_kurt(PARAMS: List) -> None:
    skew_vec = np.zeros(2)
    kurt_vec = np.ones(2) * 3
    print("")
    print("Ground truth for 2D toy model:")
    print("Mean:       [{:6.2f} {:6.2f}]".format(PARAMS[0][0], PARAMS[0][1]))
    print("Covariance:")
    print("           [{:6.2f} {:6.2f}]".format(PARAMS[1][0][0], PARAMS[1][0][1]))
    print("           [{:6.2f} {:6.2f}]".format(PARAMS[1][1][0], PARAMS[1][1][1]))
    print("Skewness:   [{:6.2f} {:6.2f}]".format(skew_vec[0], skew_vec[1]))
    print("Kurtosis:   [{:6.2f} {:6.2f}]".format(kurt_vec[0], kurt_vec[1]))
    return None