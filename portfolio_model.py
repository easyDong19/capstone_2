import pandas as pd
from scipy.optimize import minimize
import numpy as np

"""
    평균-분산 최적화 모형을 사용하여 주어진 목표 수익률을 달성하는 포트폴리오의 비중을 계산합니다.

    Parameters:
    mean_returns (array): 자산의 기대 수익률
    cov_matrix (array): 자산 수익률의 공분산 행렬
    target_return (float): 목표 수익률

    Returns:
    array: 최적화된 포트폴리오 비중
"""


def mean_variance_portfolio(mean_returns, cov_matrix, target_return=0.10):
    num_assets = len(mean_returns)

    # 포트폴리오 분산 계산 함수
    def portfolio_variance(weights, cov_matrix):
        return weights.T @ cov_matrix @ weights

    # 포트폴리오 수익률 계산 함수
    def portfolio_return(weights, mean_returns):
        return weights.T @ mean_returns

    # 제약 조건: 비중의 합은 1이어야 함
    def constraint_sum_of_weights(weights):
        return np.sum(weights) - 1

    # 제약 조건: 포트폴리오 수익률은 목표 수익률 이상이어야 함
    def constraint_min_return(weights, mean_returns, target_return):
        return portfolio_return(weights, mean_returns) - target_return

    # 초기값 설정 (균등하게 분배)
    init_guess = np.array([1.0 / num_assets] * num_assets)
    # 제약 조건을 딕셔너리 형태로 정의
    constraints = (
        {'type': 'eq', 'fun': constraint_sum_of_weights},
        {'type': 'ineq', 'fun': lambda weights: constraint_min_return(weights, mean_returns, target_return)}
    )
    # 경계 조건 설정 (각 비중은 0 이상)
    bounds = tuple((0, 1) for _ in range(num_assets))
    # 최적화 실행
    optimal_portfolio = minimize(
        portfolio_variance,
        init_guess,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return optimal_portfolio.x


def equal_weighting_portfolio(kospi_data:pd.DataFrame, selected_data:pd.DataFrame,start_date:str, end_date:str)->pd.DataFrame:
    # data에서 start_date에서 end_date까지 종가 데이터를 추출
    momentum_tickers = selected_data.columns.tolist()
    data_momentum = kospi_data[momentum_tickers].loc[start_date:end_date]

    # 수익률 계산 : 수정 종가 일별 변화율
    momentum_returns = data_momentum.pct_change().dropna()

    # 동일 비중으로 투자 했을 때 포트폴리오 수익
    daily_returns = momentum_returns.mean(axis=1)

    # 누적 수익률
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    final_cum_return = cumulative_returns.iloc[-1]
    print(f'{start_date}부터 {end_date} 까지 동일 비중 투자 누적 수익률 : {final_cum_return * 100:.2f}%')

    return daily_returns, cumulative_returns