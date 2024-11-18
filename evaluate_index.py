# 샤프지수
import numpy as np
import pandas as pd


# risk_free_rate는 KOFR에서 가져온 연간 수익률 0.03296
def calculate_sharpe(daily_returns, risk_free_rate=0.03296, trading_days=252):
    # 일간 수익률의 평균과 표준 편차 계산
    mean_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()

    # 연간화된 수익률과 표준 편차 계산
    annualized_return = mean_daily_return * trading_days
    annualized_std = std_daily_return * np.sqrt(trading_days)

    # 일간 무위험 수익률 (연간 무위험 수익률을 일간으로 나누기)
    daily_risk_free_rate = risk_free_rate / trading_days

    # 샤프 지수 계산
    sharpe_ratio = (annualized_return - daily_risk_free_rate * trading_days) / annualized_std

    return sharpe_ratio
# 소프타노 지수

def calculate_sortino(daily_returns, risk_free_rate=0.03296, trading_days=252, target_return=0):
    # 일간 수익률의 평균 계산
    mean_daily_return = daily_returns.mean()

    # 연간화된 수익률 계산
    annualized_return = mean_daily_return * trading_days

    # 일간 무위험 수익률 (연간 무위험 수익률을 일간으로 나누기)
    daily_risk_free_rate = risk_free_rate / trading_days

    # 하방 표준 편차 계산 (목표 수익률보다 낮은 수익만 고려)
    downside_returns = daily_returns[daily_returns < target_return]
    downside_deviation = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(trading_days)

    # 소르티노 지수 계산
    sortino_ratio = (annualized_return - daily_risk_free_rate * trading_days) / downside_deviation

    return sortino_ratio