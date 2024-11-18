import pandas as pd
from matplotlib import pyplot as plt


def plot_portfolio(title:str, start_date:str, end_date:str, save_url:str, data_df:pd.DataFrame) -> None:

    # 그래프 설정
    plt.figure(figsize=(10, 6))
    plt.plot(data_df.loc[start_date:end_date], label='portfolio cumulative return', color='blue')

    # 그래프 제목 및 축 설정
    plt.title(title, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Returns', fontsize=12)
    # 날짜 라벨 회전
    plt.xticks(rotation=45)

    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_url}/{start_date}~{end_date}.png', format='png', dpi=300)

    plt.show()

def compare_with_benchmark(title:str, portfolio_df:pd.DataFrame, kospi_df:pd.DataFrame ,start_date:str, end_date:str, save_url:str) -> None:
    # 그래프 설정
    plt.figure(figsize=(10, 6))

    plt.plot(portfolio_df.loc[start_date:end_date], label='portfolio', color='blue')
    plt.plot(kospi_df.loc[start_date:end_date], label='KOSPI 200', color='red')

    # 그래프 제목 및 축 설정
    plt.title(f'{title}({start_date}~{end_date})', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Returns', fontsize=12)
    # 날짜 라벨 회전
    plt.xticks(rotation=45)

    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_url}/compare_{start_date}~{end_date}.png', format='png', dpi=300)

    plt.show()