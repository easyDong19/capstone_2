import os

import networkx as nx
import pandas as pd

# 1년치 데이터 만들기
def load_dataset(kospi_data:pd.DataFrame,pickle_path:str,start_date:str,end_date:str) -> pd.DataFrame:
    if os.path.isfile(pickle_path):
        return pd.read_pickle(pickle_path)

    dataset = kospi_data.loc[start_date:end_date, :]

    dataset.to_pickle(pickle_path)

    return dataset


# 코스피 데이터 load
def load_kospi_data(pickle_path: str, excel_path: str) -> pd.DataFrame:
    if os.path.isfile(pickle_path):
        return pd.read_pickle(pickle_path)

    df = pd.read_excel(excel_path, sheet_name=0, header=None)

    kospi_data = df.iloc[14:].reset_index(drop=True)
    kospi_data.columns = df.iloc[8]

    kospi_data = kospi_data.rename(columns={kospi_data.columns[0]: 'date'})
    kospi_data['date'] = pd.to_datetime(kospi_data['date'], errors='coerce')

    kospi_data = kospi_data.set_index('date')
    kospi_data.index = pd.to_datetime(kospi_data.index, errors='coerce')

    kospi_data.columns.name = 'Code'
    kospi_data.index.name = 'Date'

    kospi_data.to_pickle(pickle_path)

    return kospi_data

import yfinance as yf

# todo : 코스피 데이터 불러와서 dataframe 만들고 timestamp 맞추고 pickle로 저장하는 로직 작성

def make_benchmark_kospi200(pickle_path:str,start_date:str, end_date:str)-> pd.DataFrame:
    if os.path.isfile(pickle_path):
        return pd.read_pickle(pickle_path)

    # 코스피 200 지수의 티커 코드
    KOSPI_TICKER = "^KS200"


    # 코스피 200 지수의 데이터를 가져오기
    ks= yf.download(KOSPI_TICKER, start=start_date, end=end_date, interval='1d')
    kospi_200 = ks['Adj Close']
    # 12월 29일 수정종가를 다음해 1월 1일 수정종가로 변경(DataGuide 정책)
    index_list = kospi_200.index.tolist()
    index_list[0] = pd.to_datetime(start_date)
    kospi_200.index = index_list

    kospi_200.to_pickle(pickle_path)
    return kospi_200

def make_cumulative_return(pickle_path:str,df:pd.DataFrame)-> pd.DataFrame:
    if os.path.isfile(pickle_path):
        return pd.read_pickle(pickle_path)

    df_return = df.pct_change().dropna()
    # 누적 수익률 계산
    df_cumulative_returns = (1 + df_return).cumprod() -1

    df_cumulative_returns.to_pickle(pickle_path)


    return df_cumulative_returns


import matplotlib.pyplot as plt

def compare_with_benchmark(data_A:pd.DataFrame, data_B:pd.DataFrame ,start_date:str, end_date:str) -> None:
    # 그래프 설정
    plt.figure(figsize=(10, 6))

    # 동일 비중 포트폴리오의 누적 수익률 시각화
    plt.plot(data_A.loc[start_date:end_date], label='Momentum- Equal-weighted Portfolio', color='blue')
    # 코스피200 누적 수익률 시각화
    plt.plot(data_B.loc[start_date:end_date], label='KOSPI 200', color='red')

    # 그래프 제목 및 축 설정
    plt.title('Momentum Portfolio vs KOSPI 200 Cumulative Returns\n(Up to 2024-10-04)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Returns', fontsize=12)
    # 날짜 라벨 회전
    plt.xticks(rotation=45)

    plt.legend()
    plt.grid(True)
    plt.savefig('./fig/Momentum_Portfolio_vs_KOSPI_200_Cumulative_Returns.png', format='png', dpi=300)

    plt.show()

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl

def visualize_edge_graph(G, partition):
    plt.figure(figsize=(10, 6))

    # 노드 배치 설정
    pos = nx.spring_layout(G)
    cmap = plt.get_cmap('viridis')

    # 노드 그리기
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=partition.keys(),
        node_size=240,
        cmap=cmap,
        node_color=list(partition.values())
    )

    # 엣지와 가중치 설정
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] * 3 for edge in edges]

    # 엣지 그리기 (컬러 매핑 추가)
    edges_collection = nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.7,
        width=weights,
        edge_color=weights,
        edge_cmap=plt.cm.Blues
    )

    # ScalarMappable 생성
    sm = mpl.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
    sm.set_array([])  # ScalarMappable에 배열 설정 필요 없음

    # Colorbar 추가 (sm을 mappable로 전달)
    cbar = plt.colorbar(sm, ax=plt.gca())  # 현재 Axes에 Colorbar 추가
    cbar.set_label('Edge Weight')  # 라벨 설정

    # 라벨 추가
    nx.draw_networkx_labels(G, pos, font_size=4)


    # 그래프 저장
    plt.savefig('../fig/visualize_edge_graph_2d.png', format='png', dpi=300)

    plt.show()
    plt.close()