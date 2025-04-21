import dgl
from dgl.data import CoraGraphDataset, PubmedGraphDataset, CiteseerGraphDataset

# 데이터셋 로드 함수
def load_dataset(name):
    if name == "Cora":
        return CoraGraphDataset()
    elif name == "Citeseer":
        return CiteseerGraphDataset()
    elif name == "Pubmed":
        return PubmedGraphDataset()
    else:
        raise ValueError(f"Unknown dataset: {name}")

# 데이터셋 전처리 함수
def preprocess_data(dataset):
    graph = dataset[0]

    # 원래 그래프의 노드 수와 엣지 수 출력
    num_nodes_original = graph.number_of_nodes()
    num_edges_original = graph.number_of_edges()
    print(f"Original graph - Number of nodes: {num_nodes_original}")
    print(f"Original graph - Number of edges: {num_edges_original}")

    # 그래프의 방향성 제거 (중복 엣지 제거)
    graph = dgl.to_bidirected(graph, copy_ndata=True)

    # 양방향 엣지를 단일 엣지로 변환
    src, dst = graph.edges()
    unique_edges = set()
    for u, v in zip(src.tolist(), dst.tolist()):
        if (v, u) not in unique_edges:
            unique_edges.add((u, v))
    src, dst = zip(*unique_edges)
    graph = dgl.graph((src, dst), num_nodes=graph.number_of_nodes())

    # 노드 데이터 복사
    graph.ndata['feat'] = dataset[0].ndata['feat']
    graph.ndata['label'] = dataset[0].ndata['label']
    graph.ndata['train_mask'] = dataset[0].ndata['train_mask']
    graph.ndata['val_mask'] = dataset[0].ndata['val_mask']
    graph.ndata['test_mask'] = dataset[0].ndata['test_mask']

    # 방향성 제거 후의 그래프의 노드 수와 엣지 수 출력
    num_nodes_undirected = graph.number_of_nodes()
    num_edges_undirected = graph.number_of_edges()
    print(f"Undirected graph - Number of nodes: {num_nodes_undirected}")
    print(f"Undirected graph - Number of edges: {num_edges_undirected}")

    features = graph.ndata['feat']
    labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']

    return graph, features, labels, train_mask, val_mask, test_mask