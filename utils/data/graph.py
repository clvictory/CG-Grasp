import networkx as nx  
import numpy as np  
  
# 定义数据点  
data = [  
    {'id': 1, 'x': 1, 'y': 2, 'angle': 30, 'width': 4},  
    {'id': 2, 'x': 5, 'y': 6, 'angle': 35, 'width': 7},
    {'id': 3, 'x': 101, 'y': 300, 'angle': 50, 'width': 1},
    # ... 更多数据点  
]  
  
# 构建图的函数  
def build_graph(data, angle_e=1, width_e=1, dis_e=1):  
    G = nx.Graph()  # 创建一个无向图  
    node_positions = {}  # 存储节点的位置和角度信息  
      
    for point in data:  
        x, y = point['x'], point['y']  
        angle = point['angle']  
        width = point['width']  
        node_positions[point['id']] = (x, y, angle, width)  # 将数据点存储为节点信息，使用id作为键  
        G.add_node(point['id'])  # 添加节点到图中，使用id作为节点  
      
    for point1 in data:  
        for point2 in data:  
            if point1 == point2:  # 跳过与自身比较的情况  
                continue  
            x1, y1, angle1, width1 = node_positions[point1['id']]  
            x2, y2, angle2, width2 = node_positions[point2['id']]  
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)  # 计算中心点间的距离  
            if dist < dis_e:  # 判断距离是否小于10  
                angle_diff = abs(angle1 - angle2)  # 判断角度差是否小于10  
                width_diff = abs(width1 - width2)  # 判断宽度差是否小于10  
                if angle_diff < angle_e and width_diff < width_e:  # 如果满足条件，添加无向边  
                    G.add_edge(point1['id'], point2['id'])  # 使用id作为边的两个节点  
      
    return G  # 返回构建的图  

def connected_graphs(data, angle_e=np.pi / 10, width_e=18, dis_e=35):
    # 构建图并找到所有连通分量
    print(len(data)) 
    G = build_graph(data, angle_e=angle_e, width_e=width_e, dis_e=dis_e)  
    connected_components = list(nx.connected_components(G))  # 使用networkx库找到所有连通分量  
    num_connected_components = len(connected_components)  # 获取连通分量的数量  
    print("Number of connected components:", num_connected_components)
    return connected_components