
import pandas as pd
import copy
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms
import xgboost as xgb
import pathlib as path
import random


SEED = 50
random.seed(SEED)

# 划分训练集和测试集
#倒数三行不读入描述符
columns_backward=-3
p_tr = path.PureWindowsPath(r"D:\Users\79864\Desktop\毕业论文——BTK\描述符\500_子结构划分\不含子结构\som划分\筛选总表02增加负样本不含子结构_MACCSKeysFingerprint_SOM_tr描述符缩减至70.csv")
p_te = path.PureWindowsPath(r"D:\Users\79864\Desktop\毕业论文——BTK\描述符\500_子结构划分\不含子结构\som划分\筛选总表02增加负样本不含子结构_MACCSKeysFingerprint_SOM_te描述符缩减至70.csv")

train_data_1 = pd.read_csv(p_tr,encoding="gbk")
test_data_1 = pd.read_csv(p_te,encoding="gbk")
train_data_copy = copy.deepcopy(train_data_1)
#对输入的训练集数据进行copy
test_data_copy = copy.deepcopy(test_data_1)
#对输入的测试集数据进行copy
train_data_random = train_data_copy.sample(frac=1,random_state=100)
#将训练集进行乱序，种子为100
test_data_random = test_data_copy.sample(frac=1,random_state=100)
#将测试集进行乱序，种子为100
train_x_index = train_data_random.columns[:columns_backward]
#提取训练集数据列名
X_train = train_data_random[train_x_index]
#从训练集中提取数据
y_train = train_data_random['y']
#从训练集中提取活性
test_x_index = test_data_random.columns[:columns_backward]
#提取测试集数据列名
X_test = test_data_random[test_x_index]
#从测试集中提取数据
y_test = test_data_random['y']


# 创建适应度类型与个体类型，取适应度最大值
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# 定义参数范围
PARAMS = {"max_depth": [5, 15],
          "learning_rate": [0.01, 0.5],
          "n_estimators": [50, 250],
          "gamma": [0.01, 0.5],

        #   "min_child_weight": [1, 5],
        #   "subsample": [0.5, 1],
        #   "colsample_bytree": [0.5, 1]
          }
# 个体和种群随机数生成
def uniform(params):
    return [random.uniform(l, u) for l, u in params]

# toolbox.register("attr_bool", np.random.randint, 0, 2) # 特征选择：0表示不选，1表示选

# 注册个体生成函数
toolbox.register("attr_float", uniform, PARAMS.values())
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 定义评价函数
def evaluate(individual):
    params = {key: max(1, int(val)) if key in ['max_depth', 'n_estimators'] 
              else max(0, val) if key in ['learning_rate', 'gamma']
              else round(val, 5) for key, val in zip(PARAMS.keys(), individual)}
    params['min_split_loss'] = 0
    print("params:", params)
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy,

# 注册遗传算法操作，评估函数、交叉函数、变异函数和选择函数。
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
# 对染色体进行高斯（正态）分布的突变，分别表示高斯分布的均值、标准差和变异概率。
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# 定义种群数量和进化代数
pop = toolbox.population(n=50)
NGEN=100

# 执行遗传算法，algorithms.varAnd() 函数集成了交叉和突变操作，toolbox.map() 函数用于并行计算所有个体的适应度
for gen in range(NGEN):
    pop = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, pop)
    for fit, ind in zip(fits, pop):
        ind.fitness.values = fit
    pop = toolbox.select(pop, k=len(pop))

# 打印最优解
best_ind = tools.selBest(pop, 1)[0]
best_params = dict(zip(PARAMS.keys(), best_ind))
print("Best individual is %s, with fitness of %s" % (best_params, best_ind.fitness.values))


