
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms
import xgboost as xgb
import random


# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建适应度类型与个体类型，取适应度最大值
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# 定义参数范围
PARAMS = {"max_depth": [3, 10],
          "learning_rate": [0.01, 0.3],
          "n_estimators": [50, 200],
          "gamma": [0, 0.5],

        #   "min_child_weight": [1, 5],
        #   "subsample": [0.5, 1],
        #   "colsample_bytree": [0.5, 1]
          }
# 个体和种群随机数生成
def uniform(params):
    return [random.uniform(l, u) for l, u in params]

# 注册个体生成函数
toolbox.register("attr_float", uniform, PARAMS.values())
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 定义评价函数
def evaluate(individual):
    params = {key: max(1, int(val)) if key in ['max_depth', 'n_estimators'] 
              else max(0, val) if key in ['learning_rate', 'min_split_loss']
              else round(val, 5) for key, val in zip(PARAMS.keys(), individual)}
    params['min_split_loss'] = 0
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy,

# 注册遗传算法操作
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# 定义种群数量和进化代数
pop = toolbox.population(n=5)
NGEN=10

# 执行遗传算法
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


