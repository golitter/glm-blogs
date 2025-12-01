[ XGBoost的原理、公式推导、Python实现和应用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/162001079)

XGBoost（eXtreme Gradient Boosting，极致梯度提升），是一种基于GBDT的算法或工程实现。

XGBoost与GBDT相同，但是做了些优化，**二阶导数使损失函数更精准；正则项避免过拟合；block存储可以并行计算**等。

> bagging：并行计算，结果综合起来。
>
> boosting：串行计算，针对前面的结果进行计算。





## GBDT

GBDT（Gradient Boosting Decision Tree，梯度提升决策树）是一种基于boosting集成思想的模型。**每次迭代都学习一颗CART树来拟合之前$t-1$棵树的预测结果与训练样本真实值的残差**。

> CART（Classification and Regression Tree，分类与回归树），是一种既可以用于分类，也可以用于回归的决策树。





高维稀疏特征，GBDT较差。

> 线性模型的正则项是对权重的惩罚，$w_1$很大，惩罚就很大，进一步压缩$w_1$的值使其不那么大。
>
> 树模型的惩罚项通常为叶子节点数和深度，一个节点最终产生的惩罚项及其小。



[XGBoost面试题 | Daily Interview - 面试必看 (datawhalechina.github.io)](https://datawhalechina.github.io/daily-interview/04-ai-algorithms/machine-learning/ensemble-learning/XGBoost.html)