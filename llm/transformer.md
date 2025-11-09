- 

[Transformer模型详解（图解最完整版） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/338817680)

## input

分为编码器Encoder、解码器Decoder两个部分。

输入是input embedding 和 position embedding。

transformer不像RNN，不知道顺序，需要用 PE来让模型知道input_i的位置信息。

将 input embedding 和 position embedding相加就得到了transformer的input。



## self-attention

self-atten在计算时需要Q,K,V，self-atten的输入是input或者上一个encoder block的输出。

Q,K,V 是通过self-atten的输入进行线性变换得到。

### Q,K,V

进入self-atten的输入假设是X，使用线性变换矩阵$W_q,W_k,W_v$计算得到Q,K,V。

> X,Q,K,V 的每一行都是一个单词。

$QK^T$得到（n，n）的矩阵，这个矩阵可以表示单词之间的atten强度。

通过$\sqrt{d_k}$进行缩放，防止$QK^T$的内积过大。

之后使用softmax计算每个单词对其他单词的atten系数。经softmax之后，每一行之和是1.

最终输出的$Z$是 再乘以$V$得到单词j与所有单词i的值$V_i$根据atten系数的比例加在一起。

## mult-head-attention

多头注意力是多个self-atten组合而成的。

将$X$（即input）分别传递到h个不同的self-atten，计算得到h个$Z_{sub}$。

h个$Z_{sub}$进行拼接得到最终的$Z$。



## encoder

encoder由N个encoder block组成。每个block包括mult-head-attention, add & norm, feed forward, add & norm 组成。

- add & norm：
  - add：残差连接，用于解决多层网络训练的问题
  - norm：layer norm，层归一化，将每一层的输入都转为均值方程一样的，可以加速收敛
- feed forward：两层的全连接，第一层激活函数relu，第二层不使用激活函数

encoder block接收一个输入矩阵$X(n,d)$，输出矩阵$O(n,d)$。

第一个block输入是句子单词的表示向量矩阵，后续block是前一个block的输出，最后一个block输出的矩阵就是编码信息矩阵$C$。

## decoder

decoder也是由N个decoder block组成，基本与encoder相似。包含：

- 两个mulit-head-attention
  - 第一个采用masked操作
  - 第二个的K,V矩阵使用encoder的编码信息矩阵$C$进行计算，而Q使用上一个decoder block输出计算
- 最后有一个softmax层计算下一个单词的概率



## 1-mha

第一个mha采用masked操作，是因为预测是顺序预测的，通过masked可以防止第i个单词知道第i之后的单词信息。

decoder在训练过程中使用teacher forcing＋并行化训练

> teacher forcing: 统一右移一个。例如" I like cat" -> "<begin> I like cat"

decoder block 的输入跟masked矩阵。masked矩阵包含第i个单词可以查看的前i-1个单词的信息。

之后操作跟self-attenyi'y，通过输入矩阵$X$计算得到Q,K,V矩阵...

在得到$QK^T$后要进行softmax时，在使用softmax之前用mask矩阵遮住每一个单词之后的信息。

使用$mask QK^T$与矩阵$V$相乘得到输出$Z_{sub}$。

通过拼接$Z_{sub}$得到$Z$。

## 2-mha

第二个mha主要区别在于self-atten的K,V矩阵不是前面的decoder block计算的，而是encoder的编码信息矩阵$C$计算的。



block最优一个softmax预测输出的单词



|            | 训练                    | 预测                           |
| ---------- | ----------------------- | ------------------------------ |
| 核心       | 并行                    | 自回归                         |
| 解码器输入 | teacher forcing（右移） | 上一步生成的序列               |
| 并行性     | 高                      | 低                             |
| 掩码机制   | 防止偷看答案            | 机制自然（不需要mask           |
| 效率       | 比RNN快                 | 生成速度缓慢，与序列长度成正比 |

