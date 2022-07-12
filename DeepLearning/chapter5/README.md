摘自《深度学习》第五章
Mitchell : "对于某类任务T和性能度量P，一个计算级程序被认为可以从经验E中学习是指，通过经验E改进后，它在任务T上由性能度量P衡量的性能有所提升。" 
# 一、常见的机器学习任务（T）
## 1、分类
计算机程序需要指定某些输入属于k类中的哪一类。为了完成这个任务，学习算法通常会返回一个函数
$$
f:R^n \rightarrow \{1,..., k\}
$$
还有一些分类问题输出的是不同类别的概率分布，比如对象识别中输入是图片，输出是表示图片物体的数字码。
## 2、输入缺失分类
当一些输入可能丢失时，学习算法必须学习一组函数，而不是单个分类函数。每个函数对应着分类具有不同缺失输入子集x。
## 3、回归
计算机程序对给定输入预测数值。为了解决这个任务，学习算法需要输出函数
$$
f:R^n \rightarrow R
$$
比如预测证券未来的价格。
## 4、转录
计算机学习系统观测一些相对非结构化表示的数据，并转录信息为离散的文本形式。比如光学字符识别要求根据文本图片返回文字序列。
## 5、机器翻译
输入时一种语言的符号序列，计算机程序将其转化为另一种语言的符号序列。比如自然语言翻译。
## 6、结构化输出
结构化输出任务的输出是向量或者其它包含多个值的数据结构，并且构成输出的这些不同元素间具有重要关系。比如语法分析，映射自然语言句子到语法结构树，并标记树的节点为动词、名词、副词等。
## 7、异常检测
计算机程序在一组事件或对象中筛选，并标记不正常或非典型的个体。
## 8、合成和采样
机器学习程序生成一些和训练数据相似的新样本。例如视频游戏可以自动生成大型物体或风景的纹理。
## 9、缺失值填补
## 10、去噪
## 11、密度估计或概率质量函数估计
算法需要学习观测到的数据的结构，知道什么情况下样本聚集出现，什么情况不太可能出现，隐式地捕获概率分布的结构。
# 二、性能度量（P）
准确率
错误率
使用测试集（test set）数据来评估系统性能。
# 三、经验（E）
根据不同经验，机器学习算法可以大致分类为无监督算法（unsupervised）和监督(supervised)算法
**无监督学习算法**训练含有很多特征的数据集，然后学习出这个数据集上有用的结构性质。无监督学习涉及观察随机变量x的好几个样本，试图显式或隐式地学习出概率分布p(x)，或者是该分布一些有意思的性质。
**监督学习算法**训练含有很多特征的数据集，不过数据集中的样本都由一个标签或目标。监督学习包含视察随机向量x及其相关联的值或向量y，然后从x预测y，通常是估计$$$p(y|x)$$$。
# 四、一些名词
* **泛化**：在先前未预测到的输入上表现良好的能力
* **训练误差**：在训练集上计算的度量误差，目标是降低寻你连误差。
* **泛化误差**
* **独立同分布假设**：每个数据集中的样本都是相互独立的；并且训练集和测试集是同分布的，采样自相同的分布。
* **数据生成分布**：共享的潜在分布，记作$$$p_{data}$$$
* **欠拟合**：指模型不能够在训练集上获得足够低的误差
* **过拟合**：指训练误差和测试误差之间的差距太大
* **容量**：指模型拟合个股中函数的能力。容量低的模型可能很难拟合训练集，容量高的模型可能会过拟合。通过调整模型的容量，可以控制模型是否偏向过拟合或欠拟合。控制容量的方法：假设空间（学习算法可以选择为解决方案的函数集）；
* **表示容量**：学习算法可以从哪些函数中选择函数
* **有效容量**：有效容量可能小于模型族的表示容量
* **Vapnik-Chevonenkis维度（VC维）**：该分类器能够分类的训练样本的最大数目。假设存在m个不同x点的训练集，分类器可以任意地标记该m个不同的x点，VC维被定义维m的最大可能值。
* **权重衰减**
* **超参数**：在开始机器学习之前，人为设置好的参数。比如聚类中类的个数、模型的学习率、深层神经网络隐藏层数。
  **模型参数**：通过训练得到的参数数据。
* **训练集**
* **验证集**：用于挑选超参数的数据子集。（测试样本不能以人任何形式参与到模型的选择中，包括设定超参数。基于这个原因，测试集中的样本不能用于验证集。因此，我们总是从训练数据中构建验证集。特别地，我们将训练数据分为两个不相交的子集，其中一个用于学习参数，另一个作为验证集，用于估计训练中或训练后的泛化误差，更新超参数。通常80%的训练数据用于训练，20%用于验证。）
* **交叉验证**：
* **k-折交叉验证算法**：将数据集分为k个不重合的子集，测试误差可以估计为k次计算后的平均测试误差