# RS知识框架

## 0. 前期任务要求

> ①该方向的已有研究（请附参考文献若干，和中文文献简介）
>
> ②该方向可以应用的推荐算法
>
> *③该方向可以使推荐算法贴合的若干特性（可选）
>
> ④实验数据（需要获取哪些数据，来进行实验和有效性评估；请一并附上准备使用的数据集的链接，例如：http://www.movielens.org/）
>
> ⑤研究的可行性评估（是否能获取足够实验的数据集，算法是否有网络教程或开源代码或论文中较为详细的介绍）
>
> *⑥……值得讨论的其他内容（例如该方向的创新点）（可选）

## 1. 实验材料

### 1.1 RS经典算法

| 序号 | 简称 | 提出时间 | 开源代码 |
| ---- | ---- | -------- | -------- |
| 1    | FM   | 2010     | 有       |
|      |      |          |          |
|      |      |          |          |

### 1.2 数据集

#### 1.2.1 最常见数据集

MovieLens、Yahoo

除非话题需要，否则，可以使用上述两种经典数据集，对算法的有效性进行评估。

#### 1.2.2 话题有可能需要的其他数据集

1. 旅行行为数据集 https://sites.google.com/site/yangdingqi/home/foursquare-dataset

   用于兴趣点推荐（POI）

2. 广告点击率Criteo数据集 https://www.kaggle.com/c/criteo-display-ad-challenge/data

3. 新闻Adressa数据集 https://reclab.idi.ntnu.no/dataset

   用于基于因果推断的推荐（Casual Inference）

### 1.3 兴趣点*

1. 反事实推荐
2. 推荐系统攻击（黑盒攻击子问题）
3. POI
4. EE问题
5. 序列推荐

### 1.4 论文分类

- **按任务划分**

- - **协同过滤 (Collaborative Filtering)**

    推荐系统最经典的算法之一，协同过滤通过挖掘用户和物品的交互信息，将相似的物品推荐给类似的用户。

    最新的针对协同过滤的研究依然在探索如何捕获交互中的相似性信息，包括建模更高阶的交互，设计更复杂的度量空间等。

  - **==序列推荐 (Sequential/Session-based Recommendation)==**

    序列推荐也是十分热门的研究方向。通过建模用户的**历史交互序列**来刻画用户偏好。

    最新热点在于与图神经网络相结合的方向。

  - **点击率预测 (CTR/CVR Prediction)**

    这类工作旨在对用户是否点击展示的商品/广告做出二分类预测。

    近期研究热点包括高阶特征交互，设计高阶向量交互运算来构造显式的特征交互；图表示学习，异构特征图预训练增强特征表示；网络结构化搜索，采用强化学习或进化算法优化特征工程搜索最佳网络结构；多任务学习，通过**点击率与转化率联合建模**实现多个任务间互相增益。

  - **知识感知的推荐 (Knowledge-aware Recommendation)**

    利用结构化的信息辅助推荐。

    知识图谱KG、GNN方向。

  - **对话推荐系统 (Conversational Recommender System)**

    对话推荐系统通过**多轮对话**，建模用户的动态偏好从而完成推荐。

    目前的主流技术包括强化学习对话策略；知识增强对话推荐系统；对话推荐中的可控增强；以及统一推荐、对话框架等。

  - **社交推荐 (Social Recommendation)**

  - **新闻推荐 (News Recommendation)**

  - **音乐推荐 (Music Recommendation)**

  - **文本感知的推荐 (Text-aware Recommendation)**

    文本感知：与推荐的评论相结合。

  - **==兴趣点推荐 (POI Recommendation)==**

    为用户推荐可能感兴趣的**地理位置**的任务称为兴趣点推荐。

    由于区域可能的兴趣点 (POI)搜索空间很大，因此学习用户接下来将访问哪个兴趣点对于个性化推荐系统来说十分具有挑战性，其中一个重要挑战是用户兴趣点矩阵的稀疏性。

    下一个兴趣点 (POI) 推荐在许多基于位置的应用程序中起着至关重要的作用，因为它为用户提供关于有吸引力的目的地的个性化建议。

    下一个兴趣点 (POI)推荐是基于位置的社交网络 (LBSN)中最重要的任务之一，旨在通过从用户的历史行为中发现偏好，向用户提出下一个合适位置的个性化推荐。

  - **在线推荐 (Online Recommendation)**

    在线推荐。例如实时短视频推荐。

  - **组推荐 (Group Recommendation)**

  - **多任务/多行为/跨域推荐 (Multi-task/Multi-behavior/Cross-domain Recommendation)**

    跨领域推荐。

  - **多模态推荐 (Multimodal Recommendation)**

  - **其他任务 (Other Tasks)**

    Multi-hop Reading on Memory Neural Network with Selective Coverage for Medication Recommendation. CIKM 2021【药物推荐】

    User Recommendation in Social Metaverse with VR. CIKM 2022 【VR 的用户推荐】

    CAPTOR: A Crowd-Aware Pre-Travel Recommender System for Out-of-Town Users. SIGIR 2022 【为乡村用户提供的旅游推荐】

    PERD: Personalized Emoji Recommendation with Dynamic User Preference. SIGIR 2022 【short paper，个性化表情推荐】

    ……

- **按话题划分**

- - **推荐系统去偏 (Debias in Recommender System)**

    推荐系统去偏旨在消除推荐数据/模型中由偏差带来的不良影响。其中偏差包括比较常见的流行度偏差 (popularity bias)，选择偏差 (selection bias)，曝光偏差 (exposure bias)，位置偏差 (position bias) 以及其他各种偏差。

  - **推荐系统公平性 (Fairness in Recommender System)**

    推荐系统的公平性**指在个性化推荐时进一步考虑推荐的公平性**。

    目前的主流技术包括①引入公平性指标，在训练时同时优化准确性和公平性的指标；②因果推断去偏，采用因果推断理论缓解偏差问题，从而提升公平性；③对抗学习方法，采用 GAN 等对抗训练方法，在训练中引入公平性的权衡；④推荐结果重排序，在已有推荐结果上重新排序， 兼顾准确性和公平性等。

  - **==推荐系统中的攻击 (Attack in Recommender System)==**

    **基于协同过滤CF：**

    1）随机攻击

    向系统中**注入随机数值**，从而达到对推荐系统的干扰的目的。比如，随机给全部item随机评分。

    2）均值攻击

    在随机攻击的基础之上，利用评分均值，**构造更加“像”的注入数据**。还以评分为例，以平均值为参数，做正态分布的评分，将评分按特定方向刷分。

    3）造势攻击

    比均值攻击更加高明的地方在于，除了有对目标物品的高评分（或者低平分）之外，还包含了很多热门物品的高评分，这样的目的在于：很多用户的使用记录也包含了这些热门物品，注入的数据更容易和用户**形成“近邻”关系**，从而更容易被推荐系统采用，更容易影响到最终的用户。

    4）局部攻击

    比造势攻击更加高明的地方在于，局部攻击能够**识别出特定的用户群体**，并据此发生攻击。

    5）针对性的打压攻击

    与抬高某个物品的评分的目的不同，打压攻击是**为了要降低某个物品的评分**。方法就是上面的方法，反其道而行之。通常，打压攻击更加容易得手——学术界还没有解释出为什么会存在这种不对称性。

    6） 点击流攻击和隐式反馈

    手段是**模拟用户在网页上的操作**，来达到注入数据的目的。

    Data Poisoning Attack against Recommender System Using Incomplete and Perturbed Data. KDD 2021 【不完整及扰动数据攻击推荐系统】

    **基于其他算法：**

    1）知识图谱：

    Knowledge-enhanced Black-box Attacks for Recommendations. KDD 2022 【知识图谱增强】

    2）序列推荐：

    Black-Box Attacks on Sequential Recommenders via Data-Free Model Extraction. RecSys 2021 【序列推荐的黑盒攻击】

    Defending Substitution-Based Profile Pollution Attacks on Sequential Recommenders. RecSys 2022 【序列推荐中基于替代的对抗性攻击算法】

    3）联邦推荐：

    FedAttack: Effective and Covert Poisoning Attack on Federated Recommendation via Hard Sampling. KDD 2022 【联邦推荐的对抗攻击】

  - **推荐系统可解释性 (Explanation in Recommender System)**

    可解释推荐要求推荐系统不仅要提供高质量的推荐，还要对结果做出解释。

    目前的主流技术包括基于强化学习，将推荐模型作为环境的一部分，并对生成的解释做出奖励；基于反事实，因果推断，基于因果推断框架量化推荐理由，分析关键因素；基于知识图谱，利用知识的语义捕捉实体关系，提升解释质量；基于预训练语言模型，利用预训练语言模型生成个性化的高质量解释。

  - **推荐系统的长尾/冷启动问题 (Long-tail/Cold-start in Recommendation)**

    冷启动问题。

    目前比较流行基于meta-learning,transfer-learning的方法，还有很多。

  - **推荐系统多样性 (Diversity in Recommendation)**

    多样化的推荐结果应该尽可能包含不同的商品来满足用户的多种兴趣。

    目前的主流技术包括基于数据增广；基于动态的图神经网络；多目标联合优化；强化学习等。

  - **推荐系统去噪 (Denoising in Recommendation)**

  - **推荐的隐私保护 (Privacy Protection in Recommendation)**

    推荐系统的隐私保护旨在保护用户隐私的前提下进行推荐。

    目前的研究热点包括联邦学习加密优化，结合密码学方法或模糊化方法,进一步优化安全性；联邦学习效率优化，基于采样算法研究加快收敛速度、减少通讯次数等。

  - **推荐系统的评价和实验分析 (Evaluation of Recommender System)**

  - **==其他话题 (Other Topics)==（EE问题）**

- **按技术划分**

- - **推荐系统预训练 (Pre-training in Recommender System)**
  - **推荐中的强化学习 (Reinforcement Learning in Recommendation)**
  - **推荐中的知识蒸馏 (Knowledge Distillation in Recommendation)**
  - **推荐中的联邦学习 (Federated Learning in Recommendation)**
  - **推荐中的图神经网络 (GNN in Recommendation)**
  - **基于对比学习的推荐 (Contrastive Learning based Recommendation)**
  - **基于对抗学习的推荐 (Adversarial Learning based Recommendation)**
  - **基于自动编码器的推荐 (Autoencoder based Recommendation)**
  - **基于元学习的推荐 (Meta Learning-based Recommendation)**
  - **基于自动机器学习的推荐 (AutoML-based Recommendation)**
  - **==基于因果推断的推荐 (Casual Inference/Counterfactual)==**
  - **其他技术 (Other Techniques)**

#### 1.5 参考专栏

1. 800 篇顶会论文纵览推荐系统的前沿进展 https://zhuanlan.zhihu.com/p/585945102

### 2. 算法概述

### 2.1 FM

全称：Factorization Machines，因子分解机，2010。

论文：https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle_et_al2011-Context_Aware.pdf

代码：https://github.com/jc-LeeHub/Recommend-System-tf2.0/tree/master/FM

代码结构：

```python
├── utils.py   
│   ├── create_criteo_dataset  # Criteo 数据预处理，返回划分好的训练集与验证集
├── model.py  
│   ├── FM_layer # FM 层的定义
│   ├── FM       # FM 模型的搭建
├── train.py 
│   ├── main     # 将处理好的数据输入 FM 模型进行训练，并评估结果
```

数据集：Criteo数据集，用于广告点击率预估任务 https://www.kaggle.com/c/criteo-display-ad-challenge/dataBlack-Box 

### 3. 论文分析

#### 3.1 反事实推荐

##### 3.1.1 用反事实推断方法缓解标题党内容对推荐系统的影响

###### 3.1.1.1 该方向的已有研究

（1）论文标题：Click can be Cheating: Counterfactual Recommendation for Mitigating Clickbait Issue

（2）发表年份：2021

（3）论文链接：https://dl.acm.org/doi/pdf/10.1145/3404835.3462962

（4）代码开源：https://github.com/WenjieWWJ/Clickbait

（5）研究内容：

①在不引入用户反馈信息的情况下，利用物品的外观特征（exposure feature）和内容特征（content feature），用反事实推断的方法，去除物品外观特征对推荐结果的直接影响，解决推荐系统中“标题党”的问题。

②在推荐系统的训练数据中，通常将用户点击过的物品作为正样本进行训练。但是，用户点击一个物品不一定是因为用户喜欢这个物品，也可能是因为物品的外观很吸引人，但是内容很差。这种现象称为Clickbait Issue——引诱点击问题。例如，在视频推荐场景下，用户点击一个视频，可能只是因为视频的封面和标题做的很好，但点进去可能并不喜欢看。在文章/新闻推荐场景下也是如此，很多标题党文章可以获得很多点击，但用户对这种文章是深恶痛绝的。Clickbait Issue会导致用户对推荐系统的信任度下降，也会导致低质量的标题党信息在系统中泛滥，产生劣币驱逐良币的效果和马太效应。因此，设计和训练推荐模型时，不能只追求点击率优化，而应该追求更高的用户满意度，避免陷入“推荐标题党内容-标题党获得更多点击-推荐更多标题党内容”的恶性循环中。

③利用物品的外观信息和内容信息，区分用户的点击是因为被标题/封面吸引，还是真的喜欢物品的内涵。

###### 3.1.1.2 该方向可以应用的推荐算法

因果推荐模型：用户可能只是被标题等信息吸引而点击一个物品，因此，建模曝光特征（E）对点击（Y）的直接因果效应。暴露信息（e）在用户点击之前就能看到，比如标题和封面图；内容信息（t）在点击之后才能看到，例如文章内容、视频内容或物品详情等。、

直观理解，如果一个物品是靠标题党来吸引流量的，则这个物品在反事实世界中的点击率会很高，从而在反事实推荐模型中被排到后面去。

###### 3.1.1.3 实验数据

（1）Tiktok（http://ai-lab-challenge.bytedance.com/tce/vc，已失效）

（2）Adressa（https://reclab.idi.ntnu.no/dataset/，新闻数据集）.

###### 3.1.1.4 研究的可行性评估

有开源新闻数据集，有开源代码，有算法讲解。

#### 3.2 黑盒攻击

###### 3.2.1.1 该方向的已有研究

（1）论文标题：Attacks on Sequential Recommenders via Data-Free Model Extraction

（2）发表年份：2021

（3）论文链接：https://dl.acm.org/doi/pdf/10.1145/3460231.3474275

（4）代码开源：https://github.com/Yueeeeeeee/RecSys-Extraction-Attack

（5）研究内容：

我们研究了模型提取是否可以用于“窃取”顺序推荐系统的权重，以及对此类攻击的受害者构成的潜在威胁。这种类型的风险在图像和文本分类中引起了关注，但据我们所知，在推荐系统中没有。我们认为，由于用于训练序列推荐系统的特定自回归机制，序列推荐系统容易受到独特的漏洞。与许多现有的推荐攻击者不同，他们假设用于训练受害者模型的数据集暴露给攻击者，我们考虑的是无数据设置，其中训练数据是不可访问的。在这种情况下，我们提出了一种基于API的模型提取方法，通过有限预算的合成数据生成和知识提取。我们研究了最先进的顺序推荐模型，并展示了它们在模型提取和下游攻击下的脆弱性。

我们分两个阶段进行攻击。（1） 模型提取：给定从黑盒推荐器检索到的不同类型的合成数据及其标签，我们通过蒸馏将黑盒模型提取为白盒模型。（2） 下游攻击：我们使用白盒推荐器生成的对抗性样本攻击黑盒模型。实验表明，我们的无数据模型提取和对序列推荐器的下游攻击在配置文件污染和数据中毒设置中都是有效的。

**序列模型是一种流行的个性化框架通过捕捉用户不断发展的兴趣和项目到项目的转换模式进行推荐。**近年来，各种基于神经网络的模型，如RNN和CNN框架（例如GRU4Rec[13]，Caser[38]、NARM[25]）和Transformer框架（例如SASRec[17]，BERT4Rec[37]）被广泛使用，并且始终优于非序列模型[12，34]以及传统的序列模型[11，35]。然而，只有少数作品研究了对推荐者的攻击，并且有一定的局限性：**（1）很少有攻击方法是针对顺序模型。**通过对抗性机器学习进行的攻击在一般推荐设置中实现了最先进的[4，6，39]，但实验是在矩阵分解模型上进行的，很难直接应用于顺序推荐；尽管一些模型不可知的攻击[2，22]可以在序列中使用设置，它们在很大程度上依赖于启发式及其有效性通常是有限的；**（2） 许多攻击方法都假设经过充分训练受害者模型的数据暴露给攻击者**[4，6，24，39，44]。攻击者可以使用这些数据来训练代理本地模型。然而，这种设置是非常严格的（或不现实的），尤其是在隐式反馈设置（例如点击、视图）中，数据会攻击者很难获得。

###### 3.2.1.2 该方向可以应用的推荐算法

因果推荐模型：用户可能只是被标题等信息吸引而点击一个物品，因此，建模曝光特征（E）对点击（Y）的直接因果效应。暴露信息（e）在用户点击之前就能看到，比如标题和封面图；内容信息（t）在点击之后才能看到，例如文章内容、视频内容或物品详情等。、

直观理解，如果一个物品是靠标题党来吸引流量的，则这个物品在反事实世界中的点击率会很高，从而在反事实推荐模型中被排到后面去。

###### 3.2.1.3 实验数据

MovieLens-1M、Steam、Beauty

###### 3.2.1.4 研究的可行性评估

有开源电影数据集，有开源代码，有算法讲解。

#### 3.3 POI

###### 3.3.1.1 该方向的已有研究

（1）论文标题：CAPTOR: A Crowd-Aware Pre-Travel Recommender System for Out-of-Town Users

（2）发表年份：2022

（3）论文链接：https://dl.acm.org/doi/pdf/10.1145/3477495.3531949

（4）代码开源：https://github.com/xhran2010/CAPTOR

（5）研究内容：

**旅行前外地推荐旨在向计划在不久的将来离开家乡但尚未决定去哪里的用户推荐兴趣点（POI）**，即他们的目的地地区和POI都未知。这是一项不平凡的任务，因为搜索空间很大，这可能会导致不同外地地区的不同旅行体验，并最终混淆决策。此外，用户的外地旅行行为不仅受到其个性化偏好的影响，而且在很大程度上受到他人旅行行为的影响。为此，我们提出了一个人群感知出行前出城推荐框架（CAPTOR），该框架由两个主要模块组成：空间仿射条件随机场（SA-CRF）和人群行为记忆网络（CBMN）。具体而言，SA-CRF捕获POI之间的空间亲和力，同时保留POI的固有信息。然后，提出了CBMN，通过三个附属块自适应地读写存储器来维持每个区域的人群出行行为。我们设计了具有动态映射机制的详细度量空间，其中用户和POI在本质上和地理上都是可区分的。在两个真实世界的全国性数据集上进行的大量实验验证了CAPTOR在旅行前外地推荐任务中的有效性。

###### 3.2.1.2 该方向可以应用的推荐算法

POI算法

###### 3.2.1.3 实验数据

旅行行为数据集 https://sites.google.com/site/yangdingqi/home/foursquare-dataset

###### 3.2.1.4 研究的可行性评估

有开源旅行数据集，有开源代码，有算法讲解。

### 3.4 EE问题

###### 3.4.1.1 该方向的已有研究

（1）论文标题：PURS: Personalized Unexpected Recommender System for Improving User Satisfaction

（2）发表年份：2020

（3）论文链接：https://dl.acm.org/doi/pdf/10.1145/3383313.3412238

（4）代码开源：https://github.com/lpworld/PURS

（5）研究内容：

当用户只收到他们熟悉的项目的推荐，使他们感到无聊和不满意时，经典的推荐系统方法通常会面临过滤器泡沫问题。为了解决过滤器气泡问题，已经提出了意想不到的推荐，以推荐明显偏离用户先前期望的项目，从而通过向用户呈现“新鲜”和先前未探索的项目来让用户感到惊讶。在本文中，我们描述了一种新的个性化意外推荐系统（PURS）模型，该模型通过提供用户在潜在空间中的兴趣的多集群建模，以及通过自我注意机制和选择适当的意外激活函数的个性化意外，将意外纳入推荐过程。在三个真实世界数据集上进行的大量离线实验表明，所提出的PURS模型在准确性和意外性测量方面显著优于最先进的基线方法。此外，我们在主要视频平台阿里巴巴优酷进行了在线A/B测试，我们的模型使每个用户的平均视频观看量增加了3%以上。该公司正在部署拟议的模型。

#### 3.5 序列推荐

……



### 其他

“Serving Each User”: Supporting Different Eating Goals Through a Multi-List Recommender Interface. RecSys 2021 【通过多列表推荐界面个性化食品推荐】

只是把单行推荐界面改成了两行推荐界面，就发了论文.........真是个天才.......）



















