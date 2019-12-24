# SentimentAnalysis

### 项目概述
- 项目参考了[`Yoon Kim`](http://www.people.fas.harvard.edu/~yoonkim/)的论文[`Convolutional Neural Networks for Sentence Classification`](https://arxiv.org/pdf/1408.5882.pdf)的实现方法，利用CNN卷积神经网络完成语句情感分析。
- 项目结构
    - `data/*`：json数据
    - `parse_data`：生成数据的json文件
    - `cnn.py`：CNN网络模型
    - `train.py`：训练脚本
    - `test.py`：测试脚本
- 参数
    ```python
    ALLOW_SOFT_PLACEMENT=True
    BATCH_SIZE=50
    CHECKPOINT_EVERY=100
    DEV_DATA_FILE=./data/dev.json
    DROPOUT_KEEP_PROB=0.5
    EMBEDDING_DIM=300
    EVALUATE_EVERY=100
    FILTER_SIZES=3,4,5
    L2_REG_LAMBDA=3.0
    LOG_DEVICE_PLACEMENT=False
    NUM_CHECKPOINTS=5
    NUM_EPOCHS=200
    NUM_FILTERS=100
    TEST_DATA_FILE=./data/test.json
    TRAIN_DATA_FILE=./data/train.json
    ```

### 步骤
- 模型数据：`https://cloud.tsinghua.edu.cn/d/e3da1c00a9e84a5d9132/`
- Train with dropout.
    ```python
    $ python test.py  --checkpoint_dir="./runs/1577169494/checkpoints/"
    Total number of test examples: 2210
    Accuracy: 0.400905
    ```

- Train with 256 or 512 hidden size.
    - 修改参数`NUM_FILTERS = 256`
        ```python
        $ python test.py  --checkpoint_dir="./runs/1577171119/checkpoints/"
        Total number of test examples: 2210
        Accuracy: 0.40905
        ```
    - 修改参数`NUM_FILTERS = 512`
        ```python
        $ python test.py  --checkpoint_dir="./runs/1577172661/checkpoints/"
        Total number of test examples: 2210
        Accuracy: 0.39819
        ```

- Train with a different number of the hidden layer.  (The number of hiddenlayer should be set to 1 and 3)
    - 修改参数`FILTER_SIZES = 3`
        ```python
        $ python test.py  --checkpoint_dir="./runs/1577174708/checkpoints/"
        Total number of test examples: 2210
        Accuracy: 0.384163
        ```
    - 修改参数`FILTER_SIZES = 3,4`
        ```python
        $ python test.py  --checkpoint_dir="./runs/1577175634/checkpoints/"
        Total number of test examples: 2210
        Accuracy: 0.39819
        ```
    - 修改参数`FILTER_SIZES = 3,4,5`
        ```python
        $ python test.py  --checkpoint_dir="./runs/1577169494/checkpoints/"
        Total number of test examples: 2210
        Accuracy: 0.400905
        ```

- Train  with  pre-trained  word  embedding.   (We  supply  GloVe  pretrainedword  embedding  with  300-dimension  for  your  experiments  and  you  canexplore  the  model  performance  with  the  same  dimension  without  pre-trained word embeddings.)
    ```python
    $ python test.py  --checkpoint_dir="./runs/1577022248/checkpoints/"
    Total number of test examples: 2210
    Accuracy: 0.414027
    ```
