mmW_FNO_Project

```text
root-
    |-data
        |-__init__.py
        |-data_interface.py
        |-xxxdataset1.py
        |-xxxdataset2.py
        |-...
    |-model
        |-__init__.py
        |-model_interface.py
        |-xxxmodel1.py
        |-xxxmodel2.py
        |-...
    |-main.py
```

deeponet 参数解析

* **`input_dim`** : 主干网络（trunk net）的输入维度，这里是 2，表示输入坐标的维度。
* **`operator_dims`** : 每个分支网络（branch net）的输入维度列表，表示每个操作的维度。
* **`output_dim`** : 输出函数的维度，这里为 1。
* **`planes_branch`** 和  **`planes_trunk`** : 表示 branch 和 trunk 网络隐藏层的大小配置（列表格式）。
* **`activation`** : 使用的激活函数，默认为 GELU。
* **`learning_rate`** : 优化器的初始学习率。
* **`step_size`** 和  **`gamma`** : 学习率调度器的配置，用于控制学习率的下降。
* **`weight_decay`** : 优化器的权重衰减参数。
* **`eta_min`** : 学习率调度器的最小学习率。
* **`grid`** : 提供计算时的空间网格，通常是离散化的输入坐标点
* `branches` 是多个分支网络的集合。每个分支网络的输入维度由 `operator_dims` 决定，隐藏层的配置由 `planes_branch` 决定。
* `trunks` 是多个主干网络的集合。主干网络的输入维度由 `input_dim` 决定，隐藏层的配置由 `planes_trunk` 决定。
* **`u_vars`** : 输入分支网络的变量列表，每个变量对应一个分支网络。
* **`y_var`** : 主干网络的输入变量（通常是坐标网格）。

```text
Input (u_vars)             Input (y_var)
                  |                          |
        +-------------------+       +-------------------+
        |  Branch Networks  |       |  Trunk Networks   |
        +-------------------+       +-------------------+
                  |                          |
                  |                          |
            Branch Features           Trunk Features
                  \                          /
                   \                        /
                    \      Fusion (B * T) /
                     +-------------------+
                            Output

```

这是因为  **`y_var` 和 `u_var` 在 DeepONet 中的设计目的不同** ，它们分别对应于 **主干网络 (trunk network)** 和 **分支网络 (branch network)** 的输入。这两个网络所需要的输入结构各自满足不同的功能需求，导致它们的形状设计有所不同。以下是详细的解释：

---

### 1. **DeepONet 的核心思想**

DeepONet 的目标是近似一个算子 G(u)(y)G(u)(y)**G**(**u**)**(**y**)**，也就是对函数 uu**u** 在点 yy**y** 上的值进行预测。
将其分解为：
G(u)(y)≈∑i=1pbi(u)⋅ti(y)G(u)(y) \approx \sum_{i=1}^{p} b_i(u) \cdot t_i(y)**G**(**u**)**(**y**)**≈**∑**i**=**1**p****b**i****(**u**)**⋅**t**i****(**y**)**
其中：

* bi(u)b_i(u)**b**i(**u**) 是分支网络的输出，对函数 uu**u** 的整体表示。
* ti(y)t_i(y)**t**i(**y**) 是主干网络的输出，定义了点 yy**y** 上的特征。

为了实现这一点：

* **`u_var`** ：用于分支网络，代表函数 uu**u** 的采样值或整体特征。
* **`y_var`** ：用于主干网络，代表目标点 yy**y** 的坐标信息。

---

### 2. **为什么 `y_var` 有 NxyN_**N**x**y

* **设计目的** ：
  主干网络需要对空间上的多个点 yy**y** 进行特征提取，来描述算子输出的空间分布。因此，`y_var` 的维度是 (batch_size,Nxy,input_dim)(batch\_size, N_{xy}, input\_dim)**(**ba**t**c**h**_**s**i**ze**,**N**x**y****,**in**p**u**t**_**d**im**)**，其中：
* NxyN_{xy}**N**x**y** 表示在一个批次中每个样本的坐标点数量。
* input_diminput\_dim**in**p**u**t**_**d**im** 表示每个点 yy**y** 的输入维度（比如 2 表示二维平面上的点坐标 (x,y)(x, y)**(**x**,**y**)**）。
* **形状的意义** ：
  主干网络的目标是针对每个点 yy**y** 输出一组特征，因此需要保留 NxyN_{xy}**N**x**y** 这一维度，用于描述 yy**y** 的空间分布。

---

### 3. **为什么 `u_var` 没有 NxyN_**N**x**y

* **设计目的** ：
  分支网络的输入 `u_var` 通常是函数 uu**u** 的采样点或整体特征，它是全局信息的表达，因此与目标点 yy**y** 的数量无关。
* 例如，`u_var` 的形状是 (batch_size,operator_dim)(batch\_size, operator\_dim)**(**ba**t**c**h**_**s**i**ze**,**o**p**er**a**t**or**_**d**im**)，其中 operator_dimoperator\_dim**o**p**er**a**t**or**_**d**im** 表示 uu**u** 被采样的总点数（如 100 个采样点组成一个向量）。
* **形状的意义** ：
  分支网络只需要对 uu**u** 的整体特征进行编码，输出一组固定的表示（如隐层维度为 `planes_branch[-1]`），供后续与主干网络特征结合使用。

---

### 4. **形状上的差异总结**

| **变量** | **输入维度**                                                                                                                                                                            | **作用**                    | **形状含义**                         |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ------------------------------------------ |
| `u_var`      | (batch_size,operator_dim)(batch\_size, operator\_dim)**(**ba**t**c**h**_**s**i**ze**,**o**p**er**a**t**or**_**d**im**)                | 表示函数uu**u**的整体特征   | 全局信息，与目标点yy**y**无关        |
| `y_var`      | (batch_size,Nxy,input_dim)(batch\_size, N_{xy}, input\_dim)**(**ba**t**c**h**_**s**i**ze**,**N**x**y****,**in**p**u**t**_**d**im**)** | 表示目标点yy**y**的坐标信息 | 局部信息，描述算子输出在不同点上的特征分布 |

---

### 5. **为何二者维度差异是合理的**

在 DeepONet 中：

* `u_var` 用于生成与函数 uu**u** 整体相关的全局特征 bi(u)b_i(u)**b**i(**u**)，它只依赖 uu**u**，与 yy**y** 的数量无关。
* `y_var` 用于生成与坐标 yy**y** 相关的局部特征 ti(y)t_i(y)**t**i(**y**)，需要保留 NxyN_{xy}**N**x**y** 来描述每个点的特征。

最终，`u_var` 和 `y_var` 的特征通过点乘或其他方式结合，得到输出 G(u)(y)G(u)(y)**G**(**u**)**(**y**)**，其形状为 (batch_size,Nxy,output_dim)(batch\_size, N_{xy}, output\_dim)**(**ba**t**c**h**_**s**i**ze**,**N**x**y****,**o**u**tp**u**t**_**d**im**)。

**4o**



FNO构建了一个从偏微分方程参数函数a(x)到解函数u(x)的映射， 输入a(x)经MLP（P）升维后经过T个傅里叶层，最后经MLP（Q）降维得到输出u(x)。
一个傅里叶层内的具体操作则包括傅里叶变换F，线性变换R，傅里叶逆变换F’等。



* **`sos`** ：输入数据的特征张量，通常为源信号。
* **`src`** ：用于特定修正项的辅助输入（如参考信号）。
