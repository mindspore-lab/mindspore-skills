# MindSpore 算子问题案例索引

从 100 个 gitcode issue 中提取的结构化案例索引。

## 统计

| 分类 | 案例数 | 状态 |
|------|--------|------|
| API/签名 | 3 | 2 DONE |
| Kernel实现 | 10 | 10 DONE |
| Shape推导 | 3 | 3 DONE |
| 其他 | 22 | 22 DONE |
| 反向传播 | 3 | 3 DONE |
| 精度/数值 | 10 | 9 DONE |
| 编译器/IR | 2 | 2 DONE |
| 运行时 | 2 | 2 DONE |

## 按分类列表

### API/签名

| Issue | 算子 | 状态 | Fix PR |
|-------|------|------|--------|
| #41954 | expm1 | DONE | https://gitee.com/mindspore/mindspore/pulls/91404 |
| #42119 | unknown |  | - |
| #42129 | norm | DONE | https://gitee.com/mindspore/mindspore/pulls/91253 |

### Kernel实现

| Issue | 算子 | 状态 | Fix PR |
|-------|------|------|--------|
| #41935 | unknown | DONE | https://gitee.com/mindspore/mindspore/pulls/91480 |
| #41943 | assign | DONE | https://gitee.com/mindspore/mindspore/pulls/91448 |
| #41945 | fft | DONE | - |
| #41947 | copy | DONE | - |
| #41948 | gelu | DONE | https://gitee.com/mindspore/mindspore/pulls/91399 |
| #41950 | unknown | DONE | - |
| #41961 | matmul | DONE | https://gitee.com/mindspore/mindspore/pulls/91422 |
| #41972 | embedding | DONE | - |
| #41977 | norm | DONE | - |
| #42191 | free | DONE | - |

### Shape推导

| Issue | 算子 | 状态 | Fix PR |
|-------|------|------|--------|
| #41926 | partial | DONE | https://gitee.com/mindspore/mindspore/pulls/91478 |
| #41956 | shape | DONE | https://gitee.com/mindspore/mindspore/pulls/91409 |
| #41971 | scatternd | DONE | https://gitee.com/mindspore/mindspore/pulls/91430 |

### 其他

| Issue | 算子 | 状态 | Fix PR |
|-------|------|------|--------|
| #41913 | sum | DONE | https://gitee.com/mindspore/mindspore/pulls/91551 |
| #41921 | unknown | DONE | - |
| #41922 | unknown | DONE | - |
| #41925 | unknown | DONE | - |
| #41930 | unknown | DONE | - |
| #41937 | sgd | DONE | - |
| #41940 | unknown | DONE | https://gitee.com/mindspore/mindspore/pulls/91392 |
| #41942 | max | DONE | https://gitee.com/mindspore/mindspore/pulls/91439 |
| #41944 | unknown | DONE | - |
| #41946 | unknown | DONE | - |
| #41949 | unknown | DONE | https://gitee.com/mindspore/mindspore/pulls/91392 |
| #41952 | unknown | DONE | https://gitee.com/mindspore/mindspore/pulls/91463 |
| #41960 | unknown | DONE | - |
| #41965 | unknown | DONE | - |
| #41975 | renorm | DONE | - |
| #42116 | mul | DONE | https://gitee.com/mindspore/mindspore/pulls/91457 |
| #42117 | unknown | DONE | - |
| #42126 | unknown | DONE | - |
| #42154 | unknown | DONE | - |
| #42160 | exp2 | DONE | - |
| #42183 | unknown | DONE | - |
| #42212 | unknown | DONE | - |

### 反向传播

| Issue | 算子 | 状态 | Fix PR |
|-------|------|------|--------|
| #41932 | pow | DONE | https://gitee.com/mindspore/mindspore/pulls/91490 |
| #41976 | binary_cross_entropy_grad | DONE | - |
| #42227 | unknown | DONE | - |

### 精度/数值

| Issue | 算子 | 状态 | Fix PR |
|-------|------|------|--------|
| #42294 | reciprocal | CANN 问题 | - (aclnnReciprocal complex64 inf 缺陷) |
| #42295 | trunc/fix | CANN 问题 | - (aclnnTrunc 910A float→int32 溢出) |
| #41931 | linear | DONE | https://gitee.com/mindspore/mindspore/pulls/70920 |
| #41933 | trace | DONE | https://gitee.com/mindspore/mindspore/pulls/91548 |
| #41934 | adam | DONE | https://gitee.com/mindspore/mindspore/pulls/70920 |
| #41941 | unknown | DONE | - |
| #41967 | acosh_ext | DONE | https://gitee.com/mindspore/mindspore/pulls/91426 |
| #41968 | unknown | DONE | - |
| #41969 | unknown | DONE | - |
| #41970 | unknown | DONE | - |
| #42166 | add | DONE | - |

### 编译器/IR

| Issue | 算子 | 状态 | Fix PR |
|-------|------|------|--------|
| #41959 | morph | DONE | https://gitee.com/mindspore/mindspore/pulls/91387 |
| #41973 | pixelshuffle | DONE | https://gitee.com/mindspore/mindspore/pulls/91361 |

### 运行时

| Issue | 算子 | 状态 | Fix PR |
|-------|------|------|--------|
| #41958 | unknown | DONE | - |
| #42184 | unknown | DONE | - |

