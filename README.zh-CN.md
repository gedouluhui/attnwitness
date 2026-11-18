# AttnWitness：注意力掩码契约检查

检查“实际 mask 表达的可见关系”是否符合你独立声明的因果、分段或 prefix 规则，输出具体 query→key 违规位置、JSON 和可离线打开的交互报告。

它不证明模型质量、整个计算图的因果性或所有训练数据泄漏问题。`PASS` 只覆盖报告中列出的声明；未知／不支持的输入不会冒充通过。实验性 0.2：这是**硬可见性契约检查器**。

## 快速使用

在项目根目录、自己的 Python 虚拟环境中执行：

```sh
python -m pip install .
attnwitness demo --case causal --html correct.html
attnwitness demo --case float-mask --json report.json --html report.html
```

第二条演示命令会以 **1** 退出，因为它确实检测到错误；不是程序崩溃。打开 `report.html` 可查看矩阵、筛选违规边并下载 JSON。已有文件不会被覆盖，除非显式添加 `--force`。

0/1 浮点掩码与布尔掩码不同：SDPA 会把浮点 mask 加到分数上，`0.0` 并不禁止读取。有限负偏置也不等于绝对屏蔽。SDPA 与 MHA 的布尔极性相反，不能直接互换。

自有文件：

```sh
python examples/make_mask.py
attnwitness check examples/causal.json examples/causal.npy --html custom.html
```

NPY 输入禁止 object/pickle；读取前有格式、形状和体量检查。报告指纹对应实际加载的同一份字节。输入文件不能被报告路径覆盖，包括通过硬链接；未加 `--force` 时，预检之后才出现的同名文件也不会被覆盖。

批量输入中，`tokens` 按 batch 给出 token 列表，`batch_size` 与 `heads` 为显式 JSON 字段。MHA 的浮点 attention mask 与浮点 key-padding mask 会按声明 dtype 先求和再判定可见性，与真实后端合并一致：溢出为 `-inf` 视为屏蔽，`NaN` 和正无穷为无效输入；混合浮点 dtype 直接拒绝。

默认展开关系预算 1,048,576 个单元，按有界 tile（`--tile-size`，默认 128）处理，峰值内存由最大 tile 而非完整逻辑关系决定。CLI 的 `--max-cells` 允许 1..16,777,216。违规总数不受 finding／预览上限影响；保留的 finding 带有各自 batch/head 的原始坐标上下文窗口（`--max-evidence`、`--window-radius`），证据截断与 finding 截断分别报告。退出码：0 PASS，1 VIOLATION，2 无效／不支持／错误。

## 真实 PyTorch 接入

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[torch]"
```

使用 `attnwitness.torch_adapter.audit_torch(mask, policy)`。只接受普通 CPU 稠密布尔／float16／float32／float64 张量，不改变梯度、训练状态或原 mask。

`reduce_case` 可以对显式传入的可信、确定性 mask 构造器做单 token 删除缩减；限制 1–64 token，保留原始 ID、位置和角色。预算耗尽不宣称最小；完成后也只保证该删除操作下的 1-minimal，不是全局最小程序。

## 支持边界

- 方形 self-attention；causal、packed、prefix、bidirectional 声明；三态部分矩阵契约。
- MHA 只支持显式 mask 且 `is_causal=False`；不猜测优化 hint。浮点 attention + key-padding mask 求和后判定，混合浮点 dtype 拒绝。
- MaskedArray 等带额外语义的子类直接拒绝，不静默丢失缺失标记。
- 无效 query 可以不约束；是否允许空行由声明决定。“允许空行”不表示数值输出一定安全。
- 不覆盖 KV-cache、cross-attention、NestedTensor、GPU 内核或任意完整模型。

运行测试：`python -m unittest discover -s tests -v`。测试包含真实 CPU SDPA/MHA 对照、独立手算结果、受控源码副本变异、HTML 注入与输入指纹回归。

[英文说明](README.md) · [在线记录示例](https://gedouluhui.github.io/attnwitness/) · [验证范围](docs/VALIDATION.md)
