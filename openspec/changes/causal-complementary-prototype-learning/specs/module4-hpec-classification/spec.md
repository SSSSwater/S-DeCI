## ADDED Requirements

### Requirement: 模块 4 提供双视图 HPEC 诊断

模块 4 SHALL 在互补训练启用时为标准和互补双曲点提供可比较的 HPEC energy 与 prototype 匹配诊断，但默认预测仍使用标准视图最终 logits。

#### Scenario: 记录双视图 HPEC 输出
- **GIVEN** `use_causal_complementary_learning == 1`
- **WHEN** 模块 4 处理标准和互补 `z_global`
- **THEN** 系统 MUST 缓存两视图的类别 energy、预测和 prototype assignment 或等价匹配量
- **AND** 最终训练/验证/测试指标 MUST 继续使用标准视图最终 logits
- **AND** 互补视图输出 MUST 不在评估阶段影响预测

### Requirement: 模块 4 调度可靠 prototype 更新

模块 4 SHALL 对外暴露训练后 prototype 更新接口，供训练循环在 optimizer 更新后调用。

#### Scenario: 仅训练期更新
- **GIVEN** 模型已经完成训练 batch 的 optimizer step
- **WHEN** 训练循环调用模块 4 prototype 更新接口
- **THEN** 接口 MUST 接收 detach 后的标准点、标签、标准最终 logits 及可选互补点
- **AND** MUST 根据 `hpec_prototype_update_mode` 执行或跳过更新
- **AND** 验证和测试循环 MUST 不调用该接口
