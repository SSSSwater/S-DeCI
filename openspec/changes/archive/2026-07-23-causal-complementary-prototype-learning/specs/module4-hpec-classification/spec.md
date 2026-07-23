## MODIFIED Requirements

### Requirement: 模块 4 调度独立可靠 TP prototype 更新

模块 4 SHALL 对外暴露训练后 prototype 更新接口，使 prototype 的数据驱动移动独立于总 loss 和 autograd。

#### Scenario: optimizer step 后更新
- **GIVEN** 训练 batch 已完成 forward、backward 和 `optimizer.step()`
- **WHEN** 训练循环调用 prototype 更新接口
- **THEN** 接口 MUST 使用 detach 后的标准 `z_global`、训练标签和标准最终 logits
- **AND** 可靠样本 MUST 同时满足预测正确与置信度阈值
- **AND** prototype 更新 MUST 在无梯度上下文执行 EMA
- **AND** 验证、测试和推理 MUST 不更新 prototype

#### Scenario: 更新不依赖互补视图
- **GIVEN** `hpec_prototype_update_mode == "reliable_tp_ema"`
- **WHEN** 模块 4 更新 prototype
- **THEN** 系统 MUST NOT 要求互补 `z_global`
- **AND** MUST NOT 使用互补一致性权重作为可靠样本必要条件

### Requirement: prototype 更新模式可回退

模块 4 SHALL 支持 `reliable_tp_ema`、`sinkhorn_ema` 和 `none` 三种明确模式。

#### Scenario: 切换模式
- **WHEN** 用户设置 `hpec_prototype_update_mode`
- **THEN** `reliable_tp_ema` MUST 不调用 Sinkhorn
- **AND** `sinkhorn_ema` MUST 仅作为 legacy 对照
- **AND** `none` MUST 禁止数据驱动 prototype 移动
