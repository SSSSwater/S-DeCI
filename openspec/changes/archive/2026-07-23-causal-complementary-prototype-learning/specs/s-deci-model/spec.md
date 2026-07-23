## MODIFIED Requirements

### Requirement: S-DeCI 协调标准视图与独立原型更新

`S-DeCI` SHALL 保持 `forward()` 返回标准最终 logits，并向训练循环暴露优化器更新后的可靠 prototype 更新接口。

#### Scenario: forward 返回保持兼容
- **WHEN** 调用 `y_hat = model(x)`
- **THEN** `forward()` MUST 返回现有训练、验证和测试流程可用的最终 logits
- **AND** prototype 更新输入和统计 MUST 通过模型属性或方法读取
- **AND** prototype EMA MUST 不作为总 loss 项或触发第二次 backward

### Requirement: 新增机制服从模块开关

`S-DeCI` SHALL 校验可靠 prototype 更新与多阶因果编码的模块依赖。

#### Scenario: 模块 3/4 关闭
- **GIVEN** 模块 3/4 未启用
- **WHEN** 用户启用可靠 TP prototype 更新或多阶因果编码
- **THEN** 系统 MUST 以清晰错误拒绝不兼容组合或明确跳过
- **AND** MUST 不影响 GCN fallback 路径

### Requirement: 互补学习退出正式 forward

`S-DeCI` SHALL 不再执行因果显著性遮挡互补分支。

#### Scenario: 计算总 loss
- **WHEN** 模型聚合分类、因果发现和流形结构损失
- **THEN** 总 loss MUST NOT 包含互补 Poincare 距离、InfoNCE 或 masked CE
- **AND** 测试标签 MUST 只用于指标和最终可视化颜色，不得进入模型 forward
