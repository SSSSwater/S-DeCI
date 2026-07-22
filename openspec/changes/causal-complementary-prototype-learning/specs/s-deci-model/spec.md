## ADDED Requirements

### Requirement: S-DeCI 协调互补视图与独立原型更新

`S-DeCI` SHALL 在保持现有 `forward()` 主返回 logits 的条件下，缓存互补视图中间量并向训练循环暴露独立 prototype 更新接口。

#### Scenario: forward 返回保持兼容
- **GIVEN** 任何新增功能处于启用或关闭状态
- **WHEN** 调用 `y_hat = model(x)`
- **THEN** `forward()` MUST 返回现有训练流程可用的标准视图最终 logits
- **AND** 互补表征、视图 loss、显著性和 prototype 更新输入 MUST 通过模型属性或方法读取

#### Scenario: 总 loss 只增加配置启用的一致性项
- **GIVEN** 模块 2、模块 3 和模块 4 均开启
- **WHEN** 训练流程计算总 loss
- **THEN** 系统 MUST 保留当前分类 loss、模块 2 loss 和已配置流形正则
- **AND** 只有 `causal_complementary_view_loss_weight > 0` 时才可增加双视图一致性项
- **AND** 可靠 TP EMA prototype 更新 MUST 不作为总 loss 项

### Requirement: 新增参数兼容所有模块开关

`S-DeCI` SHALL 对新增开关进行组合校验，避免在模块 3/4 关闭时误执行互补或 prototype 机制。

#### Scenario: 模块 3/4 关闭时拒绝或跳过新增机制
- **GIVEN** `use_hyperbolic_modules34 == 0`
- **WHEN** 用户启用互补视图、可靠 prototype 更新或多阶因果编码
- **THEN** 系统 MUST 以清晰错误拒绝不兼容组合，或以明确日志跳过对应机制
- **AND** MUST 不影响原有 GCN fallback 路径
