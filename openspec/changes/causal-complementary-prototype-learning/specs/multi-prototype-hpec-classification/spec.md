## ADDED Requirements

### Requirement: 多 prototype 支持可靠 TP 更新模式

系统 SHALL 让每类多 prototype 的数据驱动移动方式由 `hpec_prototype_update_mode` 控制，并在可靠 TP 模式中保持 HPEC energy 的多 prototype 聚合语义。

#### Scenario: 初始化可靠 TP 多 prototype
- **GIVEN** `use_hpec_module4 == 1`
- **AND** `hpec_prototype_update_mode == "reliable_tp_ema"`
- **WHEN** 初始化 HPEC 多 prototype 层
- **THEN** prototype 张量 MUST 保持 `[classes, prototypes_per_class, embedding_dim]`
- **AND** prototype 参数 MUST 不被 optimizer 的梯度更新路径移动
- **AND** 系统 MUST 保留初始化切空间 anchor 供 EMA 稳定更新使用

#### Scenario: 多 prototype 保留 energy winner 分配
- **GIVEN** 一个可靠训练样本属于类别 `k`
- **WHEN** 系统更新 `k` 的多个 prototype
- **THEN** 系统 MUST 依据该样本对 `k` 内 prototype 的 energy 或等价 HPEC 匹配量分配目标 prototype
- **AND** MUST 不强制将每个 batch 均衡分配给所有 prototype
