## REMOVED Requirements

### Requirement: 因果显著性互补视图

**移除原因**：MDD/AAL116 完整 5-fold、50 epoch 消融未超过可靠 TP EMA 基线；InfoNCE 接近 `log(batch_size)`，说明标准/遮挡表示未形成有效实例对应关系。额外前向、遮挡 CE 和一致性 loss 增加复杂度但没有稳定泛化收益。

#### Scenario: 正式训练不运行互补分支
- **WHEN** 当前 S-DeCI 执行训练、验证或测试
- **THEN** 系统 MUST 只运行标准模块 3/4 视图
- **AND** MUST NOT 重新执行共享模块 3/4 的遮挡互补前向
- **AND** MUST NOT 将互补视图距离、InfoNCE 或 masked CE 加入总 loss

### Requirement: 动态显著性遮挡与双曲一致性

**移除原因**：高显著 ROI 遮挡没有稳定形成与标准视图互补的疾病信息，反而削弱小样本分类边界。

#### Scenario: 训练入口不暴露退役参数
- **WHEN** 用户查看正式训练入口
- **THEN** 系统 MUST NOT 要求互补 mask 日程、显著性融合、双视图一致性、InfoNCE 或 masked CE 参数
- **AND** 历史结果 MAY 保留在证据文档中，但 MUST 标注为负向实验
