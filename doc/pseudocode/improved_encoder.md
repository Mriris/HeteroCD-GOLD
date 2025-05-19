算法 3: 改进的编码器
输入: 模态1的特征 $F_1$, 模态2的特征 $F_2$
输出: 增强后的特征 $F_1'$, 增强后的特征 $F_2'$

1.  通道注意力权重计算:
    a.  FOR $i \in \{1, 2\}$ DO: (处理两个模态)
        i.   计算平均池化特征: $F_{avg}^i \leftarrow AvgPool(F_i)$
        ii.  计算最大池化特征: $F_{max}^i \leftarrow MaxPool(F_i)$
        iii. 计算模态特定权重: $W_i \leftarrow \sigma(MLP(F_{avg}^i) + MLP(F_{max}^i))$
    b.  计算共享特征表示:
        i.   平均池化特征融合: $F_{avg}^s \leftarrow [AvgPool(F_1), AvgPool(F_2)]$
        ii.  最大池化特征融合: $F_{max}^s \leftarrow [MaxPool(F_1), MaxPool(F_2)]$
        iii. 计算共享权重: $W_s \leftarrow \sigma(MLP_s(F_{avg}^s + F_{max}^s))$

2.  特征增强:
    a.  模态间交互增强:
        i.   增强模态1: $F_1^{cross} \leftarrow F_1 \cdot W_2$ (应用模态2权重到模态1)
        ii.  增强模态2: $F_2^{cross} \leftarrow F_2 \cdot W_1$ (应用模态1权重到模态2) 
    b.  共享权重增强:
        i.   增强模态1: $F_1^{shared} \leftarrow F_1 \cdot W_s$
        ii.  增强模态2: $F_2^{shared} \leftarrow F_2 \cdot W_s$
    c.  残差连接:
        i.   最终模态1特征: $F_1' \leftarrow F_1^{cross} + F_1^{shared} + F_1$
        ii.  最终模态2特征: $F_2' \leftarrow F_2^{cross} + F_2^{shared} + F_2$

3.  RETURN $F_1', F_2'$ 