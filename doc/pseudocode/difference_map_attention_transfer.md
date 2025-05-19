算法 5: 差异图引导的注意力迁移损失
输入: 学生网络特征 $F_{s1}$, $F_{s2}$, 教师网络特征 $F_{t1}$, $F_{t2}$, 增强后的特征 $F_{s\_enh}$, $F_{t\_enh}$
输出: 差异图注意力蒸馏损失 $\mathcal{L}_{att\_D}$

1.  生成差异图:
    a.  FOR $i \in \{s, t\}$ DO: (分别处理学生和教师网络)
        i.   $F_1 \leftarrow $ 选择对应的第一个特征 ($F_{s1}$ 或 $F_{t1}$)
        ii.  $F_2 \leftarrow $ 选择对应的第二个特征 ($F_{s2}$ 或 $F_{t2}$)
        iii. 计算L2距离差异: $D_{L2}^i \leftarrow \sqrt{\sum((F_1 - F_2)^2) + \epsilon}$
        iv.  计算余弦距离差异: $D_{cos}^i \leftarrow 1 - \frac{F_1 \cdot F_2}{\|F_1\| \cdot \|F_2\|}$
        v.   计算组合差异: $D_{comb}^i \leftarrow (D_{L2}^i + D_{cos}^i) / 2$
        vi.  生成最终差异图: $D_i \leftarrow \tanh(2D_{comb}^i) \cdot 0.5 + 0.5$

2.  生成注意力图:
    a.  FOR $i \in \{s, t\}$ DO: (分别处理学生和教师网络)
        i.   $F_{enh} \leftarrow $ 选择对应的增强特征 ($F_{s\_enh}$ 或 $F_{t\_enh}$)
        ii.  计算空间注意力: $A_{sp}^i \leftarrow \sigma(Conv([AvgPool_c(F_{enh}), MaxPool_c(F_{enh})]))$
        iii. 计算通道注意力: $A_{ch}^i \leftarrow \sigma(MLP(AvgPool(F_{enh})) + MLP(MaxPool(F_{enh})))$

3.  计算注意力迁移损失:
    a.  差异图迁移损失: $\mathcal{L}_{map} \leftarrow MSE(D_s, D_t)$
    b.  通道注意力迁移损失: $\mathcal{L}_{ch} \leftarrow MSE(A_{ch}^s, A_{ch}^t)$
    c.  空间注意力迁移损失: $\mathcal{L}_{sp} \leftarrow MSE(A_{sp}^s, A_{sp}^t)$

4.  计算加权综合损失: $\mathcal{L}_{att\_D} \leftarrow 0.5\mathcal{L}_{map} + 0.3\mathcal{L}_{ch} + 0.2\mathcal{L}_{sp}$
    
5.  RETURN $\mathcal{L}_{att\_D}$ 