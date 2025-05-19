算法 4: 在线知识蒸馏
输入: 学生网络特征 $F_s$, 教师网络特征 $F_t$, 学生网络软输出 $S_s$, 教师网络软输出 $S_t$, 学生网络差异图 $D_s$, 教师网络差异图 $D_t$, 学生网络空间注意力 $A_{sp}^s$, 学生网络通道注意力 $A_{ch}^s$, 教师网络空间注意力 $A_{sp}^t$, 教师网络通道注意力 $A_{ch}^t$, 温度参数 $T$
输出: 蒸馏损失 $\mathcal{L}_D$

1.  特征表示迁移:
    a.  计算特征值迁移损失: $\mathcal{L}_{MSE} \leftarrow MSE(F_s, F_t)$
    b.  计算特征方向迁移损失: $\mathcal{L}_{cos} \leftarrow 1 - Cos(F_s, F_t)$
    c.  计算加权特征级损失: $\mathcal{L}_{feat} \leftarrow 0.7 \cdot \mathcal{L}_{MSE} + 0.3 \cdot \mathcal{L}_{cos}$

2.  输出分布迁移:
    a.  计算软标签KL散度: $\mathcal{L}_{out} \leftarrow D_{KL}(S_s, S_t) \cdot T^2$

3.  差异图注意力迁移:
    a.  IF 通道数 $C > 16$: (使用分组计算提高效率)
        i.  将通道分为 $G$ 组，每组 $C/G$ 个通道
        ii. FOR $g \leftarrow 1$ TO $G$ DO:
            1. 处理第 $g$ 组的注意力迁移
    b.  ELSE: (直接计算所有通道)
        i.  计算差异图迁移损失: $\mathcal{L}_{map} \leftarrow MSE(D_s, D_t)$
        ii. 计算空间注意力迁移损失: $\mathcal{L}_{sp} \leftarrow MSE(A_{sp}^s, A_{sp}^t)$
        iii.计算通道注意力迁移损失: $\mathcal{L}_{ch} \leftarrow MSE(A_{ch}^s, A_{ch}^t)$
    c.  计算差异图注意力综合损失: $\mathcal{L}_{att\_D} \leftarrow 0.5\mathcal{L}_{map} + 0.3\mathcal{L}_{ch} + 0.2\mathcal{L}_{sp}$

4.  计算总蒸馏损失:
    a.  对不同层级损失加权融合: $\mathcal{L}_D \leftarrow 0.3\mathcal{L}_{feat} + 0.4\mathcal{L}_{out} + 0.3\mathcal{L}_{att\_D}$
    b.  IF 训练epoch数 $> 10$: (逐步增强输出分布迁移权重)
        i.  $\alpha \leftarrow \min(0.4 + 0.01 \cdot (\text{epoch} - 10), 0.6)$
        ii. $\mathcal{L}_D \leftarrow (1-\alpha)\mathcal{L}_{feat} + \alpha\mathcal{L}_{out} + (1-\alpha)\mathcal{L}_{att\_D}$

5.  RETURN $\mathcal{L}_D$ 