算法 6: 基于不确定性的任务权重分配 (Uncertainty-Based Task Weight Allocation)
输入: 各任务不确定性参数 $\{v_k\}_{k=1}^K$, 当前训练轮次 $e$, 热身阶段总轮次 $e_w$, 各任务固定权重 $\{w_{f,k}\}_{k=1}^K$, 主任务索引 $k_0$
输出: 各任务的动态权重 $\{w_k\}_{k=1}^K$

1.  初始化各任务初始权重：对于 $k \in \{1,2,...,K\}$, $p_k \leftarrow softplus(-\log v_k) + \epsilon$
2.  主任务权重增强 (热身阶段后):
    a.  IF $e > e_w$: 
        i.  $\lambda \leftarrow \min(1.0 + (e - e_w)/100, 1.5)$
        ii. $p_{k_0} \leftarrow \lambda \cdot p_{k_0}$
3.  计算混合系数 $\alpha$:
    a.  IF $e \leq e_w$: $\alpha \leftarrow 0.5(1 - \cos(\frac{e}{e_w}\pi))$
    b.  ELSE: $\alpha \leftarrow 1.0$
4.  计算权重总和：$P_{sum} \leftarrow \sum_{j=1}^K p_j$
5.  计算各任务最终权重：对于 $k \in \{1,2,...,K\}$, $w_k \leftarrow (1-\alpha)w_{f,k} + \alpha \frac{p_k}{P_{sum}}$
6.  RETURN $\{w_k\}_{k=1}^K$

---

算法 7: 区域感知动态权重损失计算 (Region-Aware Dynamic Weight Loss Calculation)
输入: 教师网络原始差异图 $D_o$, 学生网络特征 $F_s$, 教师网络特征 $F_t$, 学生网络差异图 $D_s$, 学生网络空间注意力 $A_{sp}^s$, 学生网络通道注意力 $A_{ch}^s$, 教师网络差异图 $D_t$, 教师网络空间注意力 $A_{sp}^t$, 教师网络通道注意力 $A_{ch}^t$
输出: 区域感知动态权重损失 $\mathcal{L}_{DW}$

1.  计算动态调整因子：$S_d \leftarrow 2 \cdot (D_o > 1.3 \cdot \overline{D_o}) + 0.5$ ($\overline{D_o}$ 为 $D_o$ 的均值)
2.  增强差异图：$D_{aug} \leftarrow D_o \cdot S_d$
3.  归一化增强差异图：$D_n \leftarrow (D_{aug} - \min(D_{aug})) / (\max(D_{aug}) - \min(D_{aug}) + \epsilon)$
4.  计算自适应阈值：$\theta_{adapt} \leftarrow \overline{D_n} + \text{stddev}(D_n)$ ($\overline{D_n}$ 为均值, $\text{stddev}(D_n)$ 为标准差)
5.  二次增强/抑制差异图 $D_e$：
    a.  IF $D_n > \theta_{adapt}$: $D_e \leftarrow D_n \cdot (D_n / \theta_{adapt})$
    b.  ELSE: $D_e \leftarrow D_n \cdot (D_n / (2\theta_{adapt}))$
6.  计算最终权重掩码：$M_w \leftarrow \exp(2D_e) / \overline{\exp(2D_e)}$ ($\overline{\exp(2D_e)}$ 为均值)
7.  划分变化区域和非变化区域：
    a.  $R_c \leftarrow (M_w > 1.5 \cdot \overline{M_w})$ ($\overline{M_w}$ 为 $M_w$ 的均值)
    b.  $R_{nc} \leftarrow 1 - R_c$
8.  计算区域特征迁移损失：
    a.  $\mathcal{L}_c \leftarrow 2 \cdot MSE(F_s \cdot R_c, F_t \cdot R_c)$
    b.  $\mathcal{L}_{nc} \leftarrow 0.5 \cdot MSE(F_s \cdot R_{nc}, F_t \cdot R_{nc})$
9.  计算特征级损失(综合)：$\mathcal{L}_f \leftarrow \mathcal{L}_c + \mathcal{L}_{nc}$ (按文档描述，默认等权重组合)
10. 计算注意力机制迁移损失：$\mathcal{L}_a \leftarrow MSE(A_{sp}^s, A_{sp}^t) + MSE(A_{ch}^s, A_{ch}^t)$
11. 计算差异图迁移损失：$\mathcal{L}_d \leftarrow MSE(D_s, D_t)$
12. 计算总动态权重损失：$\mathcal{L}_{DW} \leftarrow 0.5\mathcal{L}_f + 0.3\mathcal{L}_a + 0.2\mathcal{L}_d$
13. RETURN $\mathcal{L}_{DW}$ 