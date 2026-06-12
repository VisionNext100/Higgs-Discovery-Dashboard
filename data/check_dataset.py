"""
CERN HiggsML 数据集全量检查脚本
用途：全面了解数据分布、缺失值、特征相关性等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("CERN HiggsML 数据集全量检查报告")
print("="*80)

# 1. 加载全量数据
print("\n1. 加载全量数据...")
df = pd.read_csv("atlas-higgs-challenge-2014-v2.csv")
print(f"   ✅ 加载完成！总行数: {len(df):,}, 总列数: {len(df.columns)}")

# 2. 数据概览
print("\n2. 数据概览")
print(f"   - 内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"   - 重复行: {df.duplicated().sum():,}")
print(f"   - 数据类型分布:")
for dtype in df.dtypes.value_counts().items():
    print(f"       {dtype[0]}: {dtype[1]} 列")

print("\n所有列标题（共{}列）:".format(len(df.columns)))
for i, col in enumerate(df.columns):
    print("   {:2d}. {}".format(i+1, col))

# 3. 标签分布
print("\n3. 标签分布 (Label)")
label_counts = df['Label'].value_counts()
print(f"   - 信号(s): {label_counts.get('s', 0):,} ({label_counts.get('s', 0)/len(df)*100:.2f}%)")
print(f"   - 背景(b): {label_counts.get('b', 0):,} ({label_counts.get('b', 0)/len(df)*100:.2f}%)")

# 4. KaggleSet 分布（数据划分）
print("\n4. KaggleSet 分布（官方数据划分）")
kaggle_counts = df['KaggleSet'].value_counts()
for kaggle_set in ['t', 'b', 'v']:
    count = kaggle_counts.get(kaggle_set, 0)
    print(f"   - {kaggle_set} ({'训练集' if kaggle_set=='t' else '公开测试集' if kaggle_set=='b' else '私有测试集'}): {count:,} ({count/len(df)*100:.2f}%)")

# 5. PRI_jet_num 分布（核心创新点）
print("\n5. PRI_jet_num 分布（喷注数量）")
jet_counts = df['PRI_jet_num'].value_counts().sort_index()
for jet_num in [0, 1, 2, 3]:
    count = jet_counts.get(jet_num, 0)
    print(f"   - {jet_num} 个喷注: {count:,} ({count/len(df)*100:.2f}%)")

# 6. 缺失值全面统计
print("\n6. 缺失值统计（每列 -999 的比例）")
missing_stats = []
for col in df.columns:
    missing_count = (df[col] == -999).sum()
    missing_pct = missing_count / len(df) * 100
    if missing_pct > 0:
        missing_stats.append((col, missing_count, missing_pct))

missing_stats.sort(key=lambda x: x[2], reverse=True)
print(f"   共有 {len(missing_stats)} 列存在缺失值，按缺失率排序：")
for col, count, pct in missing_stats[:15]:  # 显示前15个
    print(f"   - {col}: {count:,} ({pct:.2f}%)")
if len(missing_stats) > 15:
    print(f"   ... 还有 {len(missing_stats)-15} 列")

# 7. Weight 字段统计
print("\n7. Weight 字段统计（用于AMS计算）")
print(f"   - 最小值: {df['Weight'].min():.6f}")
print(f"   - 最大值: {df['Weight'].max():.6f}")
print(f"   - 均值: {df['Weight'].mean():.6f}")
print(f"   - 中位数: {df['Weight'].median():.6f}")
print(f"   - 标准差: {df['Weight'].std():.6f}")

# 8. 不同喷注数量下的缺失模式（关键洞察）
print("\n8. 不同喷注数量下的缺失模式（按PRI_jet_num分组）")
jet_groups = df.groupby('PRI_jet_num')
for jet_num in [0, 1, 2, 3]:
    sub_df = jet_groups.get_group(jet_num)
    # 检查喷注相关特征的缺失情况
    jet_features = ['PRI_jet_leading_pt', 'PRI_jet_subleading_pt', 'DER_deltaeta_jet_jet']
    missing_info = []
    for feat in jet_features:
        if feat in sub_df.columns:
            missing_pct = (sub_df[feat] == -999).sum() / len(sub_df) * 100
            missing_info.append(f"{feat}: {missing_pct:.1f}%缺失")
    print(f"   PRI_jet_num={jet_num} ({len(sub_df):,}条): {', '.join(missing_info)}")

# 9. 特征相关性分析（可选，需要安装matplotlib）
print("\n9. 特征相关性分析")
print("   正在计算特征相关性矩阵（可能需要10-20秒）...")

# 准备数据：标签转换为0/1，缺失值用-999临时填充（仅用于相关性计算）
plot_df = df.copy()
plot_df['Label_binary'] = (plot_df['Label'] == 's').astype(int)
# 选择部分特征（避免维度爆炸）
feature_cols = ['DER_mass_MMC', 'DER_pt_h', 'PRI_jet_num', 'PRI_lep_pt', 'PRI_met', 'Weight']
corr_matrix = plot_df[feature_cols + ['Label_binary']].corr()

# 打印与标签的相关性
print("\n   各特征与标签的相关性（绝对值越大越重要）:")
label_corr = corr_matrix['Label_binary'].sort_values(key=abs, ascending=False)
for feat, corr_val in label_corr.items():
    if feat != 'Label_binary':
        print(f"   - {feat}: {corr_val:.4f}")

# 10. 检查数据平衡性（按喷注数量×标签）
print("\n10. 信号/背景在喷注数量上的分布")
cross_tab = pd.crosstab(df['PRI_jet_num'], df['Label'], normalize='index')
print("   每个喷注数量组内的信号比例:")
for jet_num in [0, 1, 2, 3]:
    signal_pct = cross_tab.loc[jet_num, 's'] * 100
    print(f"   - PRI_jet_num={jet_num}: 信号占比 {signal_pct:.1f}%")

# 11. 数据样例（随机抽取5行）
print("\n11. 随机数据样例（5行）")
sample_rows = df.sample(5, random_state=42)
for idx, row in sample_rows.iterrows():
    print(f"   EventId {int(row['EventId'])}: Label={row['Label']}, PRI_jet_num={int(row['PRI_jet_num'])}, Weight={row['Weight']:.4f}")

# 12. 数据划分建议
print("\n" + "="*80)
print("数据划分建议（基于KaggleSet）")
print("="*80)
print("""
训练集 (Training):     df[df['KaggleSet'] == 't']   # 用于训练模型
公开测试集 (Public):    df[df['KaggleSet'] == 'b']   # 用于调参和验证
私有测试集 (Private):   df[df['KaggleSet'] == 'v']   # 用于最终评估

⚠️ 注意: 在调参过程中，不要在私有测试集上反复测试，只在最后评估一次！
""")

# 13. 保存报告到文件
report = f"""
========================================
HiggsML 数据集全量检查报告
========================================

数据规模:
- 总行数: {len(df):,}
- 总列数: {len(df.columns)}
- 内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

标签分布:
- 信号(s): {label_counts.get('s', 0):,} ({label_counts.get('s', 0)/len(df)*100:.2f}%)
- 背景(b): {label_counts.get('b', 0):,} ({label_counts.get('b', 0)/len(df)*100:.2f}%)

数据划分 (KaggleSet):
- 训练集(t): {kaggle_counts.get('t', 0):,}
- 公开测试集(b): {kaggle_counts.get('b', 0):,}
- 私有测试集(v): {kaggle_counts.get('v', 0):,}

喷注数量分布:
- 0个喷注: {jet_counts.get(0, 0):,}
- 1个喷注: {jet_counts.get(1, 0):,}
- 2个喷注: {jet_counts.get(2, 0):,}
- 3个喷注: {jet_counts.get(3, 0):,}

缺失值统计:
- 有缺失值的列数: {len(missing_stats)}
- 缺失最严重的列: {missing_stats[0][0]} ({missing_stats[0][2]:.2f}%)
"""
print("\n报告已生成，可复制保存。")

print("\n" + "="*80)
print("全量检查完成！")
print("="*80)