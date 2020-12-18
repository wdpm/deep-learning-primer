# Make xticks label not overlap

## scale figure size
```python
plt.figure(figsize=(12,4)) # 设置画布大小
```

## adjust labelsize
```python
plt.tick_params(axis='x', labelsize=8)    # 设置x轴标签大小
```

## flip x, y axis
```python
plt.barh(df['sport_type'], df['score'])    # 绘制横向柱状图
```

## label rotation
```python
plt.xticks(rotation=-15)    # 设置x轴标签旋转角度
```