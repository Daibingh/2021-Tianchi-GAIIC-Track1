# 2021 全球人工智能技术创新大赛【赛道一】

**——医学影像报告异常检测**


HSH 队分享，成绩 初赛 rank7，复赛 rank12

用到的模型：
- BiGRU
- BiGRU-Atten
- RCNN
- BERT
- NE-ZHA (最后提交没有使用)

## 安装依赖

```
pip installl -r requirements.txt
```

## 全流程运行

首先挂载原始数据

```
ln -s tcdata /tcdata
```

```
sh run.sh
```

