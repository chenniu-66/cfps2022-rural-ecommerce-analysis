# 数据目录

本目录不包含 CFPS 原始数据。

运行前请把以下文件放到本地任意路径，并通过命令行参数传入：

- `cfps2022famecon_202410.dta`
- `cfps2022person_202410.dta`

示例：

```bash
python analysis.py \
  --fam "D:/data/cfps2022famecon_202410.dta" \
  --adult "D:/data/cfps2022person_202410.dta" \
  --out output/real
```

## 合规提醒

CFPS 数据受北京大学中国社会科学调查中心使用协议保护。请勿：

- 把 `.dta` 文件提交到 Git；
- 把原始数据或行级衍生数据上传到第三方平台；
- 在公开作品集中提供可下载的数据副本。

本项目只保留代码和汇总结果；`data/raw/` 已加入 `.gitignore`。
