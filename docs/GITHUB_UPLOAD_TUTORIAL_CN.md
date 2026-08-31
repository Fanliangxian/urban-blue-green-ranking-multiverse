# GitHub上传教程（中文）

## 一、上传前先完成五项人工信息

1. 打开 `CITATION.cff`，把“Authors to be confirmed”替换为最终作者。
2. 填写论文DOI（若尚未发表可暂时保留预出版状态）。
3. 在README补充通讯作者公开联系方式。
4. 决定仓库名称；建议 `urban-blue-green-ranking-multiverse`。
5. 审稿阶段建议先创建Private仓库，确认期刊匿名与代码政策后再公开。

## 二、本地最终检查

在PowerShell中运行：

```powershell
cd "<发布包所在路径>\urban-blue-green-ranking-release"
python scripts\check_release.py
python -m unittest discover -s tests -v
git status
```

确认没有原始空间数据、本机路径、私人Earth Engine资产ID、账户令牌或超过100 MB的文件。

## 三、在GitHub创建空仓库

1. 登录 https://github.com/。
2. 点击右上角 `+` → `New repository`。
3. Repository name填写 `urban-blue-green-ranking-multiverse`。
4. Description可填写：`Analysis and figure-generation code for a lineage-aware urban blue-green ranking measurement multiverse across Chinese cities.`
5. 初次上传建议选择 `Private`。
6. 不要勾选自动创建README、LICENSE或.gitignore；本地发布包已经包含这些文件。
7. 点击 `Create repository`。

## 四、用本地Git上传

```powershell
cd "<发布包所在路径>\urban-blue-green-ranking-release"
git init
git branch -M main
git add README.md DATA.md CITATION.cff LICENSE .gitignore environment.yml requirements.txt CONTRIBUTING.md
git add config docs gee scripts tests RELEASE_MANIFEST.csv RELEASE_QA_REPORT.md
git status
git commit -m "Initial research code release"
git remote add origin https://github.com/YOUR_ACCOUNT/urban-blue-green-ranking-multiverse.git
git push -u origin main
```

把 `YOUR_ACCOUNT` 替换为GitHub用户名。GitHub不接受账户密码进行HTTPS推送；使用浏览器授权、GitHub Desktop或Personal Access Token。不要把Token写进命令脚本、README或远程地址。

## 五、线上检查

- README链接是否正常；
- `CITATION.cff`是否被识别为“Cite this repository”；
- DATA与REPRODUCIBILITY文档是否能打开；
- 仓库中是否意外出现个人路径或大型数据；
- 仓库仍为Private时，是否按期刊要求向编辑提供访问方式。

## 六、公开版本和Zenodo DOI

1. 论文版本冻结后，在GitHub右侧选择 `Releases` → `Draft a new release`。
2. Tag建议使用 `v1.0.0`。
3. Release标题可用 `Code accompanying the urban blue-green ranking multiverse study`。
4. 用GitHub账户登录Zenodo，在Zenodo的GitHub设置中启用该仓库。
5. 发布GitHub Release后，Zenodo会归档并生成DOI。
6. 把Zenodo DOI和正式论文信息写回 `CITATION.cff`、README及论文Code Availability，再发布修订标签。

## 七、后续更新

```powershell
git status
git add README.md CITATION.cff docs scripts config gee tests
git commit -m "Update release documentation and code"
git push
```

不要未经检查直接使用 `git add .`。如果凭证已经误传，立即将仓库设为Private、撤销凭证并清理Git历史；只删除最新文件不能从历史中移除秘密。

## 八、论文Code Availability建议写法

> The data-processing, statistical-analysis and figure-generation code used in this study is available at [GitHub URL] and archived at Zenodo ([DOI]). Raw third-party datasets are not redistributed and can be obtained from the original providers listed in the repository documentation.
