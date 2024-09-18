---
marp: true
size: 16:9
theme: am_green
math: katex
paginate: true
headingDivider: [2,3]
footer: \ *Chaebeom Sheen* *Physics for Neuromoprhic Computing* *September 22nd, 2024*

---

<!-- _class: cover_e -->
<!-- _paginate: "" -->
<!-- _footer: ![](assets/snu-wide.png) -->
<!-- _header: ![](assets/marp.png) -->


# <!-- fit --> Physics for neuromorphic computing

###### Danijela Marković, Alice Mizrahi, Damien Querlioz, et al. 


As reviewed by Chaebeom Sheen
Date: September 22nd, 2024
<cauchybs@snu.ac.kr>


---

<!-- _header: <br>CONTENTS<br>![](assets/snucn.png)-->
<!-- _class: toc_b -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

- [Introduction](#3)
- [From Von Neumann to Boltzman](#7) 
- [What are Memristors?](#11)
- [Mimicing AI](#20)
- [Mimicing Brains](#38)
- [From Circuits to Systems](#45)
- [Take Home Message](#48)

## 1. Introduction

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->


## 1. Introduction

<!-- _class: navbar -->
<!-- _header: \ ***@ SNU CN*** **Introduction** *Memristors* *Architectures* *Conclusion*-->

- CPUs are specialized for sequential processing of complicated tasks, making them ideal for general purpose computing.
- Meanwhile, GPUs and TPUs are specialized for parallel processing of simple tasks, making them ideal for artificial neural networks.

```python
import math

def iterative():
    sum = 0
    for i in range(1, 10**6 +1):
        sum += math.exp(-i)
    return sum
%timeit iterative()
```

Output: 102 ms ± 777 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)

## 1. Introduction

<!-- _class: navbar -->
<!-- _header: \ ***@ SNU CN*** **Introduction** *Memristors* *Architectures* *Conclusion*-->
```python
# vectorized with GPU
import jax
import jax.numpy as jnp

@jax.jit
def vectorized_gpu():
    return jnp.sum(jnp.exp(- jnp.arange(1, 10**6 +1)))

vectorized_gpu() # compile the function
%timeit vectorized_gpu()
```
Output: 3.99 μs ± 116 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

## 1. Introduction

<!-- _class: navbar-->
<!-- _header: \ ***@ SNU CN*** **Introduction** *Memristors* *Architectures* *Conclusion*-->
- However, both CPUs and GPUs are based on the *von Neumann architecture*, where memory and processing are separate. The so-called *von Neumann bottleneck* is a limitation on the data transfer rate between the multi-processor and memory. (Recall $R \propto A^{-1}$)
![#c h:400](assets/von-neumann.png)

## 2. From Von Neumann to Boltzmann

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

## 2. From Von Neumann to Boltzman

<!-- _class: navbar col1_ol_sq fglass -->
<!-- _header: \ ***@ SNU CN*** **Introduction** *Memristors* *Architectures* *Conclusion*-->

Biological brains have several properties which make them ideal for deep neural networks:

- **Massive parallelism**: the brain has 86 billion neurons, each with 10,000 synapses.
- **Adaptative learning**: the brain can learn from experience through synaptic plasticity. 
- **Low power consumption**: the brain consumes only 20 W of power, compared to 100kWh. 


## 2. From Von Neumann to Boltzman

<!-- _class: navbar col1_ol_sq fglass -->
<!-- _header: \ ***@ SNU CN*** *Introduction* **Memristors** *Architectures* *Conclusion*-->

Mimicing the brain's architecture to accelerate deep neural network *training* and *inference* is the goal of neuromorphic computing. However, achieving neuromorphic isomorphism requires a fundamental shift in the way we approach computation.

- **Stochasticity**: the brain is inherently stochastic.
- **Asynchronicity**: the brain is asynchronous.
- **Plasticity**: the brain is plastic.
- **Memory in situ**: the brain has integrated memory and processing.

## 2. From Von Neumann to Boltzman

<!-- _class: navbar col1_ol_sq fglass -->
<!-- _header: \ ***@ SNU CN*** **Introduction** *Memristors* *Architectures* *Conclusion*-->

Current CMOS (complementary metal-oxide-semiconductor) technology is not well-suited for neuromorphic computing. 

- **Von Neumann bottleneck**: the separation of memory and processing.
- **Bulkiness**: Processor capacity and efficiency is inherently limited by chip area. 
- **Low Interconnectivity**: Chip fan out is limited to 2-dimensions, limiting connectivity.

## 3. What are Memristors?

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

## 3. What are Memristors?

<!-- _class: navbar cols-2-->
<!-- _header: \ ***@ SNU CN*** *Introduction* **Memristors** *Architectures* *Conclusion*-->
<div class=limg>


![#c h:400](assets/memristors.png)
</div>

- Theorized by Leon Chua in 1971 $^1$
- The four fundamental quantities:  $q$, $i$, $\phi$, and $v$
  - $q$: amount of charge
  - $i$: amount of current 
  - $\phi$: amount of (magnetic) flux
  - $v$: amount of voltage
- Clearly, there can be $\binom{4}{2}$ relationships between these quantities. Two relationships naturally follow from basic physics:
  - $\mathrm{d}q = i \mathrm{d}t$ (from the definition of the current)
  - $\mathrm{d}\phi = v \mathrm{d}t$ (from Faraday's law)

## 3. What are Memristors?

<!-- _class: navbar cols-2-->
<!-- _header: \ ***@ SNU CN*** *Introduction* **Memristors** *Architectures* *Conclusion*-->
<div class=limg>


![#c h:400](assets/memristors.png)
</div>

- The other 3 relationships follow from basic circuit elements.
  - $\mathrm{d}v = R \mathrm{d} i$ (Ohm's law)
  - $\mathrm{d}\phi = L \mathrm{d} i$ (inductance)
  - $\mathrm{d}q = C \mathrm{d} v$ (capacitance)
- However, there is no quantity that relates charge and flux. This is the memristor, described by the relationship
  - $\mathrm{d}\phi = M \mathrm{d} q$
  - with $M$ the memristance.
- Note that the memristor can be written as
  - $M = \dfrac{\mathrm{d}\phi}{\mathrm{d}q} = \dfrac{v\mathrm{d}t}{i \mathrm{d}t} = \dfrac{v}{i}$

## 3. What are Memristors?
<!-- _class: navbar-->
<!-- _header: \ ***@ SNU CN*** *Introduction* **Memristors** *Architectures* *Conclusion*-->
- Trivially, if $M$ is constant, then $M \equiv R$. 
- The non-trivial non-linear case is far more revealing, and shows why it is called a **mem**ristor.
- For a simple charge-controlled memristor, where the memristance is a single-valued function of $q$,
$$ v(t) = M(q(t)) i(t) $$
- Alternatively, for a flux-controlled memristor, where memristance is a function of $\phi$,
$$ i(t) = W(\phi(t)) v(t)$$
- where $W$ is the inverse of $M$.
- $M(q)$ is referred to as the incremental memristance, and $W(\phi)$ is referred to as the incremental memductance. 

## 3. What are Memristors?
<!-- _class: navbar-->
<!-- _header: \ ***@ SNU CN*** *Introduction* **Memristors** *Architectures* *Conclusion*-->


## 《微观经济学：现代观点》

<!-- _class: col1_ol_sq fglass -->

渲染效果为**单列+有序列表+方形序号**
自定义样式为：`<!-- _class: col1_ul_sq fglass -->`

- 预算约束和消费者的最优选择
- 劳动力和储蓄的供给函数
- 福利经济学：单人模型和多人模型
- 企业理论：单投入品和多投入品模型
- 完全竞争市场
- 完全垄断、垄断竞争与双寡头垄断
- 交换经济与生产经济
- 不确定性、期望效用和不对称信息

## 《微观经济学：现代观点》

<!-- _class: col1_ol_ci fglass -->

渲染效果为**单列+有序列表+圆形序号**
自定义样式为：`<!-- _class: col1_ul_ci fglass -->`

- 预算约束和消费者的最优选择
- 劳动力和储蓄的供给函数
- 福利经济学：单人模型和多人模型
- 企业理论：单投入品和多投入品模型
- 完全竞争市场
- 完全垄断、垄断竞争与双寡头垄断
- 交换经济与生产经济
- 不确定性、期望效用和不对称信息

## 5. 引用、链接和 Callouts

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

## 5. 引用、链接和 Callouts

- 引用的呈现效果为：

> 合成控制法 (Synthetic Control Method) 最早由 Abadie and Gardeazabal (2003) 提出，用来研究西班牙巴斯克地区恐怖活动的经济成本，属于案例研究范畴 (Case Study)。

- 链接的呈现效果：
  - [经管数据清洗与 Stata 实战：三大地级市数据库和 CSMAR 上市公司数据](https://mp.weixin.qq.com/s/D0cYVPJJsNiu61GcYwV6cg)
  - [Stata 基础：从论文文件夹体系的建立说起](https://mp.weixin.qq.com/s?__biz=MzkwOTE3NDExOQ==&mid=2247486489&idx=1&sn=2eb51e85a01541c7a552a9434e087512&scene=21#wechat_redirect)
- Callouts 是 Awesome Marp 提供的自定义的样式，有 5 种颜色可选：
  - [紫色](#40)：`bq-purple`
  - [蓝色](#41)：`bq-blue`
  - [绿色](#42)：`bq-green`
  - [红色](#43)：`bq-red`
  - [黑色](#44)：`bq-black`

## 5. 引用、链接和引用盒子

<!-- _class:  bq-purple -->

- 自定义样式为：`<!-- _class:  bq-purple -->`

> 合成控制法 (Synthetic Control Method) 
> 
> SCM 最早由 Abadie and Gardeazabal (2003) 提出，用来研究西班牙巴斯克地区恐怖活动的经济成本，属于案例研究范畴 (Case Study)。Athey & Imbens (2017) 认为它是过去 15 年计量方法领域最重要的创新。<br>
> 合成控制法的基本思想是：虽然无法找到巴斯克地区的最佳控制地区，但可对西班牙的若干大城市进行适当的线性组合（赋予不同的权重），以构造一个更为贴切的「合成控制地区」 (Synthetic Control Region)，然后将真实的巴斯克地区与「合成的巴斯克地区」进行对比，即可得到恐袭的影响。

[返回](#39)

## 5. 引用、链接和引用盒子

<!-- _class:  bq-blue -->

- 自定义样式为：`<!-- _class:  bq-blue -->`

> 合成控制法 (Synthetic Control Method) 
> 
> SCM 最早由 Abadie and Gardeazabal (2003) 提出，用来研究西班牙巴斯克地区恐怖活动的经济成本，属于案例研究范畴 (Case Study)。Athey & Imbens (2017) 认为它是过去 15 年计量方法领域最重要的创新。<br>
> 合成控制法的基本思想是：虽然无法找到巴斯克地区的最佳控制地区，但可对西班牙的若干大城市进行适当的线性组合（赋予不同的权重），以构造一个更为贴切的「合成控制地区」 (Synthetic Control Region)，然后将真实的巴斯克地区与「合成的巴斯克地区」进行对比，即可得到恐袭的影响。

[返回](#39)

## 5. 引用、链接和引用盒子

<!-- _class:  bq-green -->

- 自定义样式为：`<!-- _class:  bq-green -->`

> 合成控制法 (Synthetic Control Method) 
> 
> SCM 最早由 Abadie and Gardeazabal (2003) 提出，用来研究西班牙巴斯克地区恐怖活动的经济成本，属于案例研究范畴 (Case Study)。Athey & Imbens (2017) 认为它是过去 15 年计量方法领域最重要的创新。<br>
> 合成控制法的基本思想是：虽然无法找到巴斯克地区的最佳控制地区，但可对西班牙的若干大城市进行适当的线性组合（赋予不同的权重），以构造一个更为贴切的「合成控制地区」 (Synthetic Control Region)，然后将真实的巴斯克地区与「合成的巴斯克地区」进行对比，即可得到恐袭的影响。

[返回](#39)

## 5. 引用、链接和引用盒子

<!-- _class:  bq-red -->

- 自定义样式为：`<!-- _class:  bq-red -->`

> 合成控制法 (Synthetic Control Method) 
> 
> SCM 最早由 Abadie and Gardeazabal (2003) 提出，用来研究西班牙巴斯克地区恐怖活动的经济成本，属于案例研究范畴 (Case Study)。Athey & Imbens (2017) 认为它是过去 15 年计量方法领域最重要的创新。<br>
> 合成控制法的基本思想是：虽然无法找到巴斯克地区的最佳控制地区，但可对西班牙的若干大城市进行适当的线性组合（赋予不同的权重），以构造一个更为贴切的「合成控制地区」 (Synthetic Control Region)，然后将真实的巴斯克地区与「合成的巴斯克地区」进行对比，即可得到恐袭的影响。

[返回](#39)

## 5. 引用、链接和引用盒子

<!-- _class:  bq-black -->

- 自定义样式为：`<!-- _class:  bq-black -->`

> 合成控制法 (Synthetic Control Method) 
> 
> SCM 最早由 Abadie and Gardeazabal (2003) 提出，用来研究西班牙巴斯克地区恐怖活动的经济成本，属于案例研究范畴 (Case Study)。Athey & Imbens (2017) 认为它是过去 15 年计量方法领域最重要的创新。<br>
> 合成控制法的基本思想是：虽然无法找到巴斯克地区的最佳控制地区，但可对西班牙的若干大城市进行适当的线性组合（赋予不同的权重），以构造一个更为贴切的「合成控制地区」 (Synthetic Control Region)，然后将真实的巴斯克地区与「合成的巴斯克地区」进行对比，即可得到恐袭的影响。

[返回](#39)


## 6. 导航栏

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

## 6. 导航栏

<!-- _class: navbar -->
<!-- _header: \ ***@Awesome Marp*** *关于模板* *封面页* *目录页* *分栏与分列* *引用盒子* **导航栏** *基础知识*-->

- 一句题外话：打造 Awesome Marp 模板的最早初衷就是来自几位公众号粉丝朋友的询问，「Marp 是否也能实现想 Beamer 那样的顶部导航栏？」为了实现导航栏的效果，我又多学了一些 CSS 的知识，这套模板才得以成型

- 自定义样式为 `navbar`：`<!-- _class: navbar -->` 
- 导航栏修改自 header，最前面必须加入 `\ `
- 当前活动标题，使用粗体 `**粗体**`
- 其余非活动标题，使用斜体 `*斜体*`
- 如果左侧有文字，需要使用斜粗体 `***粗斜体***`
- 默认根据内容自动分配间距，如果希望右对齐，可以手动增加空格的方式来推动右对齐 


## 6. 导航栏

<!-- _header: \ ***@Awesome Marp*** *关于模板* *封面页* *目录页* *分栏与分列* *引用盒子* **导航栏** *基础知识*-->
<!-- _class: navbar -->

这张页面的部分 Markdown 源码：

```markdown
<!-- _class: navbar -->
<!-- _header: \ ***虹鹄山庄***      

- 自定义样式为 `navbar`：`<!-- _class: navbar -->` 
- 导航栏修改自 header，最前面必须加入 `\ `
- 当前活动标题：使用粗体 `**粗体**`
- 其余非活动标题：使用斜体 `*斜体*`
- 如果左侧有文字：使用斜粗体 `***粗斜体***`
- 默认根据内容自动分配间距，如果希望右对齐，可以手动增加空格的方式来推动右对齐 
``` 

## 7. 其他自定义样式

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->



## 7.1 固定标题行：更像 Beamer 了（`fixedtitleA`）

<!-- _class: fixedtitleA -->

- 自定义样式：`<!-- _class: fixedtitleA -->`
  
  - 使当前页面的标题栏固定在顶部，而非随着内容的多少浮动
  
  - 同时，页面内容也会从顶部起笔，而非垂直方向上居中显示


## 7.1 固定标题行：更像 Beamer 了（`fixedtitleB`）

<!-- _class: fixedtitleB -->


<div class="div">

- 自定义样式：`<!-- _class: fixedtitleB -->`
  
  - `fixedtitleB` 相比于 `fixedtitleA`，标题增加了底色色块，同时缩小了标题大小
  
  - 其余效果与 `fixedtitleA` 相同 
  
  - 但是页面正文内容需要包裹在 `<div class="div'></div>` 标签中 
</div>

---

<!-- _class: footnote -->

<div class="tdiv">

#### 7.2 脚注的自定义样式：`footnote`

使用方法：

- 自定义样式：`<!-- _class: footnote -->`
- 页面除脚注外的其他内容，写在 `<div class = "tdiv"></div>` 
- 页面的脚注内容，写在 `<div class = "bdiv"></div>` 

举个例子，展示一下显示效果：

- 一方面，经济金融化程度的加深，使得金融部门能够凭借资本跨期配置提前抽取其他部门的未来价值，从而扩大金融和非金融部门之间的外部收入差距$^1$。另一方面，经济金融化不断增加企业股东权力，促使企业更加追求股东价值最大化，这一导向将弱化普通劳动者阶层的议价能力，食利者阶层的财产性收入增加必然会挤压劳动收入份额，从而扩大了内部收入差距$^2$。

</div>

<div class="bdiv">

1 张甜迪. 金融化对中国金融、非金融行业收入差距的影响[J]. 经济问题, 2015(11): 40-46.
2 Hein E. Finance-dominated capitalism and re-distribution of income: a Kaleckian perspective[J]. Cambridge Journal of Economics, 2015, 39(3): 907-934.
</div>

## 7.3 调节文字大小的自定义样式

<!-- _class: largetext -->

对于字体大小的调节，直接修改 CSS 文件应该很方便的。但有小伙伴提出，“希望可以增加字体调节的自定义样式”，于是目前提供了四种微调样式：

- 自定义样式 1：`<!-- _class: tinytext -->` （是默认字体大小的 0.8 倍）
- 自定义样式 2：`<!-- _class: smalltext -->` （是默认字体大小的 0.9 倍）
- 自定义样式 3：`<!-- _class: largetext -->` （是默认字体大小的 1.15 倍）
- 自定义样式 4：`<!-- _class: hugetext -->` （是默认字体大小的 1.3 倍）

比如，本页面采用的自定义样式为 `largetext` 

## 7.4 图表标题的自定义样式：`caption`

<!-- _class: caption -->

- 通过 `<div class="caption">宇宙的奥妙</div>` 来定义图表的标题 

![#c h:380](https://mytuchuang-1303248785.cos.ap-beijing.myqcloud.com/picgo/202401131712626.png)

<div class="caption">
宇宙的奥妙
</div>


## 需要知道的基础知识……

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->


## Markdown 概览

<!-- _header: \ ***@Awesome Marp*** *关于模板* *封面页* *目录页* *分栏与分列* *引用盒子* *导航栏* **基础知识**-->
<!-- _class: navbar -->

- Markdown 是一种**极轻量**的文本标记语言，允许人们使用**易读易写**的纯文本格式编写文档，而且对于表格、代码、图片、公式等支持良好
- 应用广泛：网站、课程笔记/讲义、演示文稿、撰写学术论文等
- Markdown 基础语法：
  - 参阅：[Markdown 中文文档](https://markdown-zh.readthedocs.io/en/latest/)、[Markdown 指南](https://www.markdown.xyz/)、[Markdown 菜鸟教程](https://www.runoob.com/markdown/md-tutorial.html)
  - 标题 `#`、粗体 `** **`、斜体 `* *`、删除线 `~~ ~~`、分割线 `---`、超链接 `[]()`
  - 引用 `>`、列表 `-` / `1. `、代码块 
  - 脚注 `[^1]` / `[^1]:`、待办事项 `[ ]` / `[x]`
- Markdown 进阶语法：
  - 图片 `![]()`：本地路径、网络路径（参阅：[图床与 PicGo——让你爱上记录与分享](https://sspai.com/post/65716)）
  - 数学公式：行内公式 `$...$`、行间公式 `$$...$$`
  - 支持 HTML 元素：`<br>`/`<hr>`/`<b></b>`/`<i></i>`/`<kbd></kbd>` 等
  
## 推荐的 Markdown 编辑器

<!-- _class: cols-2-64 navbar -->
<!-- _header: \ ***@Awesome Marp*** *关于模板* *封面页* *目录页* *分栏与分列* *引用盒子* *导航栏* **基础知识**-->

<div class=ldiv>

**VS Code**
- Visual Studio Code[下载地址](https://code.visualstudio.com/Download)
- VS Code 插件：
  - 配合 Markdown：[Markdown Preview Enhanced](https://marketplace.visualstudio.com/items?itemName=shd101wyy.markdown-preview-enhanced)、[Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)
  - 图床：[PicGo](https://marketplace.visualstudio.com/items?itemName=Spades.vs-picgo)
  - 格式化文档：[Pangu-Markdown](https://marketplace.visualstudio.com/items?itemName=xlthu.Pangu-Markdown)
  - Markdown 转 PPT：[Marp for VScode](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode)
  - Markdown 转思维导图：[Markmap for VScode](https://marketplace.visualstudio.com/items?itemName=gera2ld.markmap-vscode)
  - 配合 Zotero：[Citation Picker for Zotero](https://marketplace.visualstudio.com/items?itemName=mblode.zotero)、[Pandoc Citer](https://marketplace.visualstudio.com/items?itemName=notZaki.pandocciter)

</div>

<div class=rdiv>

**Obsidian**
- [Obsidian 主页](https://obsidian.md/)
- 基于 Markdown 的本地知识管理软件
- 除官方同步和发布功能外，对个人使用者完全免费
- 功能丰富、插件众多、开发社区活跃

</div>


## Marp 基本用法

<!-- _header: \ ***@Awesome Marp*** *关于模板* *封面页* *目录页* *分栏与分列* *引用盒子* *导航栏* **基础知识**-->
<!-- _class: navbar fixedtitleB -->

<div class="div">

- 几个字总结 [Marp](https://marp.app/)：使用 Markdown 创作演示文稿
  - 来自 Marp 官方网页的一段话：Marp (also known as the Markdown Presentation Ecosystem) provides an intuitive experience for creating beautiful slide decks. You only have to focus on writing your story in a Markdown document.

- 在 Markdown 文件的顶部 YAML 区域，通过 `marp: true` 启动 Marp，然后即可开启侧边预览，VS Code 界面左边是代码区域，右边为预览区域
- 内容遵循 Markdown 语法，但 Marp 增加了一些内置指令，而且指令分为全局指令和[局部指令](https://marpit.marp.app/directives?id=local-directives-1)，全局指令建议放置于 YAML 区，局部指令位于当前页面，不同页面通过 `---` 切分
- 推荐阅读：Marpit [官方文档](https://marpit.marp.app)及[中译版](https://caizhiyuan.gitee.io/categories/skills/20200730-marp.html#%E5%8A%9F%E8%83%BD)，五分钟学会 Marp[（上）](https://www.lianxh.cn/news/97fccdca2d7a5.html)、[（下）](https://www.lianxh.cn/news/521900220dd33.html)
</div>

## Marp 基本用法

<!-- _header: \ ***@Awesome Marp*** *关于模板* *封面页* *目录页* *分栏与分列* *引用盒子* *导航栏* **基础知识**-->
<!-- _class: navbar -->

```yaml
---
marp: true        # 开启 Marp 
size: 16:9        # 设定页面比例，常见有 16:9 或 4:3，默认为16:9
theme: gaia       # 切换主题，内置 3 种样式的主题，可以自定义主题
paginate: true    # 开启页码
headingDivider: 2 # 通过二级标题切分页面，省去手动换页的麻烦
footer: 初虹 # 设置页脚区域的内容，如果设定页眉的内容，则为 header
---
```

- 如果想让页面同时被多个级别的标题切分，比如，以二级~四级标题分割页面，可以 `headingDivider: [2,3,4]` 
- 想要使得多个自定义样式渲染同一个页面，可直接将不同自定义样式以空格连接，比如：`<!-- _class: cols-2-64 fglass -->`


---

<!-- _class: lastpage -->
<!-- _footer: "" -->

###### Q&A 
