# 2025年3月30日起将装修github仓库（预计一到两周时间），我们将重新整理已上传代码，在此期间可能部分内容消失，大家可在公众号历史推文中查找使用，装修完毕我们会发微信推文通知，重磅升级中，敬请期待！</font>*

### 核心速览 本文提出了一种名为“Quadrangle Attention (QA)”的新型视觉变换器注意力机制，旨在通过学习数据驱动的四边形配置来计算局部注意力，从而提高视觉变换器对不同大小、形状和方向物体的适应性。 ### 研究背景 - **研究问题**：现有的基于窗口的注意力机制虽然在性能、计算复杂度和内存占用方面取得了平衡，但其设计的窗口是手工制作的，无法适应不同大小、形状和方向的物体，限制了变换器捕捉丰富上下文信息的能力。如何设计一种新的注意力机制，以提高视觉变换器对各种目标的适应性，并捕捉丰富的上下文信息？ - **研究难点**：现有的基于窗口的注意力机制由于其固定形状的窗口设计，无法有效处理不同大小、形状和方向的目标，这限制了变换器捕捉长距离依赖关系的能力。此外，如何在不显著增加计算成本的情况下，通过学习机制来适应不同目标的特征表示，也是一个挑战。 - **文献综述**：之前的研究尝试通过扩大窗口大小、设计不同形状的窗口（如Focal attention、cross-shaped window attention和Pale）来改进基于窗口的注意力机制，以增强图像分类性能。然而，这些方法仍然采用固定形状的窗口进行注意力计算，没有充分利用数据驱动的方法来自动学习最优的窗口参数。本文提出了一种数据驱动的解决方案，通过将默认的矩形窗口扩展到四边形，使变换器能够学习更好的特征表示来处理各种不同的物体。 ### 研究方法 - **四边形注意力机制（QA）**：提出了一种新颖的四边形注意力机制，该机制通过学习数据来确定注意力区域，扩展了基于窗口的注意力到四边形的通用形式。该方法包含一个端到端可学习的四边形回归模块，预测变换矩阵以将默认窗口转换为目标四边形，从而实现对不同形状和方向目标的建模，并捕获丰富的上下文信息。 - **QFormer架构**：将QA集成到平面和分层视觉变换器中，创建了一个名为QFormer的新架构。该架构仅需要对现有代码进行少量修改，并且额外的计算成本可以忽略不计。QFormer在多个视觉任务上均表现出色，包括分类、目标检测、语义分割和姿态估计。 - **实验验证**：在多个公共基准数据集上进行了广泛的实验和消融研究，以验证QA的有效性和QFormer在各种视觉任务上的优越性。实验结果表明，QFormer在各种视觉任务上均优于现有的代表性视觉变换器。 ### 实验设计 - **数据集与评估指标**：在ImageNet-1k、ADE20k和MS COCO等知名公共数据集上进行实验，评估指标包括Top-1和Top-5准确率、平均精度均值（mAP）和交并比（IoU）。 - **模型规格**：QFormer模型包括不同层数和通道数的变体，例如QFormer h-T、QFormer h-S和QFormer h-B，以及对应的plain架构QFormerp-B。这些模型在不同任务上表现出色，具有不同的参数和计算复杂度。 - **训练细节**：使用AdamW优化器和不同的学习率调度策略进行训练，训练过程包括预训练和微调阶段。对于不同的下游任务，使用ImageNet-1k预训练权重进行初始化。 ### 结果与分析 - **图像分类**：QFormer在ImageNet-1k数据集上的分类任务中表现出色，特别是在处理不同大小和形状的目标时。例如，QFormer h-T在224×224输入尺寸下达到了82.5%的Top-1准确率，比Swin-T高出1.3%。 - **目标检测与实例分割**：在MS COCO数据集上，QFormer在目标检测和实例分割任务中均优于基线方法Swin-T。例如，QFormer h-T在使用Mask RCNN检测器时，相较于Swin-T在1×训练计划下提高了2.2 mAPbb和1.7 mAPmk。 - **语义分割**：在ADE20k数据集上，QFormer在语义分割任务中也取得了优异的成绩。例如，QFormer h-T在512×512图像上达到了43.6 mIoU，比使用固定窗口注意力的ViT-B模型高出2.0 mIoU。 - **姿态估计**：在MS COCO数据集上，QFormer在姿态估计任务中同样表现出色。例如，QFormer h-T在使用Mask RCNN检测器时，相较于Swin-T在1×训练计划下提高了0.6 APbbs、0.9 APblb和1.6 APmlk。 ### 总体结论 - **研究发现**：提出的四边形注意力机制（QA）能够有效地从数据中学习注意力区域，显著提升了视觉变换器处理不同大小、形状和方向目标的能力。通过将QA集成到视觉变换器中，创建了QFormer架构，该架构在多个视觉任务上均表现出色，包括分类、目标检测、语义分割和姿态估计。 - **解释与分析**：QFormer通过学习适应性窗口配置，能够更好地建模长距离依赖关系，并促进跨窗口信息交换，从而学习到更好的特征表示。实验结果表明，QFormer在各种视觉任务上均优于现有的代表性视觉变换器，证明了QA的有效性和QFormer架构的优越性。 - **意外发现**：尽管QFormer在性能上有所提升，但其在推理速度上仅比Swin Transformer慢约13%，这表明QA在实现速度和准确性之间的更好权衡方面具有巨大潜力。此外，QFormer在处理不同尺度对象时表现出色，这表明其学习到的四边形能够适应各种形状和方向的目标。


![0 0【更多免费资源，在此获取】_00](https://github.com/user-attachments/assets/3f4393eb-1fef-4200-8110-2df8bb3f2f4b)

![副本_ai缝合术 (1)](https://github.com/user-attachments/assets/aee8eecc-e4d7-408d-b0b8-4cecbec23ffd)

## 最新更新请关注微信公众号【AI缝合术】

## Plug-play-modules（即插即用模块）
2025年全网最全即插即用模块，免费分享！CVPR2025，AAAI2025，ICLR2025，TNNLS2025，arXiv2025......包含人工智能全领域（机器学习、深度学习等），适用于图像分类、目标检测、实例分割、语义分割、全景分割、姿态识别、医学图像分割、视频目标分割、图像抠图、图像编辑、单目标跟踪、多目标跟踪、行人重识别、RGBT、图像去噪、去雨、去雾、去阴影、去模糊、超分辨率、去反光、去摩尔纹、图像恢复、图像修复、高光谱图像恢复、图像融合、图像上色、高动态范围成像、视频与图像压缩、3D点云、3D目标检测、3D语义分割、3D姿态识别等各类计算机视觉和图像处理任务，以及自然语言处理、大语言模型、多模态等其他各类人工智能相关任务。持续更新中......

# 论文解读

### 1. [动态滤波模块 (AAAI 2024) DynamicFilter.py](https://github.com/AIFengheshu/Plug-play-modules/blob/main/(AAAI%202024)%20DynamicFilter.py)

题目：FFT-based Dynamic Token Mixer for Vision

中文题目：基于FFT的动态令牌混合器在视觉中的应用

地址：https://arxiv.org/pdf/2303.03932

![image](https://github.com/user-attachments/assets/a94aebac-98b5-40de-96c6-7ec30e3853c0)

论文解读：[【AAAI 2024】计算复杂度更低，基于FFT的动态滤波器模块，即插即用](https://mp.weixin.qq.com/s/Kv_cc31-n27LuSgt10MBTA)

### 2. [小波卷积 (ECCV 2024) WTConv.py](https://github.com/AIFengheshu/Plug-play-modules/blob/main/(ECCV%202024)%20WTConv.py)

题目：Wavelet Convolutions for Large Receptive Fields

地址：https://arxiv.org/pdf/2407.05848v2

![image](https://github.com/user-attachments/assets/52e02a5f-93a8-4f62-9cfe-0da99fffea87)

论文解读：[【ECCV 2024】大感受野的小波卷积，即插即用，显著提高CNN的有效感受野](https://mp.weixin.qq.com/s/I3Qh1yPWbr9sqEql1DAsjg)

### 3. [FreqFusion特征融合模块 (TPAMI 2024) FreqFusion.py](https://github.com/AIFengheshu/Plug-play-modules/blob/main/(TPAMI%202024)%20FreqFusion.py)

题目：Frequency-aware Feature Fusion for Dense Image Prediction

地址：https://arxiv.org/pdf/2408.12879

![image](https://github.com/user-attachments/assets/606f7faa-44c5-43e0-9a08-cbd578762d4c)

论文解读：[【TPAMI 2024】即插即用的FreqFusion特征融合模块，语义分割、目标检测、实例分割和全景分割统统涨点！](https://mp.weixin.qq.com/s/u5gmg66VnFGzKCiMTHGyRw)

### 4. [缩放点积注意力 (arXiv 2023) ScaledDotProductAttention.py](https://github.com/AIFengheshu/Plug-play-modules/blob/main/(arXiv%202023)%20ScaledDotProductAttention.py)

题目：Attention Is All You Need

地址：https://arxiv.org/pdf/1706.03762

![image](https://github.com/user-attachments/assets/3c16748d-6fda-4f03-bc2b-f29ebabf6c05)

论文解读：[【被引13w+】Scaled Dot-Product Attention（缩放点积注意力），被引最高的注意力机制](https://mp.weixin.qq.com/s/lAxZmKu8jgO-kALtv-uBgg)

### 5. [代理注意力 (ECCV 2024) Agent-Attention.py](https://github.com/AIFengheshu/Plug-play-modules/blob/main/(ECCV%202024)%20Agent-Attention.py)

题目：Agent Attention: On the Integration of Softmax and Linear Attention

地址：https://arxiv.org/pdf/2312.08874

![image](https://github.com/user-attachments/assets/59cc83ea-605f-45d2-9065-82bae343d92c)

论文解读：[【ECCV 2024】新注意力范式——Agent Attention，整合Softmax与线性注意力](https://mp.weixin.qq.com/s/giFUFLCCA5WLHnq_kF3NpQ)

### 6. [可变形卷积 (CVPR 2019) DCNv2.py](https://github.com/AIFengheshu/Plug-play-modules/blob/main/(CVPR%202019)%20DCNv2.py)

题目：Deformable ConvNets v2: More Deformable, Better Results

地址：https://arxiv.org/pdf/1811.11168

![image](https://github.com/user-attachments/assets/da7a0a6c-9f9a-4612-b33b-8e933005ac95)

论文解读：[可变形卷积（DCNv2），即插即用，直接替换普通卷积，助力模型涨点！增强网络特征提取能力！](https://mp.weixin.qq.com/s/ptGGWtCmsJqxKLGAbYHwLA)

### 7. [高效多尺度注意力 (ICCASSP 2023) EMA.py](https://github.com/AIFengheshu/Plug-play-modules/blob/main/(ICCASSP%202023)%20EMA.py)

题目：Efficient Multi-Scale Attention Module with Cross-Spatial Learning

中文题目：高效的跨空间学习多尺度注意力模块

地址：https://arxiv.org/pdf/2305.13563v2

![image](https://github.com/user-attachments/assets/a1006ff0-52bd-4399-8b55-07eb61935bef)

论文解读：[【ICCASSP 2023】即插即用的高效多尺度注意力EMA，战胜SE、CBAM、SA、CA等注意力](https://mp.weixin.qq.com/s/F7G1LvO4N_HX5wooLztCew)

### 8. [无参数注意力模块 (ICML 2021) SimAM.py](https://github.com/AIFengheshu/Plug-play-modules/blob/main/(ICML%202021)%20SimAM.py)

论文题目：SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks 

SimAM:一种用于卷积神经网络的简单、无参数的注意力模块

论文链接：http://proceedings.mlr.press/v139/yang21o/yang21o.pdf

![image](https://github.com/user-attachments/assets/7d93efc5-e2a6-43f2-aeee-8ac9db198a2e)

论文解读：[【ICML 2021】无参数注意力模块SimAM，即插即用，不超过10行代码，有效涨点！](https://mp.weixin.qq.com/s/OI5RYlm100Lpiqjxj1e0-g)

### 9. [部分卷积 (CVPR 2023) PConv.py](https://github.com/AIFengheshu/Plug-play-modules/blob/main/(CVPR%202023)%20PConv.py)

论文题目：Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks

跑起来！不要走：追求更高FLOPS以实现更快的神经网络

论文链接：https://arxiv.org/pdf/2303.03667

![image](https://github.com/user-attachments/assets/82a037ed-c270-460c-ad17-5867677f0d60)

论文解读：[【CVPR 2023】部分卷积（PConv），简单有效，即插即用，快速涨点](https://mp.weixin.qq.com/s/6tss5ZaOoolnzfbK4eKm-A)

### 10. [大选择核模块 (IJCV 2024) LSK.py](https://github.com/AIFengheshu/Plug-play-modules/blob/main/(IJCV%202024)%20LSK.py)

论文题目：LSKNet: A Foundation Lightweight Backbone forRemote Sensing

中文题目: LSKNet: 一种用于遥感的基础轻量级骨干网络论文

链接：https://arxiv.org/pdf/2403.11735

所属机构: 南开大学天津，湖南先进技术研发研究院长沙，深圳福田NKAIRI

关键词： 遥感，CNN骨干，大核，注意力，目标检测，语义分割

![image](https://github.com/user-attachments/assets/d59bd472-2765-42ef-a76a-9aadd1b414f0)

论文解读：[【IJCV 2024】大选择核模块LSK，可当作卷积块进行替换，即插即用，极大增加感受野](https://mp.weixin.qq.com/s/G9gLLbfUV0FzRn-vczREQA)

### 11. [(TPAMI 2021) OutlookAttention.py](https://github.com/AIFengheshu/Plug-play-modules/blob/main/(TPAMI%202021)%20OutlookAttention.py)

论文题目：VOLO: Vision Outlooker for Visual Recognition

中文题目:  VOLO：视觉识别的视觉展望器

论文链接：https://arxiv.org/pdf/2106.13112

所属机构: Sea AI Lab和新加坡国立大学

![image](https://github.com/user-attachments/assets/6670b07c-d50d-460e-9418-a387453477b7)

论文解读：[【TPAMI 2022】Outlook Attention，即插即用，捕捉细节和局部特征，助力模型涨点](https://mp.weixin.qq.com/s/v4AWmS0dP4vyTmvtoIPxqA)

### 12. [Haar小波下采样 (PR2023) HaarWDownsampling.py](https://github.com/AIFengheshu/Plug-play-modules/blob/main/(PR2023)%20HaarWDownsampling.py)

论文题目：Haar wavelet downsampling: A simple but effective downsampling module for semantic segmentation

中文题目:  Haar小波下采样：一个简单但有效的语义分割下采样模块

论文链接：https://www.sciencedirect.com/science/article/pii/S0031320323005174

![image](https://github.com/user-attachments/assets/c3cdccda-4357-45a3-a6e6-97a54404a934)

论文解读：[【PR 2023】Haar小波下采样，即插即用，几行代码，简单有效提高语义分割准确性](https://mp.weixin.qq.com/s/n1YQeSN2dOcB8cIm3RLXOw)

### 13. [空间和通道协同注意力 (arXiv 2024) SCSA.py](https://github.com/AIFengheshu/Plug-play-modules/blob/main/(arXiv%202024)%20SCSA.py)

论文题目：SCSA: Exploring the Synergistic Effects Between Spatial and Channel Attention

中文题目:  SCSA: 探索空间注意力和通道注意力之间的协同效应

论文链接：https://arxiv.org/pdf/2407.05128

所属机构：浙江师范大学计算机科学与技术学院，杭州人工智能研究所，北京极客智能科技有限公司

关键词：多语义信息，语义差异，空间注意力，通道注意力，协同效应

![image](https://github.com/user-attachments/assets/973ff9d4-ab46-4256-98b2-6e60cd11a156)

论文解读：[【arXiv 2024】最新！空间和通道协同注意力SCSA，即插即用，分类、检测、分割涨点！](https://mp.weixin.qq.com/s/RK-bVHt8-D5dCUI1yJ5NpA)

### 14. [上下文锚点注意力CAA (CVPR 2024) CAA.py](https://github.com/AIFengheshu/Plug-play-modules/blob/main/(CVPR%202024)%20CAA.py)

论文题目：Poly Kernel Inception Network for Remote Sensing Detection

中文题目:  面向遥感检测的多核Inception

网络论文链接：https://openaccess.thecvf.com/content/CVPR2024/papers/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.pdf

所属机构：南京理工大学, 中国传媒大学, 浙江大学关键词：遥感图像, 目标检测, 多尺度卷积核, 上下文锚点注意力, PKINet

![image](https://github.com/user-attachments/assets/04a36fd7-7ca2-446d-853e-a11b1678706e)

论文解读：[【CVPR 2024】上下文锚点注意力CAA，即插即用，助力目标检测涨点！](https://mp.weixin.qq.com/s/KjtmN4OWf7AiivV5cvuOSQ)

### 15. [线性可变形卷积LDConv (IVC 2024)LDConv.py](https://github.com/AIFengheshu/Plug-play-modules/blob/main/(IVC%202024)LDConv.py)

论文题目：LDConv: Linear deformable convoluton for improving convolutioanl neural networks

中文题目:  LDConv：用于改进卷积神经网络的线性可变形卷积

论文链接：https://doi.org/10.1016/j.imavis.2024.105190

所属机构：重庆师范大学，西南大学

关键词：新型卷积操作、任意采样形状、任意数量的参数、目标检测

![image](https://github.com/user-attachments/assets/bcb557aa-015b-40a9-81e7-44f2b4df4f25)

论文解读：[【IVC 2024】线性可变形卷积LDConv，即插即用，高效提取特征，YOLO网络涨点神器！](https://mp.weixin.qq.com/s/rBZqtUM-87hHBNtuVgElCA)

### 16. [全维度动态卷积ODConv (ICLR 2022) ODConv.py](https://github.com/AIFengheshu/Plug-play-modules/blob/main/(ICLR%202022)%20ODConv.py)

论文题目：Omni-Dimensional Dynamic Convolution

中文题目:  全维度动态卷积论文链接：https://openreview.net/pdf?id=DmpCfq6Mg39

官方github：https://github.com/OSVAI/ODConv

所属机构：英特尔中国实验室，香港中文大学-商汤科技联合实验室

关键词：动态卷积, 注意力机制, 卷积核空间, 深度学习, 计算机视觉

![image](https://github.com/user-attachments/assets/de920e27-0e5c-4214-be9f-35b6bd565efb)

论文解读：[【ICLR 2022】全维度动态卷积ODConv，即插即用，用于CV所有任务涨点！](https://mp.weixin.qq.com/s/Q-eS1Jt8K5Ri2LUd_PLxlA)
