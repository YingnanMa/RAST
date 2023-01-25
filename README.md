# RAST: Restorable Arbitrary Style Transfer via Multi-restoration

This is the official PyTorch implementation of our paper: ["RAST: Restorable Arbitrary Style Transfer via Multi-restoration"](https://openaccess.thecvf.com/content/WACV2023/html/Ma_RAST_Restorable_Arbitrary_Style_Transfer_via_Multi-Restoration_WACV_2023_paper.html)(**WACV 2023**)   


Arbitrary style transfer aims to reproduce the target image with the artistic or photo-realistic styles provided. Even though existing approaches can successfully transfer style information, arbitrary style transfer still faces many challenges, such as the content leak issue. Specifically, the embedding of artistic style can lead to content changes. In this paper, we solve the content leak problem from the perspective of image restoration. In particular, an iterative architecture is proposed to achieve the Restorable Arbitrary Style Transfer (RAST), which can realize transmission of both content and style information through multi-restorations. We control the content-style balance in stylized images by the accuracy of image restoration. In order to ensure effectiveness of the proposed RAST architecture, we design two novel loss functions: multi-restoration loss and style difference loss. In addition, we propose a new quantitative evaluation method to measure content preservation performance and style embedding performance. Comprehensive experiments comparing with state-of-the-art methods demonstrate that our proposed architecture can produce stylized images with superior performance on content preservation and style embedding.

<div align=center>
<img src="https://github.com/xudongLi-Alex/RAST/blob/main/pic.png" width="1200" alt="Pipeline"/><br/>
</div>


## Requirements  
- python 3.8
- PyTorch 1.8.0
- CUDA 11.1


## Model Testing
- Download [VGG pretrained](https://drive.google.com/file/d/1cI6ubAziMdOsSJZEvfofW-iCtnCmsONL/view?usp=share_link) model to ./model/ folder/
- Put content images to *./content/* folder.
- Put style images to *./style/* folder.
- Run the following command:
```
python Eval.py --content_dir ./content --style_dir ./style
```
