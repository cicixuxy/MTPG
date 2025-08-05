# "Learning Artistic Image Aesthetics from Multi-level Text Prompts Generation".

## Abstract
Artistic Image Aesthetics Assessment (AIAA) aims to emulate human artistic perception to evaluate the aesthetics of artistic images. Due to the highly specialized nature of human artistic perception, obtaining large-scale aesthetic annotations for model analysis presents significant challenges. Furthermore, the subjectivity of artistic aesthetics makes it difficult for existing AIAA methods to quantify aesthetic scores solely based on visual features. To address the two challenges, we introduce an AIAA model based on multi-level text prompts generation. Firstly, we leverage a text prompt-based self-supervised learning approach to augment artistic image data and adopt a multi-task learning paradigm to pretrain our multi-modal AIAA model. To further capture the abstract aesthetic characteristics of artistic images, we then adopt a domain-specific multi-modal large language model(MLLM) to simulate human artistic perception and generate aesthetic textual descriptions for images, and employ a multi-modal fusion module to integrate image features with the text features of artistic aesthetics for better feature representation. Finally, the proposed multi-modal AIAA model is trained by text-prompt learning based on the aesthetic quality levels of artistic images. By generating multi-level text prompts, our method can introduce human-perceived artistic aesthetic knowledge and obtain a more effective AIAA model. Experimental results on several AIAA datasets demonstrate that our method is superior to the state-of-the-art AIAA methods. 

## Dataset
The test dataset with art-level desccriptions can be downloaded on:[test dataset]( https://drive.usercontent.google.com/download?id=1jgxjCo1yOQXuhhWmYyN_Gkh3_ajubsak)

If you want to generate descriptions by yourself, you should download BAID database [Link](https://github.com/Dreemurr-T/BAID)  and use the domain-specific MLLM called GalleryGPT to generate.

## Code
### Requirements

* python == 3.8.19
* torch == 2.4.0
* torchvision == 0.19.0

Install the necessary dependencies using:

```sh
pip install -r requirements.txt
```

* demo.py for predicting the scores of two artistic images by using MTPG-AIAA model.

* test.py for model testing.

* The test dataset with art-level desccriptions can be downloaded on:[test dataset]( https://drive.usercontent.google.com/download?id=1jgxjCo1yOQXuhhWmYyN_Gkh3_ajubsak)

* The longclip parameters can be downloaded on: [longclip](https://drive.usercontent.google.com/download?id=1bDKBAqCnvMeEXKecMBB6UCgxj8aglPPq)

  
* The checkpoint can be downloaded on:[checkpoint]( https://drive.usercontent.google.com/download?id=19uSUuZ_5jCfgKzLBqAERUgmAv9Yd8Oci)









