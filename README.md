# Weekly-Progress-Report
王鑫煜-周报 
# 2023-12-18
## Progress for Past Week  
1.Read three papers:  
(1)"HIGNN-TTS: HIERARCHICAL PROSODY MODELING WITH GRAPH NEURAL NETWORKS FOR EXPRESSIVE LONG-FORM TTS"    
[https://doi.org/10.1109/PowerTech55446.2023.10202713](https://doi.org/10.48550/arXiv.2309.13907)  
In this paper, a hierarchical prosodic modeling method named HiGNN-TTS is proposed to achieve high expressive speech synthesis of long-form speech. This method uses graph neural network (GNN) to assist speech synthesis. By extending the hierarchical prosody modeling capability of GNN, including the word, single sentence and cross-sentence levels, high expressive long speech synthesis is realized. Specifically, the method adds virtual global nodes to the syntax tree, enhancing the connection relationship between word nodes.   
(2)"Attention Is All You Need"   
https://doi.org/10.48550/arXiv.1706.03762  
Attention is all you need, a paper presented at the NIPS 2017 conference, swept through the natural language processing community as quickly as Mars hit Earth, and quickly replaced the recurrent neural network family as the standard in future language models. For example, we are familiar with the GPT(generative pre-trained model) series models and BERT(Bidirectional encoder representation from transformer) series models, which are inspired by this article to adopt part of the transformer architecture, and at the same time, based on a large number of text pre-training, thus achieving a series of breakthrough effects.    
(3)"Denoising Diffusion Probabilistic Models"    
https://doi.org/10.48550/arXiv.2006.11239  
This paper is the foundation work of DDPM and is one of the most classic papers in the field. In fact, diffusion model is not a new concept, this paper is the first to give a rigorous mathematical derivation, can be repeated code, improve the whole reasoning process. Later papers about diffusion models basically inherit the system of forward plus noise - reverse noise reduction - training.  
2.Make a real-time whisper model   
## Plan for Coming Week  
1.Adjust the real_time model to make it smoother, make latency less than 5s or even less  
2.Use the medical audio data to test this model

# 2023-12-10
## Progress for Past Week  
1.Read three papers:  
(1)"Denoising Diffusion Probabilistic Models"  
https://doi.org/10.1109/PowerTech55446.2023.10202713  
Diffusion model is not a new concept, this paper is the first to give a rigorous mathematical derivation, can be repeated code, improve the entire reasoning process. Later papers about diffusion models basically inherit the system of forward plus noise - reverse noise reduction - training.  
(2)"High-Resolution Image Synthesis with Latent Diffusion Models"  
[10.48550/arXiv.2112.10752](https://arxiv.org/abs/2112.10752)  
There are two key points worth noting in this paper: First, the encoder decoder is used to reduce the operation to the latent domain, and it returns to the most classic structure in the generation field, operating on the latent domain (i.e. z), which is also commonly used in vae. The second is the structure of cross-attention, which was used as early as the handwriting diffusion paper in 2020, but it did not attract widespread attention at the time. Since then cross-attention has become a common approach to multimodality and a new common conditional diffusion model.  
(3)"High-resolution image reconstruction with latent diffusion models from human brain activity"  
[10.1101/2022.11.18.517004 ](https://www.biorxiv.org/content/10.1101/2022.11.18.517004v2)   
In terms of model architecture, the original LDM model is introduced first. Then, the corresponding relationship between the feature vectors extracted from different data sources and the LDM module and the model training process are introduced. And the corresponding relationship between the characteristics of different noise conditions, different diffusion stages, different U-Net layers and brain parts was compared.
2.Try to trans the whisper to real-time whisper  
## Plan for Coming Week  
1.Make a real-time whisper model

# 2023-12-3
## Progress for Past Week
1.Read papers about Vits
2.Combine the speech data and finetune the whisper model
## Plan for Coming Week
1.Analyse the reason for the high CER in this model

# 2023-11-27
## Progress for Past Week
1.Read papers about Vits
2.Write GUI for whisper
## Plan for Coming Week
1.Combine the speech data
2.Finetune the Whisper with the data

# 2023-11-19
## Progress for Past Week
1.Read papers about Vits
2.Finetune vits models 
## Plan for Coming Week
1.Write GUI for whisper

# 2023-11-12
## Progress for Past Week
1.Summarizing the TTS prior development. 
2.Run couqi program.
## Plan for Coming Week
1.Run so-vits-vec program.

# 2023-11-5
## Progress for Past Week
1.Read a paper Scaling Laws for Neural Machine Translation. 
2.Learn the principles of TTS.
## Plan for Coming Week
1.Learn the usage of TTS and ASR.

# 2023-10-29
## Progress for Past Week
1.Read papers about Whisper.
2.Combine the speech from medical txt.
## Plan for Coming Week
1.Learn the usage of Audio diagram.

# 2023-10-23
## Progress for Past Week
1.Fine-tuning whisper code and evalute the cer and wer of finetune-model.
## Plan for Coming Week
1.Get the speech data of medical field.
2.Evalute the difference between Large-V2 model and finetune-model.

# 2023-10-15
## Progress for Past Week
1.Fine-tuning whisper code
2.get train-test data about ASR
## Plan for Coming Week
1.Evaluate the fine-tuned code

# 2023-10-9
## Progress for Past Week
1.Read some papers on ASR in depression
2.Investigate the development status of ASR in disease diagnosis
## Plan for Coming Week
1.Determine the research direction of ASR in disease diagnosis

# 2023-10-2
## Progress for Past Week
1.Read some papers on voice-to-word
2.Investigate the development status of speech-to-text
3.Configure the environment of openpose project
## Plan for Coming Week
1.Read more papers on voice-to-word
2.Determine the research program I want to carry out


