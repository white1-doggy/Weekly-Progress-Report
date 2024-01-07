# Weekly-Progress-Report
王鑫煜-周报  
# 2023-1-7
## Progress for Past Week  
1.Read three papers:  
(1)"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"        
https://doi.org/10.48550/arXiv.1810.04805      
BERT, which stands for Bidirectional Encoder Representation from Transformers, is a pre-trained representation model of language. It emphasizes that masked language model (MLM) is not used for pre-training by traditional one-way language model or shallow combination of two one-way language models as before, but a new Masked language model (MLM) is used to generate deep two-way language representations. BERT's paper was published with new state-of-art results for 11 NLP (Natural Language Processing) tasks.         
(2)"Improving Language Understanding by Generative Pre-Training"       
[https://doi.org/10.48550/arXiv.1703.10135](https://www.mikecaptain.com/resources/pdf/GPT-1.pdf)        
This paper uses Generative Pre-Training (GPT) to improve language understanding. Existing Natural Language Processing (NLP) models often require a lot of Supervised Learning for specific tasks, ignoring the richness and diversity of the language itself. Therefore, a new approach is to pre-train a general-purpose language generation model with large-scale Unlabeled Text, and then fine-tune the model with small amounts of Labeled Data for different types of language understanding tasks.           
(3)"Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions"      
https://doi.org/10.48550/arXiv.1712.05884   
The architecture of Tacotron 2 is based on a combination of two main components: Tacotron and WaveNet. Tacotron is responsible for converting text into spectrograms, while WaveNet is responsible for converting spectrograms into high-quality speech waveforms. This end-to-end structure enables Tacotron 2 to generate speech directly from text without additional intermediate steps.Tacotron 2 uses an attention mechanism, which means that it can pay different degrees of attention to different parts of the input text during the generation of the spectrogram to improve the naturalness and fluency of speech.      
2.Adjust the real_time model to make it smoothe. The latency have reached levels that are slightly higher than those of commercial products.       
## Plan for Coming Week  
1.Solve the question analyze of the low recognition rate. Device issues or python library issues?  

# 2023-12-29
## Progress for Past Week  
1.Read three papers:  
(1)"FastSpeech 2: Fast and High-Quality End-to-End Text to Speech"        
https://doi.org/10.48550/arXiv.2006.04558    
First of all, the author abandoned the teacher-student training of knowledge distillation and adopted the way of training directly on the ground-truth. Secondly, more inputs that can control speech are introduced into the model, including phoneme duration mentioned in FastSpeech, energy, pitch and other new quantities. The authors name this model FastSpeech2. The author builds on this to propose the FastSpeech2s, a model that can generate speech directly from text instead of mel-spectrogram. The experimental results prove that FastSpeech2 is trained 3 times faster than FastSpeech, and FastSpeech2s has a faster synthesis speed than other models. In terms of sound quality, both FastSpeech2 and 2s surpass previous auto-regressive models.    
(2)"Tacotron: Towards End-to-End Speech Synthesis"    
https://doi.org/10.48550/arXiv.1703.10135      
The Tacotron model is the first end-to-end TTS deep neural network model. Compared with traditional speech synthesis, it does not have complex phonetics and acoustic feature modules. Tacotron is an attention-based seq2seq model. These include encoders, attention-based decoders, and post-processing networks. In a high-level perspective, our model takes characters as input and generates an idiom spectrum, which is then converted into a waveform.        
(3)"Toolformer: Language Models Can Teach Themselves to Use Tools"        
https://doi.org/10.48550/arXiv.2302.04761        
In this paper, the author proposes Toolformer to fine-tune the language model in a self-supervised manner, so that the model learns to automatically call the API without losing the commonality of the model. By invoking a range of tools, including calculators, question answering systems, search engines, translation systems, and calendars, Toolformer achieves substantially improved zero-sample performance in a variety of downstream tasks, often competing with larger models without sacrificing its core language modeling capabilities.         
2.Alleviates the problem of low real-time audio quality.The whisper model has a hallucination problem.                
## Plan for Coming Week  
1.There is no consideration of context understanding in the code logic, which will result in the low quality of the transcribed text. Try to solve the problem.               

# 2023-12-24
## Progress for Past Week  
1.Read three papers:  
(1)"FastSpeech: Fast, Robust and Controllable Text to Speech"    
https://doi.org/10.48550/arXiv.1905.09263   
FastSpeech, an end-to-end TTS model based on Transformer. Traditional end-to-end TTS models such as Tacotron2 are slow to generate speech because of the use of auto-regressive architecture. In order to speed up the calculation, the author constructs a model based on Transformer, thus realizing the parallel generation of mel-spectrogram. The experimental results show that FastSpeech is 270 times faster in mel-spectrogram generation and 38 times faster in speech generation than the traditional end-to-end TTS model, and has no influence on speech quality.      
(2)"Learning Transferable Visual Models From Natural Language Supervision"   
https://doi.org/10.48550/arXiv.2103.00020   
This paper trains a transferable visual model from natural language, which is of great significance to the subsequent development of many fields, including object detection, image generation, video retrieval, zero-shot learning, etc., and even includes the recently popular diffusion model. There are also some work combining CLIP and diffusion model. In addition, OpenAI open source the trained model and API (https://github.com/openai/CLIP), which can be directly used for downstream tasks of reasoning, recently very fire image generation field, there are a lot of work to use the trained CLIP model.    
(3)"LoRA: Low-Rank Adaptation of Large Language Models"    
https://doi.org/10.48550/arXiv.2106.09685     
LoRA (Low-Rank Adaptation) is a method for fine-tuning large language models. Its core idea is to fine-tune the model by adding a low-rank adjustment to the original weight of the model. This approach effectively reduces the number of parameters required for fine tuning while maintaining model performance.   
2.Adjust the real_time model to make it smoother, make latency less than 5s or even less      
## Plan for Coming Week  
1.Keep finetuneing the real_time model, analyze the reason for the low recognition rate    

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


