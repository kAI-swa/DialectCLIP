## Approach
__DialectCLIP is a LLM-based dialect speech recognition system which utilizes the constructed Speech-Dialect-Transcription triplet for end-to-end training, aligning the speech encoder with LLM in the feature space, and simultaneously processing audio signals and instructions.__

### Triple CLIP Framework
__We introduce a triple CLIP framework that jointly trains on the constructed Speech-Dialect-Transcription triplet__
![triple_clip](https://github.com/kAI-swa/DialectCLIP/assets/146005327/2f9fe949-d90b-412e-80c7-bc749b76dc9f)

### Grouping Layer
__The sparsity of speech features may affect the alignment between modalities and introduce a large amount of computational consumption. To mitigate this problem, we introduce a novel adapter module called Grouping Layer inspired by GroupViT__
![temp_GLCG](https://github.com/kAI-swa/DialectCLIP/assets/146005327/20f9b981-9f36-45df-9e07-e8a1cd735c8e)

## Tested environment
- **CPU: Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz**
- **Memory: 256 GB**
- **System: Ubuntu 20.04.5 LTS**
- **Python: 3.9.15**
