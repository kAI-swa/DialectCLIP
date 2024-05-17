## Approach
__DialectCLIP is a LLM-based dialect speech recognition system which utilizes the constructed Speech-Dialect-Transcription triplet for end-to-end training, aligning the speech encoder with LLM in the feature space, and simultaneously processing audio signals and instructions.__

### Triple CLIP Framework
__We introduce a triple CLIP framework that jointly trains on the constructed Speech-Dialect-Transcription triplet__

![CLIP+EU_git](https://github.com/kAI-swa/DialectCLIP/assets/146005327/fafbee42-b567-4518-ac71-8e4bb1a75357)

### Grouping Layer
__The sparsity of speech features may affect the alignment between modalities and introduce a large amount of computational consumption. To mitigate this problem, we introduce a novel adapter module called Grouping Layer inspired by GroupViT__

![GL+CG_git](https://github.com/kAI-swa/DialectCLIP/assets/146005327/c438d0a8-6848-463f-8949-3ece8cb2bf39)


## Tested environment
- **CPU: Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz**
- **Memory: 256 GB**
- **System: Ubuntu 20.04.5 LTS**
- **Python: 3.9.15**
