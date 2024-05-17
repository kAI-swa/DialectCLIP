## Approach
__DialectCLIP is a LLM-based dialect speech recognition system which utilizes the constructed Speech-Dialect-Transcription triplet for end-to-end training, aligning the speech encoder with LLM in the feature space, and simultaneously processing audio signals and instructions.__

### Triple CLIP Framework
__We introduce a triple CLIP framework that jointly trains on the constructed Speech-Dialect-Transcription triplet__

![CLIP+EU](https://github.com/kAI-swa/DialectCLIP/assets/146005327/92956bc0-4ea2-4ab4-9b2b-ec0140ddee96)


### Grouping Layer
__The sparsity of speech features may affect the alignment between modalities and introduce a large amount of computational consumption. To mitigate this problem, we introduce a novel adapter module called Grouping Layer inspired by GroupViT__

![GL+CG](https://github.com/kAI-swa/DialectCLIP/assets/146005327/b3239162-68bd-4a21-a992-fe682b56a2eb)

## Tested environment
- **CPU: Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz**
- **Memory: 256 GB**
- **System: Ubuntu 20.04.5 LTS**
- **Python: 3.9.15**
