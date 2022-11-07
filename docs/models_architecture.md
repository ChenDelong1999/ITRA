# Model Architechture

A summery of supported model architechtures:
- **Image Backbone**
    - From `OpenCLIP` (v2.0.2)

        - ResNet
        - Vision Transformer
        
        [OpenCLIP](https://github.com/mlfoundations/open_clip) is an open source implementation of [OpenAI's CLIP](https://github.com/openai/CLIP) (Contrastive Language-Image Pre-training). To check all supported model architecture and pretrained weigths, run:

        ```python
        >>> import open_clip
        >>> open_clip.list_pretrained()
        ```

    - From `Torchvision` [(v0.12)](https://pytorch.org/vision/0.12/)
        - AlexNet
        - ResNet
        - Mobilenet
        - EfficientNet
        - DenseNet
        - ConvNext
        - Vision Transformer
        - ...

        To check all supported model architecture and pretrained weigths, run the following command or see [this page](https://pytorch.org/vision/0.12/models.html)

        ```python
        >>> import torchvision
        >>> torchvision.models.__dict__.keys()
        ```

- **Text Backbone**
    - From `OpenCLIP`
        Choices of the text encoder is the same as OpenCLIP's image backbone.

    - 🤗 From Hugging Face Transformer    
        - BERT (`bert-base-uncased`)
        - RoBERTa (`roberta-base-cased`)
        - ...

            For more details, see [Hugging Face Transformers](https://huggingface.co/docs/transformers). Currently, only 'from pretrained' mode is supported (i.e., you cannot train a huggingface transformer from scratch now). 
            
            Standard models like BERT/RoBERTa are supported, but whether other models are also supported is not sure...

        - [Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html)
            - SBERT
            - Semantic Search Models
            - Word Embeddings (GloVe)

                Loading sentence transformers via huggingface and specify `--text-pooler='mean'` is recommended, though it is also supported to load the model via sentence transformer:

                ```bash
                # recommended: 
                --text-model-builder 'huggingface'  --text-model 'sentence-transformers/all-mpnet-base-v2' --text-pooler='mean' 
                # not recommended:
                --text-model-builder 'sbert'  --text-model 'all-mpnet-base-v2' 
                ```

        - Adapted Huggingface Transformer (via [Adapter-Transformers](https://github.com/adapter-hub/adapter-transformers))
            - [Bottleneck adapters](https://docs.adapterhub.ml/overview.html#bottleneck-adapters)
            - [Language Adapters](https://docs.adapterhub.ml/overview.html#language-adapters-invertible-adapters) 
            - [Prefix Tuning](https://docs.adapterhub.ml/overview.html#prefix-tuning)
            - [Compacter](https://docs.adapterhub.ml/overview.html#compacter)
            - [LoRA](https://docs.adapterhub.ml/overview.html#lora)
            - [(IA)^3](https://docs.adapterhub.ml/overview.html#ia-3)
            - [Mix-and-Match   Adapters](https://docs.adapterhub.ml/overview.html#mix-and-match-adapters)
            - [UniPELT](https://docs.adapterhub.ml/overview.html#unipelt)
            - ...

                [Adapter-Transformers](https://github.com/adapter-hub/adapter-transformers) is an extension of [HuggingFace's Transformers](https://github.com/huggingface/transformers) library, integrating adapters into state-of-the-art language models by incorporating [AdapterHub](https://adapterhub.ml/), a central repository for pre-trained adapter modules. 
                
                For more details, see: [Docs](https://docs.adapterhub.ml/) | [Model Overview](https://docs.adapterhub.ml/model_overview.html)

                | Method                                                                                                        | args.adapter       |         |
                |---------------------------------------------------------------------------------------------------------------|--------------------|------------|
                | [Bottleneck   adapters](https://docs.adapterhub.ml/overview.html#bottleneck-adapters)                         | `bottleneck_adapter` |          |
                | [Language Adapters](https://docs.adapterhub.ml/overview.html#language-adapters-invertible-adapters)           | `lang_adapter`       |          |
                | [Prefix   Tuning](https://docs.adapterhub.ml/overview.html#prefix-tuning)                                     | `prefix_tuning`      |          |
                | [Compacter](https://docs.adapterhub.ml/overview.html#compacter)                                               | `dummy`              |          |
                | [LoRA](https://docs.adapterhub.ml/overview.html#lora)                                                         | `lora_adapter`       |          |
                | [(IA)^3](https://docs.adapterhub.ml/overview.html#ia-3)                                                       | `ia3_adapter`        |          |
                | [Mix-and-Match   Adapters](https://docs.adapterhub.ml/overview.html#mix-and-match-adapters)                   | `mam_adapter`        |          |
                | [UniPELT](https://docs.adapterhub.ml/overview.html#unipelt)                                                   | `unipelt`            |          |


- **Projection Head**
    - Linear projection head

    - [DINO MLP Head](https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/vision_transformer.py#L257) (optionally with a prototype layer in the last)



