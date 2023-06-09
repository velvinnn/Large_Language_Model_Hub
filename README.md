# Large_Language_Model_Hub
Papers, tutorials, code for large language models

# Table of Contents
1. [Key Language Models](#key-language-models)
2. [LLM surveys](#llm-surveys)
3. [Training and Optimization](#training-and-optimization)
4. [Evaluation and benckmarks](#evaluation-and-benckmarks)
5. [Qualitative evaluation](#qualitative-evaluation)
6. [Hallucination](#hallucination)
7. [Prompt-tuning](#prompt-tuning)
8. [ChatGPT / GPT-4](#chatgpt--gpt-4)
9. [Augmented LLM](#augmented-llm)
10. [Retrieval-base NLP](#retrieval-base-nlp)
11. [Clinical Applications](#clinical-applications)
12. [Knowledge Distillnation](#knowledge-distillation)
13. [Machine Translation](#machine-translation)
14. [Open Source Data](#open-source-data)
15. [Safety](#safety)
16. [Learning resources](#learning-resources)
17. [Analysis and Inspection](#analysis-and-inspection)
18. [Discussions](#discussions)


## Key language models
- [A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT](https://arxiv.org/pdf/2302.09419.pdf)
- [A Comprehensive Survey of AI-Generated Content (AIGC): A History of Generative AI from GAN to ChatGPT](https://arxiv.org/abs/2303.04226)
- [T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
  - [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5)
- [GPT3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
  - [Stanford Webinar - GPT-3 & Beyond](https://www.youtube.com/watch?v=-lnHHWRCDGk) ([Slides](https://docs.google.com/presentation/d/1WPYaLEEVJJI_-DOzjudeVoYpl_y0yUi1kWs0VFBnba4/edit#slide=id.g1c79e641885_1_554))
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) 
  - [Vedio with Paper Explained](https://www.youtube.com/watch?v=E5OnoYF2oAk)
  - [LLaMA-Adapter: Efficient Fine-tuning of LLaMA](https://github.com/ZrrSkywalker/LLaMA-Adapter)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)
  - [Pathways Language Model and Model Scaling - Aakanksha Chowdhery | Stanford MLSys #69](https://www.youtube.com/watch?v=CV_eBVwzOaw)
- [FLAN: Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)
- [Flan-PaLM: Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)
  - Paper explained with related work: [video](https://www.youtube.com/watch?v=QdwETwqyREY), [slides](https://samuelalbanie.com/files/digest-slides/2022-10-scaling-instruction-finetuned-language-models.pdf)
- [Primer: Searching for Efficient Transformers for Language Modeling](https://arxiv.org/abs/2109.08668)
- [CodeGen2: Lessons for Training LLMs on Programming and Natural Languages](https://arxiv.org/pdf/2305.02309.pdf)
- [Unlimiformer: Long-Range Transformers with Unlimited Length Input](https://arxiv.org/abs/2305.01625)
- [LaMP: When Large Language Models Meet Personalization](https://arxiv.org/abs/2304.11406)

## LLM surveys
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223)
- [Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond](https://arxiv.org/abs/2304.13712)
- 

## Training and optimization
- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
  - [Explanation vedio: How ChatGPT is trained](https://www.youtube.com/watch?v=VPRSBzXzavo)
- [The Large Language Model Training Playbook](https://github.com/huggingface/large_language_model_training_playbook)
- [Scaling Expert Language Models with Unsupervised Domain Discovery](https://arxiv.org/abs/2303.14177)
- [Summary of open-source APIs, tools and services for LLM](https://github.com/kasperjunge/LLM-Guide)
- [UL2: Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131)
- [Glam: Efficient scaling of language models with mixture-of-experts](https://proceedings.mlr.press/v162/du22c.html)
- [State-of-the-art Parameter-Efficient Fine-Tuning (PEFT) methods](https://github.com/huggingface/peft/tree/main)
  - [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
  - Adaptive LoRA: [Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)
  - [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/)
  - [P-Tuning: Prompt Tuning Can Be Comparable to Fine-tuning Across Scales and Tasks](https://aclanthology.org/2022.acl-short.8/)
  - Prompt tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)


## Evaluation and benckmarks
- [BIG-bench: Beyond the Imitation Game benchmark](https://arxiv.org/abs/2206.04615)
- [MMLU: Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
- [TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages](https://arxiv.org/abs/2003.05002)
- [Language Models are Multilingual Chain-of-Thought Reasoners](https://arxiv.org/abs/2210.03057)
- [GPTEval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634)
- [Open AI Evals](https://github.com/openai/evals)

## Qualitative evaluation
- [Emergent abilities of large language models](https://www.jasonwei.net/blog/emergence)
- [Larger language models do in-context learning differently](https://arxiv.org/abs/2303.03846)
- [Capabilities of GPT-4 on Medical Challenge Problems](https://arxiv.org/abs/2303.13375)
- [Zero-shot Clinical Entity Recognition using ChatGPT](https://arxiv.org/pdf/2303.16416v1.pdf)
- [ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks](https://arxiv.org/abs/2303.15056)
- [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/pdf/2304.15004.pdf)
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)

## Hallucination
- [Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629)
- [The Internal State of an LLM Knows When its Lying](https://arxiv.org/abs/2304.13734)


## Prompt-tuning
- [The Art of Asking ChatGPT for High-Quality Answers: A complete Guide to Prompt-Engineering Technique](https://www.amazon.com/Art-Asking-ChatGPT-High-Quality-Answers/dp/B0BT2JB67Y)
- [Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)
  - [1-hour tutorial](https://www.youtube.com/watch?v=dOxUroR57xs) ([Slides](https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/lecture/Prompt-Engineering-Lecture-Elvis.pdf))
- [Large language models are implicitly topic models: Explaining and finding good demonstrations for in-context learning](https://arxiv.org/pdf/2301.11916.pdf)
- [AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts](https://arxiv.org/abs/2010.15980)
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190L)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Large Language Models Can Self-Improve](https://arxiv.org/abs/2210.11610)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
- [Active Prompting with Chain-of-Thought for Large Language Models](https://arxiv.org/abs/2302.12246) ([code](https://github.com/shizhediao/active-prompt))
- [Boosting Theory-of-Mind Performance in Large Language Models via Prompting](https://arxiv.org/abs/2304.11490)



## ChatGPT / GPT-4
- [GPT-4 Technical Report](https://cdn.openai.com/papers/gpt-4.pdf)
- [Sparks of Artificial General Intelligence: Early experiments with GPT-4](https://arxiv.org/abs/2303.12712v1)
  - [Talk by the first author](https://www.youtube.com/watch?v=qbIk7-JPB2c&ab_channel=SebastienBubeck)
- [GPT-4 developer demo livestream](https://www.youtube.com/watch?v=outcGtbnMuQ&ab_channel=OpenAI)
- [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace](https://arxiv.org/abs/2303.17580)

## Augmented LLM
- [Augmented Language Models: a Survey](https://arxiv.org/abs/2302.07842)
- [Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback](https://arxiv.org/abs/2302.12813)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)

## Retrieval-base NLP
- [Memorizing Transformers](https://arxiv.org/abs/2203.08913)
- [Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP](https://arxiv.org/abs/2212.14024)
- [Chat-gpt retrieval plugin](https://github.com/openai/chatgpt-retrieval-plugin)



## Clinical applications
- [Open source diagnosis generator](https://glass.health/ai)
- [ChatDoctor: A Medical Chat Model Fine-tuned on LLaMA Model using Medical Domain Knowledge](https://arxiv.org/abs/2303.14070v1)
- [Med-PaLM: Large Language Models Encode Clinical Knowledge](https://arxiv.org/abs/2212.13138)
- [Clinical-T5: Large Language Models Built Using MIMIC Clinical Text](https://physionet.org/content/clinical-t5/1.0.0/)
- [Large AI Models in Health Informatics: Applications, Challenges, and the Future](https://arxiv.org/abs/2303.11568)
- [Performance of ChatGPT on USMLE: Potential for AI-assisted medical education using large language models](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9931230/)

## Knowledge Distillnation
- [Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes](https://arxiv.org/abs/2305.02301)

## Machine Translation
- [Pseudo-label Training and Model Inertia in Neural Machine Translation](https://urldefense.com/v3/__https://openreview.net/forum?id=eXkhH12DTD9__;!!K-Hz7m0Vt54!mPbn2m9VGpvdlbDmyEweve8305fS_1D7hjvSKEJmo1VfUX1R4ZEEoDcuLiuBPFYWp4_7i-57OXdS7YCdnQ$)
- [Evaluating Robustness to Input Perturbations for Neural Machine Translation](https://urldefense.com/v3/__https://aclanthology.org/2020.acl-main.755/__;!!K-Hz7m0Vt54!mPbn2m9VGpvdlbDmyEweve8305fS_1D7hjvSKEJmo1VfUX1R4ZEEoDcuLiuBPFYWp4_7i-57OXfrGmdqAw$)
- [CoCoA-MT: A Dataset and Benchmark for Contrastive Controlled MT with Application to Formality](https://urldefense.com/v3/__https://arxiv.org/abs/2205.04022__;!!K-Hz7m0Vt54!mPbn2m9VGpvdlbDmyEweve8305fS_1D7hjvSKEJmo1VfUX1R4ZEEoDcuLiuBPFYWp4_7i-57OXdp5rrfDg$)
- [MT-GenEval: A Counterfactual and Contextual Dataset for Evaluating Gender Accuracy in Machine Translation](https://urldefense.com/v3/__https://arxiv.org/abs/2211.01355__;!!K-Hz7m0Vt54!mPbn2m9VGpvdlbDmyEweve8305fS_1D7hjvSKEJmo1VfUX1R4ZEEoDcuLiuBPFYWp4_7i-57OXccTO29IQ$)
- [How Good Are GPT Models at Machine Translation? A Comprehensive Evaluation](https://urldefense.com/v3/__https://arxiv.org/pdf/2302.09210.pdf__;!!K-Hz7m0Vt54!mPbn2m9VGpvdlbDmyEweve8305fS_1D7hjvSKEJmo1VfUX1R4ZEEoDcuLiuBPFYWp4_7i-57OXd36gwjEg$)
- [Large language models effectively leverage document-level context for literary translation, but critical errors persist](https://urldefense.com/v3/__https://arxiv.org/pdf/2304.03245.pdf__;!!K-Hz7m0Vt54!mPbn2m9VGpvdlbDmyEweve8305fS_1D7hjvSKEJmo1VfUX1R4ZEEoDcuLiuBPFYWp4_7i-57OXdCqkwFXQ$)
- [No Language Left Behind: Scaling Human-Centered Machine Translation](https://urldefense.com/v3/__https://arxiv.org/abs/2207.04672__;!!K-Hz7m0Vt54!mPbn2m9VGpvdlbDmyEweve8305fS_1D7hjvSKEJmo1VfUX1R4ZEEoDcuLiuBPFYWp4_7i-57OXdP6YrABA$)


## Open Source Data
Large open source datasets
- [RedPajama-Data: An Open Source Recipe to Reproduce LLaMA training dataset](https://github.com/togethercomputer/RedPajama-Data): several datasets, trillions of tokens, in comparison with LLaMA
- [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k): Instruction pairs (>15k)
- [OpenAssistant Conversations Dataset (OASST1)](https://huggingface.co/datasets/OpenAssistant/oasst1): >10k conversation trees, with human-in-the-loop style annotations
- [LongForm: Optimizing Instruction Tuning for Long Text Generation with Corpus Extraction](https://github.com/akoksal/LongForm): Instruction style

## Safety
- [NeMo-Guardrails by NVIDIA](https://github.com/NVIDIA/NeMo-Guardrails/tree/main): Topics, Moderation, Fact Checking and Hallucination, Secure Execution, Jailbreak, etc.
- [A Watermark for Large Language Models](https://arxiv.org/pdf/2301.10226.pdf)

## Learning resources
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [Reasoning in Large Language Models](https://github.com/jeffhj/LM-reasoning)
- [A set of AI tools](https://github.com/meetpateltech/AI-Infinity)
- [LLM summaries and tutorials: Awesome-LLM Awesome](https://github.com/Hannibal046/Awesome-LLM)
- [Stanford CS324: Large Language Models](https://stanford-cs324.github.io/winter2022/)
- [Princeton COS597G(Fall 2022): Understanding Large Language Models](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/)
- [Princeton COS 597F: Embodied Language Understanding](https://sites.google.com/princeton.edu/cos597f)


## Analysis and Inspection
- [Memorization Without Overfitting: Analyzing the Training Dynamics of Large Language Models](https://openreview.net/forum?id=u3vEuRr08MT)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

## Discussions
- [Yann LeCun: do large language models need sensory grounding for meaning and understanding](https://drive.google.com/file/d/1BU5bV3X5w65DwSMapKcsr0ZvrMRU_Nbi/view)
- [Full interview: "Godfather of artificial intelligence" talks impact and potential of AI](https://www.youtube.com/watch?v=qpoRO378qRY)
- [Sam Altman: OpenAI CEO on GPT-4, ChatGPT, and the Future of AI | Lex Fridman Podcast #367](https://www.youtube.com/watch?v=L_Guz73e6fw)
- [Whose Opinions Do Language Models Reflect?](https://arxiv.org/abs/2303.17548)
- [The Low-Rank Simplicity Bias in Deep Networks](https://arxiv.org/abs/2103.10427)
