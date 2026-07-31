\# TMCL Multimodal Consistency Verification Demo



Interactive demo of a two-tower multimodal model (ViT + BERT) for \*\*image–text consistency verification\*\* with visual and textual explanations.



\*\*Framework:\*\* TMCL / Align11  

\*\*Stage:\*\* Stage-1 (contrastive pretraining) + Stage-2 (explainability)



\---



\## Features



\- Upload an image + enter a textual description

\- Real-time consistency score (cosine similarity)

\- Decision: \*\*Consistent\*\* / \*\*Inconsistent\*\* (configurable threshold)

\- \*\*Visual Explanation\*\*: ViT attention heatmap overlay

\- \*\*Textual Explanation\*\*: BERT token importance ranking

\- Explanation summary + Risk interpretation guide

\- Verification history



\---



\## Project Structure



```text

tmcl-multimodal-consistency-demo/

├── app.py

├── model/

│   ├── tmcl.py

│   └── explain.py

├── utils/

│   └── image\_utils.py

├── examples/

├── config.yaml

├── requirements.txt

└── README.md

