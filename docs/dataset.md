# CRAG-MM Dataset Documentation

**CRAG-MM (Comprehensive RAG Benchmark for Multi-modal, Multi-turn)** is a factual visual-question-answering corpus created to evaluate and train retrieval-augmented generation (RAG) systems in both **single-turn** and **multi-turn** settings.

Latest public release: **v0.1.2**
Available on Hugging Face:

| Modality    | URL                                                                                                                                                |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Single-turn | [https://huggingface.co/datasets/crag-mm-2025/crag-mm-single-turn-public](https://huggingface.co/datasets/crag-mm-2025/crag-mm-single-turn-public) |
| Multi-turn  | [https://huggingface.co/datasets/crag-mm-2025/crag-mm-multi-turn-public](https://huggingface.co/datasets/crag-mm-2025/crag-mm-multi-turn-public)   |

---

## 1  Dataset highlights

1. **Images** – A mix of egocentric photos captured with Ray-Ban | Meta Smart Glasses and openly licensed web imagery.
2. **Thirteen domains** – e.g. shopping, food, math & science.
3. **Diverse question types** – recognition, multi-hop reasoning, aggregation, comparison and more.
4. **Single- vs multi-turn** – one Q-A pair or extended dialogues about the same image.
5. **Quality variations** – normal, low-light, blurred and others.
6. **Phase 2 evaluation** – every egocentric image is down-sampled to **960 × 1280** before scoring; non-egocentric images retain their native resolution.

---

## 2  Data splits

Public release **v0.1.2** contains the **validation** split only.
The former *sample* split has been retired.

---

## 3  Data structure (v0.1.2)

### A note on the wrapper

In v0.1.2 the inner columns `turns` and `answers` are stored as **dictionary-of-columns** rather than list-of-dicts:

```text
# single-turn example
type(sample["turns"])   # -> <class 'dict'>
# multi-turn example
sample["turns"]["query"]        # -> list[str]  (length = number of turns)
sample["answers"]["ans_full"]   # -> list[str]  (same length)
```

### Single-turn schema

```jsonc
{
  "session_id": "string",
  "image": Image(),          // PIL.Image or None
  "image_url": "string",     // empty when 'image' is provided
  "turns": {
    "interaction_id": ["string"],
    "domain": ["int"],
    "query_category": ["int"],
    "dynamism": ["int"],
    "query": ["string"],
    "image_quality": ["int"]
  },
  "answers": {
    "interaction_id": ["string"],
    "ans_full": ["string"]
  }
}
```

### Multi-turn schema

```jsonc
{
  "session_id": "string",
  "image": Image(),
  "image_url": "string",
  "turns": {
    "interaction_id": ["string", ...],
    "domain": ["int", ...],
    "query_category": ["int", ...],
    "dynamism": ["int", ...],
    "query": ["string", ...],
    "image_quality": ["int", ...]
  },
  "answers": {
    "interaction_id": ["string", ...],
    "ans_full": ["string", ...]
  }
}
```

* `interaction_id` aligns each question with its answer.
* `domain`, `query_category`, `dynamism`, `image_quality` are integer-coded categorical labels.
* Either `image` **or** `image_url` is guaranteed to be present.

---

## 4  Quick access examples

```python
from datasets import load_dataset

# --- load datasets ---------------------------------------------------------
st = load_dataset("crag-mm-2025/crag-mm-single-turn-public",
                  split="validation", revision="v0.1.2")
mt = load_dataset("crag-mm-2025/crag-mm-multi-turn-public",
                  split="validation", revision="v0.1.2")

# --- iterate over turns, schema-proof --------------------------------------
def iter_turns(sample):
    """Yield (turn_dict, answer_dict) pairs in either single- or multi-turn rows."""
    if isinstance(sample["turns"], dict):
        n = len(sample["turns"]["interaction_id"])
        for i in range(n):
            turn =  {k: v[i] for k, v in sample["turns"].items()}
            ans  =  {k: v[i] for k, v in sample["answers"].items()}
            yield turn, ans
    else:  # older releases only
        for turn, ans in zip(sample["turns"], sample["answers"]):
            yield turn, ans

# inspect first multi-turn conversation
for t, a in iter_turns(mt[0]):
    print(f"Q: {t['query']}\nA: {a['ans_full']}\n")

# show the (possibly down-sampled) image
import matplotlib.pyplot as plt
plt.imshow(st[0]["image"])
plt.axis("off")
plt.show()
```

> **Tip for Phase 2:** if you feed raw pixels to your model, resize egocentric inputs to `width=960, height=1280` before preprocessing so your pipeline matches evaluation conditions.

---

## 5  No-code users—good news

If you rely on the provided `crag_batch_iterator.py` (updated in the repository), no code changes are required. The iterator transparently:

* Accepts both list-of-dicts (v0.1.1) and dict-of-columns (v0.1.2) layouts.
* Downloads images when only `image_url` is present.
* Resizes egocentric pictures to 960 × 1280.

Pull the latest commit and continue training.

---

## 6  License and citation

* **License:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0)
* **Citation:**

```bibtex
@inproceedings{crag-mm-2025,
  title  = {CRAG-MM: A Comprehensive RAG Benchmark for Multi-modal, Multi-turn Question Answering},
  author = {CRAG-MM Team},
  year   = {2025},
  url    = {https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025}
}
```
