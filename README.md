# LongRAG
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kiendoantrung/LongRAG)

LongRAG retrieves large tokens at a time, with each retrieval unit being ~6k tokens long, consisting of entire documents or groups of documents. This contrasts the short retrieval units (100 word passages) of traditional RAG. LongRAG is advantageous because results can be achieved using only the top 4-8 retrieval units, and long-context LLMs can better understand the context of the documents because long retrieval units preserve their semantic integrity.

## Installation

```bash
git clone https://github.com/kiendoantrung/LongRAG.git
```

```bash
cd LongRAG
```

```bash
pip install -e .
```

## Usage

```bash
python src/main.py
```

## Endpoints

### Query Endpoint

- **URL**: `http://localhost:8000/api/query`
- **Method**: `POST`
- **Content-Type**: `application/json`

Example using `curl`:

```sh
curl -X POST "http://localhost:8000/api/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Your query"}'
```

## References

- [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs](https://arxiv.org/pdf/2406.15319)

- [LongRAG using LlamaIndex workflows](https://docs.llamaindex.ai/en/stable/examples/workflow/long_rag_pack)

