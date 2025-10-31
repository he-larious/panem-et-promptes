# panem-et-promptes

This project implements a RAG pipeline for analyzing academic essays about *The Hunger Games* and evaluating the system’s retrieval and generation quality.

## 🧩 Setup & Installation

### Environment Setup
Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### API Key Setup
1. Get an OpenAI API key here: https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key" and copy the key that begins with `sk-....`
4. Duplicate the included `.env.example` file and rename it to `.env`
5. Paste your API key in the `OPENAI_API_KEY` field like so:
    ```bash
    OPENAI_API_KEY=sk-your-key-here
    ```

### Running the Program
To run the interactive QA system:
```bash
python main.py
```

To run the evaluation script:
```bash
python evaluate.py
```

## 🧠 Approach & Design Decisions
All design choices were made with the scope and scale of this project in mind (a handful of academic essays rather than a large, production level corpus).

### Pipeline Overview
1. **Preprocessing:** Each PDF is parsed with PyMuPDF. Cover pages, citations, and reference sections are detected and removed using regex-based heuristics to ensure only meaningful text is indexed.
2. **Chunking**: The essays are divided into fixed size word chunks (with overlap) to provide enough context for semantic retrieval while keeping inputs within the token limits of both the embedding model and LLM.
3. **Embedding & Indexing:** Each text chunk is embedded using SentenceTransformers (all-MiniLM-L6-v2), producing 384-dimensional semantic vectors. These vectors are normalized and stored in a FAISS Inner Product index, allowing the system to retrieve chunks based on cosine similarity.
4. **Retrieval & Generation:** When a user asks a question, the system embeds it and searches the FAISS index to find the most relevant chunks. The top k results are formatted into a structured prompt and passed to the OpenAI GPT model to generate a context-based answer.

### Design Decisions
- **Word-based chunking:** The essays were split into fixed size chunks of 250 words with a 50 word overlap to maintain context continuity across chunk boundaries. This approach ensures each chunk captures enough semantic context for retrieval while avoiding token overflow for the embedding model and LLM.
- **Cosine similarity (FAISS Inner Product):** The FAISS index uses inner product search on normalized vectors, which is mathematically equivalent to cosine similarity (since FAISS doesn't natively support it). This method measures directional closeness between embeddings instead of magnitude, which captures semantic relatedness more accurately.
- **Embedding model:** all-MiniLM-L6-v2 was selected for its balance of speed, size (384 dimensions), and accuracy. Larger models offer better semantic precision but introduce unnecessary computational overhead for a small dataset like this one.
- **LLM:** gpt-3.5-turbo was chosen for its efficiency and cost effectiveness in producing coherent answers.
- **Top 5 Retrieval:** The retriever returns the five most relevant chunks for each query. This value was high enough to capture potentially relevant sources, but small enough to avoid overwhelming the LLM with redundant or off topic context.
- **Retrieval threshold:** A similarity cutoff of 0.4 was applied to filter out weakly relevant chunks. This improves answer quality and reduces noise during prompt generation.
- **Prompt structure:** The generation prompt explicitly instructs the model to use only retrieved context and to respond with "The chunks were not in your favor." when relevant information is missing to minimize hallucination.
- **Page-level evaluation:** Precision and recall were computed at the page level rather than by individual chunks to reflect how sources are cited in academic writing.

## ⚙️ Familiar vs. New Technologies
| Category                       | Technologies                                             | Familiarity | Notes                                                                                 |
| ------------------------------ | -------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------- |
| **Core Language & Frameworks** | Python, OpenAI API                                       | Familiar  | Previously used for small projects and LLM applications.                            |
| **Concepts**                   | Retrieval Augmented Generation, Prompt Engineering | Familiar  | Comfortable designing prompts and integrating model outputs into pipelines.           |
| **Libraries & Tools**          | PyMuPDF (`fitz`), FAISS, SentenceTransformers            | New      | Learned how to parse and clean PDFs, create embeddings, and build a vector index for retrieval. |

## 📊 Evaluation
The system was evaluated on six questions and one control question using page level precision, recall, and cosine similarity between generated and expected answers.

### Results

| Metric                | Average Score |
| :-------------------- | :-----------: |
| **Cosine Similarity** |   0.69-0.73*  |
| **Precision@5**       |   0.71        |
| **Recall@5**          |   0.96        |

*The average score for cosine similarity across these 7 questions fell between a range of 0.69-0.73 across multiple runs of the evaluation script.

### Interpretation

The results had a high average recall which means that the relevant pages were consistently being retrieved within the top 5 results. The slightly lower precision shows that it also grabbed a few extra pages that weren’t as useful, probably because the essays overlap in topics and wording. The cosine similarity score shows that the generated answers still matched the expected ones pretty closely, so the model was using the right context well.

The system performed best on symbolic and thematic questions where key concepts are clearly reflected in the essays. However, it struggled with abstract questions (like comparing the Hunger Games to reality TV), despite retrieving relevant context for the question. Finally, the irrelevant control question (economic policies of District 14) behaved as expected, showing that the model knows when there isn’t relevant context.

Overall, the pipeline retrieves and applies information accurately. The few weaker answers were mostly due to the language model’s ability to connect ideas across multiple sources, not a flaw in the retrieval process itself.

## ⚠️ Known Limitations

## 🚀 Future Improvements
