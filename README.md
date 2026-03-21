# Toastd Product Recommendation System

A semantic search and recommendation system for [Toastd.in](https://toastd.in) products. Uses sentence embeddings for similarity matching and optionally leverages Google's Gemini for query understanding and result reranking.

Built this as a way to explore how modern search systems work beyond simple keyword matching.

---

## Table of Contents

1. [Overview](#overview)
2. [How the Data Was Collected](#how-the-data-was-collected)
3. [Data Organization](#data-organization)
4. [The Search Pipeline](#the-search-pipeline)
5. [Caching with Redis](#caching-with-redis)
6. [Fast Mode vs Normal Mode](#fast-mode-vs-normal-mode)
7. [Project Structure](#project-structure)
8. [Setup Instructions](#setup-instructions)
9. [API Reference](#api-reference)
10. [Tradeoffs and Considerations](#tradeoffs-and-considerations)

---

## Overview

The goal was to build a recommendation system that actually understands what users are looking for, not just match keywords. When someone searches "gifts for my girlfriend", the system should understand they probably want jewelry, accessories, or beauty products - not literally items with the word "gift" in the name.

The system works in two modes:
- **Normal Mode**: Uses Gemini to expand queries and rerank results (better quality, slower)
- **Fast Mode**: Uses only embeddings for similarity matching (good quality, much faster)

---

## How the Data Was Collected

The product catalog was scraped from Toastd's public API. The whole process is documented in `experiments/fafo01.ipynb` (yes, the name is intentional - it was exploratory work).

### API Discovery

Found the API endpoint by inspecting network requests on the Toastd website:

```
GET https://api.toastd.in/api/product
```

The API supports pagination with `page` and `size` parameters, and sorting by `viewCount:desc` to get trending products first.

### Data Extraction

For each product, we extracted:
- Basic info: id, name, title, slug, descriptions
- Pricing: original price, discount percentage, final price, discount codes
- Engagement metrics: view count, vote count
- Media: images, videos, thumbnails
- Categorization: tags, category ID, brand ID
- Metadata: creation date, shipping details, return policy

The scraper handles nested JSON structures (prices are stored as `{amount: value}` objects) and cleans up the data into a flat structure.

### What We Got

```
Total products: 466
Products with discounts: 347
Unique brands: 55
Unique categories: 16
Price range: ₹40 - ₹25,000
Total views: 1,380,800
Products with tags: 389 (83.5%)
```

---

## Data Organization

The raw data was split into multiple specialized CSVs for different use cases:

| File | Purpose | Records |
|------|---------|---------|
| `toastd_products.csv` | Complete data (38 columns) | 466 |
| `toastd_text_corpus.csv` | Text fields for NLP | 466 |
| `toastd_tags.csv` | Tags for content filtering | 389 |
| `toastd_features.csv` | Numerical features for ML | 466 |
| `toastd_engagement.csv` | View/vote metrics | 466 |
| `toastd_media.csv` | Image/video URLs | 466 |

For the search system, we use a cleaned JSON file (`src/products.json`) with only the fields needed:

```json
{
  "id": "...",
  "name": "Product Name",
  "title": "Tagline",
  "slug": "url-slug",
  "short_description": "...",
  "headline_description": "...",
  "meta_description": "...",
  "price": 1999.0,
  "tags": "Tag1|Tag2|Tag3",
  "main_image": "https://...",
  "view_count": 1234,
  "vote_count": 56,
  "product_url": "https://toastd.in/product/..."
}
```

---

## The Search Pipeline

### Step 1: Embedding Generation (Offline)

Before the system can search, we need to convert all product text into vector embeddings. This is done once using `prod_embeddings.py`.

**Model**: `all-MiniLM-L6-v2` from Sentence Transformers
- 384-dimensional embeddings
- Fast inference (~100 products/second on CPU)
- Good balance between speed and quality

For each product, we concatenate:
- Product name
- Title/tagline
- Short description
- Headline description
- Tags (pipe-separated, converted to spaces)
- Meta description

This combined text is then encoded into a 384-dimensional vector. All embeddings are saved to `product_data.pkl` along with the product data.

### Step 2: Query Expansion (Normal Mode Only)

When a user searches, Gemini analyzes the query to understand the actual intent. For example:

**Input**: "gifts for my girlfriend"

**Output**:
```json
{
  "search_intent": "User wants to buy a thoughtful, romantic gift",
  "product_categories": ["jewelry", "necklaces", "accessories", "beauty products"],
  "key_attributes": ["romantic", "elegant", "feminine", "giftable"],
  "context_clues": "Romantic relationship, likely birthday or anniversary",
  "semantic_expansion": "romantic elegant jewelry beautiful necklace bracelet..."
}
```

The `semantic_expansion` field is a dense text representation that captures synonyms, related terms, and use cases. This gets encoded and used for similarity search.

### Step 3: Semantic Search

The query embedding (either raw query or expanded) is compared against all product embeddings using cosine similarity:

```python
similarities = cosine_similarity(query_embedding, product_embeddings)
```

We take the top 30 candidates for reranking (or return directly in fast mode).

### Step 4: LLM Reranking (Normal Mode Only)

The 30 candidates are sent to Gemini with the search context. The model reranks them based on:
- Relevance to actual intent (not just keyword overlap)
- Quality signals (views, votes)
- Price appropriateness
- Description alignment with user needs

Gemini returns a reranked list with relevance scores and reasoning for each product.

### Step 5: Final Scoring

The final score combines AI relevance with popularity signals:

```python
final_score = (
    0.70 * relevance_score +  # AI-determined relevance
    0.20 * normalized_votes +  # Community validation
    0.10 * normalized_views    # General popularity
)
```

This prevents the system from only showing popular items while still giving them a reasonable boost.

---

## Caching with Redis

Making two Gemini API calls per search is slow (3-5 seconds) and expensive. Redis caching helps with repeated queries.

### Cache Architecture

```
src/cache.py - RedisCache class
├── Connection handling with graceful fallback
├── SHA256 key hashing for consistent storage
├── JSON serialization with numpy array support
└── Configurable TTL per cache type
```

### What Gets Cached

| Cache Key Pattern | TTL | What's Stored |
|-------------------|-----|---------------|
| `expand:{query}` | 24 hours | Gemini query expansion result |
| `embed:{text}` | 24 hours | Query text embedding (as list) |
| `final:{query}:{top_k}` | 1 hour | Complete search results |

### Cache Flow

```
User searches "minimalist desk setup"
    │
    ├─ Check cache: "final:minimalist desk setup:10"
    │   └─ HIT? → Return immediately (< 10ms)
    │   └─ MISS? → Continue pipeline
    │
    ├─ Check cache: "expand:minimalist desk setup"
    │   └─ HIT? → Skip Gemini expansion
    │   └─ MISS? → Call Gemini, cache result
    │
    ├─ Check cache: "embed:{expanded_text}"
    │   └─ HIT? → Load embedding from cache
    │   └─ MISS? → Compute embedding, cache result
    │
    └─ Complete pipeline, cache final results
```

### Fallback Behavior

If Redis is unavailable (not running, connection refused, etc.), the system continues without caching. No errors are thrown - just slower responses.

```python
# From cache.py
try:
    self.client.ping()
    self.enabled = True
except Exception as e:
    print(f"CACHE: redis connection failed ({e}), caching disabled")
    self.client = None
```

### Environment Variables

```
REDIS_HOST=localhost  # default
REDIS_PORT=6379       # default
```

---

## Fast Mode vs Normal Mode

### Normal Mode (AI Enhanced)

**Pipeline**:
1. Query expansion with Gemini
2. Embed expanded query
3. Cosine similarity search (top 30)
4. Rerank with Gemini
5. Apply final scoring

**Latency**: 3-5 seconds (uncached), <100ms (cached)

**Strengths**:
- Understands vague queries ("something for a cozy night")
- Reranks based on actual intent, not just similarity
- Provides reasoning for each recommendation

**Weaknesses**:
- Slow on first request
- Costs money (Gemini API)
- Overkill for simple queries

### Fast Mode (Embeddings Only)

**Pipeline**:
1. Embed raw query directly
2. Cosine similarity search (top k)
3. Apply final scoring

**Latency**: 100-300ms (uncached), <50ms (cached)

**Strengths**:
- Near-instant results
- No API costs
- Works offline (once embeddings are generated)

**Weaknesses**:
- Literal matching (searching "laptop" won't find "notebook computer")
- No query understanding
- Generic reasoning ("matched by semantic similarity")

### When to Use Which

| Scenario | Recommended Mode |
|----------|------------------|
| User is typing (autocomplete) | Fast |
| User pressed Enter (final search) | Normal |
| High traffic / cost concerns | Fast |
| Complex queries ("gifts for...") | Normal |
| Simple queries ("wireless earbuds") | Fast |

### Switching Modes

The frontend has a toggle switch. The API accepts a `fast_mode` boolean:

```json
{
  "query": "minimalist desk setup",
  "top_k": 10,
  "fast_mode": true
}
```

---

## Project Structure

```
toastd/
├── src/
│   ├── search.py          # Main search logic (SemanticSearchSystem)
│   ├── server.py          # FastAPI endpoints
│   ├── cache.py           # Redis caching module
│   ├── prod_embeddings.py # Offline embedding generation
│   ├── products.json      # Cleaned product catalog
│   ├── product_data.pkl   # Products + embeddings (pickle)
│   └── .env               # API keys (gitignored)
│
├── frontend/
│   └── index.html         # Search UI
│
├── experiments/
│   ├── fafo01.ipynb       # Data scraping notebook
│   └── *.csv              # Various data exports
│
└── README.md
```

---

## Setup Instructions

### Prerequisites

- Python 3.9+
- Redis (optional, for caching)

### 1. Install Dependencies

```bash
pip install fastapi uvicorn sentence-transformers google-generativeai scikit-learn redis python-dotenv
```

### 2. Set Up Environment Variables

Create `src/.env`:

```
GEMINI_API_KEY=your_api_key_here
REDIS_HOST=localhost
REDIS_PORT=6379
```

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 3. Generate Embeddings (if not already done)

```bash
cd src
python prod_embeddings.py
```

This creates `product_data.pkl` (~2-3 MB for 466 products).

### 4. Start Redis (optional)

On Windows, easiest way is [Memurai](https://www.memurai.com/get-memurai) (Redis-compatible).

Or with Docker:
```bash
docker run -d --name redis -p 6379:6379 redis
```

### 5. Start the Server

```bash
cd src
uvicorn server:app --reload
```

API runs at `http://localhost:8000`

### 6. Open the Frontend

Just open `frontend/index.html` in a browser. Or serve it:

```bash
cd frontend
python -m http.server 3000
```

---

## API Reference

### `POST /search`

Search for products.

**Request**:
```json
{
  "query": "gifts for my girlfriend",
  "top_k": 10,
  "fast_mode": false
}
```

**Response**:
```json
{
  "query": "gifts for my girlfriend",
  "total_results": 10,
  "processing_time_ms": 3245.67,
  "results": [
    {
      "product": {
        "id": "...",
        "name": "Pink Promise Charm Necklace",
        "description": "...",
        "price": 1199.0,
        "image": "https://...",
        "url": "https://...",
        "views": 310,
        "votes": 1
      },
      "final_score": 0.85,
      "relevance_score": 0.92,
      "reasoning": "Romantic jewelry perfect for a girlfriend gift"
    }
  ]
}
```

### `GET /health`

Health check endpoint.

### `GET /product/{product_id}`

Get a single product by ID.

---

## Tradeoffs and Considerations

### Embedding Model Choice

Used `all-MiniLM-L6-v2` because:
- 384 dimensions is reasonable (not too large for memory)
- Fast enough for real-time encoding
- Good general-purpose performance

Could upgrade to `all-mpnet-base-v2` for better quality but slower inference.

### Cache TTL Decisions

- **Query expansion (24h)**: The interpretation of "gifts for girlfriend" doesn't change often
- **Embeddings (24h)**: The encoded text is deterministic
- **Final results (1h)**: Shorter because product popularity can change

### Why Not Cache Reranking?

The reranking step depends on:
- The exact set of 30 candidates (changes if embeddings change)
- The expanded context (cached separately)

Caching the final results covers this case anyway.

### Scoring Weights

```python
final_score = 0.70 * relevance + 0.20 * votes + 0.10 * views
```

These weights are somewhat arbitrary. Could be tuned with A/B testing on real users. The idea is:
- Relevance should dominate (that's the whole point)
- Votes indicate quality (people liked it enough to upvote)
- Views indicate interest (but could also mean misleading titles)

### Cold Start Problem

New products with zero views/votes won't get much boost from popularity signals. This is actually fine - it lets relevance dominate for new items.

---

## Future Improvements

Things that could make this better but weren't implemented:

1. **Hybrid search**: Combine BM25 keyword search with semantic search
2. **Personalization**: Track user preferences and adjust rankings
3. **Category filters**: Let users narrow down by price range, tags, etc.
4. **Streaming responses**: Show results as they come in instead of waiting
5. **Better evaluation**: Set up proper metrics (NDCG, MRR) with labeled data

---

## License

This is a personal project for learning purposes. The product data belongs to Toastd.in.
