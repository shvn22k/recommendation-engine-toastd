# search system using embeddings + gemini for better results
# uses cosine similarity for initial search, then gemini reranks

import json
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import pickle
from datetime import datetime


class SemanticSearchSystem:
    
    def __init__(self, gemini_api_key, product_data_path='product_data.pkl'):
        # setup gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # load same model used for creating embeddings
        print("loading model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # load products and their embeddings
        print("loading products...")
        with open(product_data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.products = data['products']
        self.product_embeddings = data['embeddings']
        
        print(f"loaded {len(self.products)} products")
        
        self._calculate_stats()
    
    def _calculate_stats(self):
        # get max values for normalizing popularity scores
        view_counts = [p.get('view_count', 0) for p in self.products]
        vote_counts = [p.get('vote_count', 0) for p in self.products]
        
        # avoid division by zero if no views/votes in dataset
        self.max_views = max(view_counts) if view_counts and max(view_counts) > 0 else 1
        self.max_votes = max(vote_counts) if vote_counts and max(vote_counts) > 0 else 1
    
    def expand_query(self, user_query):
        # use gemini to understand what user really wants
        # expands the query into categories, attributes, etc
        
        prompt = f"""You are an e-commerce search expert. Your job is to understand what users are REALLY looking for when they search.

User Query: "{user_query}"

Analyze this query deeply and return a JSON object that helps find the right products.

Your JSON must have these fields:

1. "search_intent": One sentence describing what the user actually wants
2. "product_categories": List of product types/categories that would satisfy this (be specific, 5-10 items)
3. "key_attributes": Important product qualities/features the user cares about (e.g., style, use case, quality level)
4. "context_clues": Any implicit context (occasion, recipient, urgency, price sensitivity, etc.)
5. "semantic_expansion": A rich 40-60 word text that represents this search intent, written to match against product descriptions (include synonyms, related terms, use cases)

Examples to guide you:

Query: "gifts for my girlfriend"
{{
  "search_intent": "User wants to buy a thoughtful, romantic gift for their romantic partner",
  "product_categories": ["jewelry", "necklaces", "bracelets", "rings", "accessories", "beauty products", "fragrances", "handbags", "fashion items", "personal care"],
  "key_attributes": ["romantic", "elegant", "feminine", "thoughtful", "beautiful", "high-quality", "giftable", "special"],
  "context_clues": "Romantic relationship, wants to impress, likely birthday or anniversary or spontaneous gesture, willing to spend reasonably, needs gift packaging",
  "semantic_expansion": "romantic elegant jewelry beautiful necklace bracelet ring feminine accessories thoughtful gift girlfriend partner love special occasion anniversary birthday present beautiful fragrance beauty products stylish handbag fashion items personal care premium quality giftable"
}}

Query: "workout equipment for home"
{{
  "search_intent": "User wants to set up home gym or fitness area",
  "product_categories": ["dumbbells", "resistance bands", "yoga mats", "fitness equipment", "weights", "exercise gear", "workout accessories", "home gym equipment"],
  "key_attributes": ["durable", "compact", "effective", "versatile", "quality", "space-saving", "functional"],
  "context_clues": "Work from home or limited gym access, wants convenience, likely beginner to intermediate, needs space-efficient solutions",
  "semantic_expansion": "home workout equipment fitness gear exercise dumbbells weights resistance bands yoga mat gym equipment training accessories compact space-saving durable quality functional versatile strength training cardio home gym setup"
}}

Query: "minimalist desk accessories"
{{
  "search_intent": "User wants clean, simple desk items with aesthetic appeal",
  "product_categories": ["desk organizers", "pen holders", "cable management", "desk lamps", "stationery", "office accessories", "desk decor", "workspace items"],
  "key_attributes": ["minimalist", "clean design", "functional", "aesthetic", "simple", "organized", "modern", "sleek"],
  "context_clues": "Values aesthetics and organization, likely remote worker or student, prefers quality over quantity, willing to pay for good design",
  "semantic_expansion": "minimalist desk accessories office simple clean design modern workspace organizer aesthetic functional stationery pen holder cable management sleek desk lamp organization tools workspace decor contemporary style productivity clutter-free"
}}

Now analyze: "{user_query}"

Return ONLY valid JSON, no other text.
"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            text = response.text.strip()
            
            # clean up markdown formatting
            text = text.replace('```json', '').replace('```', '').strip()
            
            expanded = json.loads(text)
            return expanded
            
        except Exception as e:
            print(f"query expansion failed: {e}")
            print("using simple fallback")
            
            return {
                "search_intent": f"Find products related to: {user_query}",
                "product_categories": [user_query],
                "key_attributes": [],
                "context_clues": "General search",
                "semantic_expansion": user_query
            }
    
    def search_products(self, query_text, top_k=30):
        # search using cosine similarity
        # returns top k matching products
        
        # embed the query
        query_embedding = self.encoder.encode([query_text], convert_to_numpy=True)
        
        # calculate similarities with all products
        similarities = cosine_similarity(query_embedding, self.product_embeddings)[0]
        
        # get top k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results
    
    def rerank_with_gemini(self, user_query, expanded_context, candidates, top_k=10):
        # use gemini to rerank the initial results
        # considers intent, not just keywords
        
        # prepare product data for gemini
        products_for_llm = []
        for i, candidate in enumerate(candidates):
            product = candidate['product']
            products_for_llm.append({
                'index': i,
                'id': product.get('id', ''),
                'name': product.get('name', ''),
                'description': product.get('short_description', ''),
                'headline': product.get('headline_description', ''),
                'tags': product.get('tags', ''),
                'price': product.get('final_price', product.get('price', 0)),
                'popularity': {
                    'views': product.get('view_count', 0),
                    'votes': product.get('vote_count', 0)
                }
            })
        
        prompt = f"""You are an expert product recommender for an e-commerce platform.

User Query: "{user_query}"

Search Context:
- Intent: {expanded_context['search_intent']}
- Looking for: {', '.join(expanded_context['product_categories'][:5])}
- Key attributes: {', '.join(expanded_context['key_attributes'][:5])}

Candidate Products:
{json.dumps(products_for_llm, indent=2)}

Task: Rank the top {top_k} products that BEST match what the user wants.

Consider:
1. Relevance to the actual intent (not just keyword match)
2. Quality signals (views, votes indicate popularity/trust)
3. Price appropriateness for the context
4. Product description alignment with user needs
5. Overall likelihood of customer satisfaction

Return ONLY valid JSON array with exactly {top_k} products:
[
  {{
    "index": 0,
    "relevance_score": 0.95,
    "reasoning": "One clear sentence explaining why this is highly relevant"
  }},
  ...
]

Sort by relevance (best first). Return valid JSON only, no other text."""

        try:
            response = self.gemini_model.generate_content(prompt)
            text = response.text.strip()
            
            text = text.replace('```json', '').replace('```', '').strip()
            
            reranked = json.loads(text)
            
            # add full product data
            for item in reranked:
                idx = item['index']
                if idx < len(candidates):
                    item['product'] = candidates[idx]['product']
            
            return reranked[:top_k]
            
        except Exception as e:
            print(f"reranking failed: {e}")
            print("using similarity scores instead")
            
            # fallback to just using similarity
            return [
                {
                    'index': i,
                    'product': candidates[i]['product'],
                    'relevance_score': candidates[i]['similarity'],
                    'reasoning': 'High semantic similarity to search query'
                }
                for i in range(min(top_k, len(candidates)))
            ]
    
    def apply_final_scoring(self, reranked):
        # combine ai relevance with popularity
        
        for item in reranked:
            product = item['product']
            
            # normalize popularity
            view_score = product.get('view_count', 0) / self.max_views
            vote_score = product.get('vote_count', 0) / self.max_votes
            
            # weighted score
            item['final_score'] = (
                0.70 * item['relevance_score'] +  # ai relevance most important
                0.20 * vote_score +                # votes matter
                0.10 * view_score                  # views help a bit
            )
        
        reranked.sort(key=lambda x: x['final_score'], reverse=True)
        
        return reranked
    
    def search(self, user_query, top_k=10, verbose=True):
        # main search function
        # does everything: expand, search, rerank, score
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"searching: '{user_query}'")
            print(f"{'='*60}")
        
        # expand query with gemini
        if verbose:
            print("\n1. expanding query...")
        
        expanded = self.expand_query(user_query)
        
        if verbose:
            print(f"   intent: {expanded['search_intent']}")
            print(f"   categories: {', '.join(expanded['product_categories'][:5])}")
        
        # semantic search
        if verbose:
            print("\n2. searching...")
        
        search_text = expanded['semantic_expansion']
        candidate_indices = self.search_products(search_text, top_k=30)
        
        candidates = [
            {
                'product': self.products[idx],
                'similarity': score
            }
            for idx, score in candidate_indices
        ]
        
        if verbose:
            print(f"   found {len(candidates)} candidates")
        
        # rerank with gemini
        if verbose:
            print("\n3. reranking...")
        
        reranked = self.rerank_with_gemini(user_query, expanded, candidates, top_k=top_k)
        
        # final scoring
        if verbose:
            print("\n4. final scoring...")
        
        final_results = self.apply_final_scoring(reranked)
        
        if verbose:
            print(f"\ndone! returning {len(final_results)} results")
            print(f"{'='*60}\n")
        
        return final_results


def display_results(results):
    # print results nicely
    
    for i, result in enumerate(results, 1):
        product = result['product']
        
        print(f"\n{i}. {product['name']}")
        print(f"   price: ₹{product.get('final_price', product.get('price', 0))}")
        print(f"   score: {result['final_score']:.3f} (ai: {result['relevance_score']:.2f})")
        print(f"   why: {result['reasoning']}")
        print(f"   popularity: {product.get('view_count', 0)} views, {product.get('vote_count', 0)} votes")
        print(f"   url: {product.get('product_url', 'N/A')}")


if __name__ == "__main__":
    
    # setup
    GEMINI_API_KEY = "your gemini api key"
    
    search_system = SemanticSearchSystem(
        gemini_api_key=GEMINI_API_KEY,
        product_data_path=r'C:\Projects\toastd\src\product_data.pkl'
    )
    
    # test queries
    test_queries = [
        "gifts for my girlfriend",
        "home workout equipment",
        "minimalist desk setup",
        "skincare routine products",
        "travel accessories for backpacking"
    ]
    
    print("\n" + "="*60)
    print("testing searches")
    print("="*60)
    
    for query in test_queries:
        results = search_system.search(query, top_k=5, verbose=True)
        display_results(results)
        print("\n" + "-"*60)