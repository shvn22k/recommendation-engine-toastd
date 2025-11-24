# generate embeddings for products so we can do semantic search later
# this needs to run once, or when we add new products

import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import pickle
from tqdm import tqdm


class ProductEmbeddingGenerator:
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # using all-MiniLM-L6-v2 because it's fast and good enough
        # could try all-mpnet-base-v2 for better results but slower
        print(f"loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("model loaded")
    
    def create_product_text(self, product):
        # combine all the text fields from product into one string
        parts = []
        
        if product.get('name'):
            parts.append(product['name'])
        
        if product.get('title'):
            parts.append(product['title'])
        
        if product.get('short_description'):
            parts.append(product['short_description'])
        
        if product.get('headline_description'):
            parts.append(product['headline_description'])
        
        # tags are separated by |, split and add them
        if product.get('tags'):
            tags = product['tags'].split('|')
            parts.append(' '.join(tags))
        
        if product.get('meta_description'):
            parts.append(product['meta_description'])
        
        text = ' '.join(parts)
        text = ' '.join(text.split())  # clean extra spaces
        
        return text
    
    def generate_embeddings(self, products, batch_size=32):
        print(f"\ngenerating embeddings for {len(products)} products...")
        
        # first create text for each product
        texts = []
        for product in tqdm(products, desc="creating texts"):
            text = self.create_product_text(product)
            texts.append(text)
        
        # now generate embeddings
        print("encoding texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"done! embeddings shape: {embeddings.shape}")
        return embeddings
    
    def save_data(self, products, embeddings, output_path='product_data.pkl'):
        # save everything to a pickle file
        data = {
            'products': products,
            'embeddings': embeddings,
            'model_name': self.model.get_sentence_embedding_dimension()
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        file_size = os.path.getsize(output_path) / 1024 / 1024
        print(f"saved {len(products)} products to {output_path}")
        print(f"file size: {file_size:.2f} MB")
    
    def process_and_save(self, products_file, output_file='product_data.pkl'):
        # load products from json or csv
        print(f"loading products from {products_file}")
        
        if products_file.endswith('.json'):
            with open(products_file, 'r', encoding='utf-8') as f:
                products = json.load(f)
        elif products_file.endswith('.csv'):
            df = pd.read_csv(products_file)
            products = df.to_dict('records')
        else:
            raise ValueError("need json or csv file")
        
        print(f"loaded {len(products)} products")
        
        embeddings = self.generate_embeddings(products)
        self.save_data(products, embeddings, output_file)
        
        return products, embeddings


def load_product_data(file_path='product_data.pkl'):
    # load the saved data
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['products'], data['embeddings']


if __name__ == "__main__":
    import os
    
    # setup
    generator = ProductEmbeddingGenerator(model_name='all-MiniLM-L6-v2')
    
    # change this to your products file path
    output_path = r'C:\Projects\toastd\src\product_data.pkl'
    products, embeddings = generator.process_and_save(
        products_file=r'C:\Projects\toastd\src\products.json',
        output_file=output_path
    )
    
    print("\n" + "="*50)
    print("DONE!")
    print("="*50)
    print(f"processed {len(products)} products")
    print(f"embeddings: {embeddings.shape[0]} x {embeddings.shape[1]}")
    print(f"saved to {output_path}")
    print("\nnow you can use these embeddings for search")