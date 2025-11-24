# simple fastapi server to expose the search system as an api
# run with: uvicorn server:app --reload

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os

from search import SemanticSearchSystem

# config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCss7mONolh071ibtAkyqK7sI-MjgMJgdw")
PRODUCT_DATA_PATH = "product_data.pkl"

# setup fastapi
app = FastAPI(
    title="Product Search API",
    description="semantic search for products",
    version="1.0.0"
)

# cors so frontend can access this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_system = None


# request/response models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=50)


class ProductResponse(BaseModel):
    id: str
    name: str
    description: str
    headline: Optional[str]
    price: float
    image: Optional[str]
    url: Optional[str]
    tags: Optional[str]
    views: int
    votes: int


class SearchResultItem(BaseModel):
    product: ProductResponse
    final_score: float
    relevance_score: float
    reasoning: str


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResultItem]
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    products_loaded: int
    model_ready: bool


# startup/shutdown
@app.on_event("startup")
async def startup_event():
    global search_system
    
    print("\n" + "="*60)
    print("starting api server...")
    print("="*60)
    
    try:
        search_system = SemanticSearchSystem(
            gemini_api_key=GEMINI_API_KEY,
            product_data_path=PRODUCT_DATA_PATH
        )
        
        print(f"loaded {len(search_system.products)} products")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"failed to start: {e}")
        print("make sure product_data.pkl exists")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    print("\nshutting down...")


# endpoints
@app.get("/", response_model=dict)
async def root():
    return {
        "service": "Product Search API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "search": "/search (POST)",
            "product": "/product/{product_id} (GET)"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if search_system is None:
        raise HTTPException(status_code=503, detail="not ready")
    
    return HealthResponse(
        status="healthy",
        products_loaded=len(search_system.products),
        model_ready=True
    )


@app.post("/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    # main search endpoint
    if search_system is None:
        raise HTTPException(status_code=503, detail="not ready")
    
    try:
        import time
        start_time = time.time()
        
        # run search
        results = search_system.search(
            user_query=request.query,
            top_k=request.top_k,
            verbose=False
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # format results
        formatted_results = []
        for result in results:
            product = result['product']
            
            formatted_results.append(
                SearchResultItem(
                    product=ProductResponse(
                        id=product.get('id', ''),
                        name=product.get('name', ''),
                        description=product.get('short_description', ''),
                        headline=product.get('headline_description'),
                        price=product.get('final_price', product.get('price', 0)),
                        image=product.get('main_image'),
                        url=product.get('product_url'),
                        tags=product.get('tags'),
                        views=product.get('view_count', 0),
                        votes=product.get('vote_count', 0)
                    ),
                    final_score=result['final_score'],
                    relevance_score=result['relevance_score'],
                    reasoning=result['reasoning']
                )
            )
        
        return SearchResponse(
            query=request.query,
            total_results=len(formatted_results),
            results=formatted_results,
            processing_time_ms=round(processing_time, 2)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search failed: {str(e)}")


@app.get("/product/{product_id}", response_model=ProductResponse)
async def get_product(product_id: str):
    # get single product by id
    if search_system is None:
        raise HTTPException(status_code=503, detail="not ready")
    
    # find product
    product = None
    for p in search_system.products:
        if p.get('id') == product_id:
            product = p
            break
    
    if product is None:
        raise HTTPException(status_code=404, detail="not found")
    
    return ProductResponse(
        id=product.get('id', ''),
        name=product.get('name', ''),
        description=product.get('short_description', ''),
        headline=product.get('headline_description'),
        price=product.get('final_price', product.get('price', 0)),
        image=product.get('main_image'),
        url=product.get('product_url'),
        tags=product.get('tags'),
        views=product.get('view_count', 0),
        votes=product.get('vote_count', 0)
    )


# run the server
if __name__ == "__main__":
    import uvicorn
    
    print("""
    ========================================
    product search api
    ========================================
    
    to run:
    1. pip install fastapi uvicorn
    2. set GEMINI_API_KEY env variable
    3. uvicorn server:app --reload
    
    then check:
    - http://localhost:8000/docs
    - http://localhost:8000/health
    ========================================
    """)
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )