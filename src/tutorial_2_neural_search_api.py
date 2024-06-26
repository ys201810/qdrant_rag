# coding=utf-8
from fastapi import FastAPI
from tutorial_2_neural_searcher import NeuralSearcher

app = FastAPI()
neural_searcher = NeuralSearcher(collection_name="startups")


@app.get("/api/search")
def search_startup(q: str):
    return {"result": neural_searcher.search(text=q)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



