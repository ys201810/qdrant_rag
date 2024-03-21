# neural search
## 参考
[Create a Simple Neural Search Service](https://qdrant.tech/documentation/tutorials/neural-search/)  

## データ用意
```
$ cd data/startups/
$ wget https://storage.googleapis.com/generall-shared-data/startups_demo.json
```

## エンべディング用モデル用意とベクトル化
```
entence_transformer_model = "all-MiniLM-L6-v2"
model = SentenceTransformer(
    entence_transformer_model,
    device="cpu"  # "cuda"だとGPU
)

df = pd.read_json("../data/startups/startups_demo.json", lines=True)

vectors = model.encode(
    [row.alt + ". " + row.description for row in df.itertuples()],
    show_progress_bar=True,
)
```

上記で約20分かかる。(macbookairのCPU。40474件のテキスト。)  



## qdrantの利用設定
```
$ cd shell
$ sh start_qdrant.sh
```

## pythonモジュールinstall(FASTAPI)
```
$ poetry add fastapi uvicorn
```

## コード
### tutorial_2_neural_search_make_data.py
データから、collectionを作成する。

### tutorial_2_neural_searcher.py
検索用のclassファイル。search関数内で検索を実施する。filteringも可能。  

### tutorial_2_neural_search_api.py
FASTAPIでのAPIの作成。  

```
$ poetry run python tutorial_2_neural_search_api.py
```

を実行後、`http://localhost:8000/docs`にアクセスして、パラメータqに任意の文字列を入れて動作を確認できる。  
