# History

## Erros
```
openai.APIStatusError: Error code: 413 - {'code': 20042, 'message': 'input batch size 864 > maximum allowed batch size 64', 'data': None}
```
Your error means you are sending too many documents (chunks) at once to the OpenAI embeddings API. The batch size limit is 64, but your code is sending 864.
To fix this, you need to split your chunks into batches of 64 or fewer before calling embed_documents. I will update your script to process the chunks in batches and combine the results.