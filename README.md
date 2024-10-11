## Overview
A simple QA app that answer questions based on specified data imported in docx and pdf formats

Sample: [sample1.docx](./samples/sample1.docx)

Result:

<img width="663" alt="Screen Shot 2024-10-11 at 11 42 03 PM" src="https://github.com/user-attachments/assets/d5a14b3b-c541-4dc8-b6b7-a38993210bed">


## Installation
### install libraries
```
pip install -r requirements.txt
```

### copy env file and put your OPENAI_API_KEY
```
cp .env.example .env
```


### put your documents(docx,pdf) inside data/documents

### create indexes for your documents
```
python3 QA/index_documents.py
```

## Usage

### run the project

```
python3 QA
```

### you will be prompted to enter your question then you will get the answer
