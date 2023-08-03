
# Writing assessment

## How to define the comprehention and proficiency from the answer of ESL user.


## Use pyenv
Active virtual environment
```bash
source ./env/bin/activate
```
Install required packages
```bash
pip install -r requirements.txt
```

## execute
```bash
python index.py
```


## Explains

Result example
```
Similarity between prompt and response: 0.7665253281593323
Readability score: 97.5
Spache Readability score: 3.42
```

### Similarity between prompt and response
It can represent coherence of the response from the original question.

### Readability score
It is the value of flesch_reading_ease from the textstat library.
```
Score	Difficulty
90-100	Very Easy
80-89	Easy
70-79	Fairly Easy
60-69	Standard
50-59	Fairly Difficult
30-49	Difficult
0-29	Very Confusing
```
- Proficiency of the user can be defined by this value.
- It has inversely proportional relationship with the proficiency.


### Spache Readability score
- It is the value of spache_readability from the textstat library.
- [Spache readability formula](https://en.wikipedia.org/wiki/Spache_readability_formula)
- It is the grade level of the text.
- It has directly proportional relationship with the proficiency.