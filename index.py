from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
from textstat import flesch_reading_ease, spache_readability
# c.f. https://pypi.org/project/textstat/

# Load a pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Tokenize and encode the prompt and the response
prompt = "Write about the effects of climate change."

# response = "Climate change is making the world hotter. Ice in the North and South poles is melting. This can make sea levels go up."
# Similarity between prompt and response: 0.7665253281593323
# Readability score: 97.5
# Spache Readability score: 3.42

# response = "It's making the world hotter."
# Similarity between prompt and response: 0.5627638101577759
# Readability score: 83.32
# Spache Readability score: 3.26

# response = "I'm hot 😊"
# Similarity between prompt and response: 0.5204846858978271
# Readability score: 120.21
# Spache Readability score: 1.12

response = "ㄴㅇㄹ ㅋzzz 🏃‍♂️ íåødf zzzz"
# Similarity between prompt and response: 0.4239700436592102
# flesch_reading_ease score: 118.18
# Spache Readability score: 1.4


prompt_tokens = tokenizer.encode(prompt, return_tensors='pt')
response_tokens = tokenizer.encode(response, return_tensors='pt')

# Get the text embeddings
with torch.no_grad():
    prompt_embeddings = model(prompt_tokens)[0].mean(dim=1)
    response_embeddings = model(response_tokens)[0].mean(dim=1)

# Compute the cosine similarity between the embeddings
similarity = cosine_similarity(
    prompt_embeddings.detach().numpy(), 
    response_embeddings.detach().numpy()
)[0][0]

# Compute the Flesch Reading Ease score
readability_score = flesch_reading_ease(response)

print(f"Similarity between prompt and response: {similarity}")
print(f"flesch_reading_ease score: {readability_score}")
# Score	Difficulty
# 90-100	Very Easy
# 80-89	Easy
# 70-79	Fairly Easy
# 60-69	Standard
# 50-59	Fairly Difficult
# 30-49	Difficult
# 0-29	Very Confusing

print(f"Spache Readability score: {spache_readability(response)}")




# Response examples:

# A1: 
# Climate change is making the world hotter. Ice in the North and South poles is melting. This can make sea levels go up.
# 
# Similarity between prompt and response: 0.7665253281593323
# Readability score: 97.5
# Spache Readability score: 3.42


# A2:
# Climate change is causing a lot of problems. It makes the Earth hotter which causes the ice at the poles to melt. This is dangerous because it causes the sea levels to rise. That means some places near the sea might go under water. It also makes weather more extreme, so we have bigger storms and longer heatwaves.
# 
# Similarity between prompt and response: 0.7381290197372437
# Readability score: 85.08
# Spache Readability score: 3.66

# B1:
# Climate change is having a profound effect on our planet. Firstly, the rise in global temperature, often referred to as global warming, is causing polar ice caps to melt. This leads to rising sea levels and the potential flooding of low-lying areas. Climate change also affects weather patterns. More extreme weather events, such as hurricanes, droughts, and heatwaves, are seen. Finally, it also impacts biodiversity. With changing weather and habitats, many species find it hard to survive.
# 
# Similarity between prompt and response: 0.7558901906013489
# Readability score: 60.31
# Spache Readability score: 5.18

# B2:
# Climate change is having diverse and profound effects on our planet. One of the most immediate effects is the rising global temperature, commonly known as global warming. This phenomenon is causing the polar ice caps to melt at an alarming rate, leading to a subsequent rise in sea levels. This could potentially result in the displacement of millions of people living in coastal areas and islands. Another noticeable effect of climate change is the increase in frequency and intensity of extreme weather events such as hurricanes, floods, droughts, and heatwaves. These events not only pose a significant risk to human lives and infrastructure but also disrupt ecosystems and agriculture. Moreover, climate change affects biodiversity, with numerous species threatened due to changing habitats and increased competition for limited resources.
# 
# Similarity between prompt and response: 0.7250790596008301
# Readability score: 44.44
# Spache Readability score: 6.64

# C1:
# Climate change represents one of the most significant environmental challenges of our time, and its effects are wide-ranging and multidimensional. Global warming, one of the most well-known consequences, results from increased concentrations of greenhouse gases in the atmosphere. It is causing polar ice caps and glaciers to melt, leading to a rise in sea levels. This poses a significant threat to low-lying areas and islands due to the potential for widespread flooding and displacement of populations. Moreover, climate change is radically altering global weather patterns, leading to increased frequency and severity of extreme weather events. This not only threatens human safety and property through phenomena like hurricanes and floods, but also affects agricultural productivity due to unpredictable seasons and severe droughts. Furthermore, changing climatic conditions threaten biodiversity by altering habitats and disrupting the delicate balance of ecosystems. Species are finding it increasingly difficult to adapt quickly enough to these changes, leading to population declines and even extinction in the worst cases. As climate change accelerates, it's evident that its effects are not isolated but interconnected, producing a complex cascade of environmental, social, and economic impacts. 
# 
# Similarity between prompt and response: 0.7121914625167847
# Readability score: 33.65
# Spache Readability score: 7.22


