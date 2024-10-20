from sentence_transformers import SentenceTransformer, util

def calculateATSscore_with_bert(resume_data, job_description):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Pre-trained BERT model
    resume_embedding = model.encode(resume_data)
    job_desc_embedding = model.encode(job_description)
    similarity_value = util.cos_sim(resume_embedding, job_desc_embedding)
    return similarity_value.item()



