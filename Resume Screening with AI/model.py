import csv
import io
import requests
import json
import html  # For escaping HTML characters
from bs4 import BeautifulSoup
from openai import OpenAI 

# Initialize OpenAI API with Nvidia's Llama 3.1 70b nemotron model
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

def clean_text_output(text):
    """
    Cleans the output to handle HTML characters and unwanted tags.
    """
    text = html.unescape(text)  # Unescape HTML entities
    soup = BeautifulSoup(text, 'html.parser')  # Use BeautifulSoup to handle HTML tags
    cleaned_text = soup.get_text(separator="\n").strip()  # Remove tags and handle newlines
    return cleaned_text

def modelFeedback(ats_score, resume_data, job_description):
    input_prompt = f"""
    You are now an ATS Score analyzer and given ATS Score is {int(ats_score * 100)}%. 
    Your task is to provide feedback to the user based on the ATS score.
    Print ATS score first. Mention where the resume is good and where the resume lacks. 
    Show list of missing skills and suggest improvements. 
    Show list of weak action verbs and suggest improvements.
    Show weaker sentences and suggest improvements.
    Talk about each section of the user's resume and discuss good and bad points of it only if it has any. 
    Resume Data: {resume_data}
    Job Description: {job_description}
    """

    try:
        # Generate response using the OpenAI API
        response = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",  # Using Llama 3.1 70b
            messages=[
                {"role": "user", "content": input_prompt}
            ],
            temperature=0.03,  # Lowering temperature for precise output
            top_p=0.7,  # Prioritize high-probability tokens
            max_tokens=700,  # Allow longer content
        )

        # Extract and clean the response
        feedback_text = response.choices[0].message.content.strip()  # Corrected line
        cleaned_feedback = clean_text_output(feedback_text)

        return cleaned_feedback

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {str(e)}")
        return "Error: Unable to generate feedback."