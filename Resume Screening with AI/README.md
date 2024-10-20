# ATS Scanner
[Hugging Face Spaces - ATS Scanner](https://huggingface.co/spaces/bainskarman/ATSScanner)
Welcome to the **ATS Scanner**! This application evaluates your resume and provides an ATS (Applicant Tracking System) score, along with actionable feedback to improve your chances of getting noticed by recruiters.
## Overview
The ATS Scanner utilizes advanced natural language processing models to analyze your resume content. It shortens your ATS score using the `paraphrase-MiniLM-L6-v2` model and generates a detailed review using the `nvidia/llama-3.1-nemotron-70b-instruct` model. The app highlights areas for improvement and what aspects of your resume are strong.
## Features
- **ATS Score Assessment**: Receive a concise ATS score based on your resume content.
- **Detailed Feedback**: Get a comprehensive review of your resume, including:
  - Strengths to emphasize
  - Areas needing improvement
  - Tips for optimization
## How to Use
1. **Input Your Resume**: Upload your resume document or paste the text directly into the provided field.
2. **Generate ATS Score**: Click the button to analyze your resume.
3. **Review Feedback**: Examine the generated ATS score and feedback to understand how to enhance your resume.
## Technologies Used
- **paraphrase-MiniLM-L6-v2**: For scoring the ATS effectiveness of your resume.
- **nvidia/llama-3.1-nemotron-70b-instruct**: For generating detailed, actionable feedback on your resume.