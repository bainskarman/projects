# Import necessary libraries
from convert import ExtractPDFText
from ATS_score import calculateATSscore
from model import modelFeedback
import streamlit as st
import time


if "page_number" not in st.session_state:
    st.session_state.page_number = 1

if "resume_data" not in st.session_state:
    st.session_state.resume_data = ""

if "jobdescription" not in st.session_state:
    st.session_state.jobdescription = ""
    
def set_page_number_and_reset_data():
    st.session_state.page_number = 1
    st.session_state.resume_data = ""


def page1():
    st.title("AI-Powered ATS Screening")
    if not st.session_state.resume_data:
        pdf = st.file_uploader(label="Upload your resume", type="pdf")
        st.write("No Resume Yet? Create one [here](https://simplify.jobs/builder)")

        if pdf:
            st.success("Resume uploaded successfully.")
            st.session_state.resume_data = ExtractPDFText(pdf)

def page2():
    st.title("AI-Powered ATS Screening: Job Description")
    st.session_state.jobdescription = st.text_area("Job Description: ")
    st.info("You can just copy paste from the job portal")
    submit = st.button("Submit")

    if submit:
        start()

def page3():
    st.title("Your Resume data: ")
    if st.session_state.resume_data:
        st.write(st.session_state.resume_data)
    else:
        st.error("Please upload your resume to view the extracted data")

def start():
    if st.session_state.resume_data and st.session_state.jobdescription:
        with st.spinner("Hold on, we're calculating your ATS Score..."):
            ATS_score = calculateATSscore(st.session_state.resume_data, st.session_state.jobdescription)
            model_feedback = modelFeedback(ATS_score, st.session_state.resume_data,st.session_state.jobdescription)

        st.subheader("AI FEEDBACK:")
        st.write(model_feedback) 
        
    else:
        st.info("Please, upload Resume and Provide the Job Description")

if st.session_state.page_number == 1:
    page1()
elif st.session_state.page_number == 2:
    page2()
elif st.session_state.page_number == 3:
    page3()

if st.session_state.page_number == 1:
    st.button("View your Extracted Resume data", on_click = lambda: setattr(st.session_state,"page_number", 3))
    st.button("Go to Job Description Page", on_click=lambda: setattr(st.session_state, "page_number", 2))
else:
    st.button("Go to PDF Upload Page", on_click=lambda: set_page_number_and_reset_data())

