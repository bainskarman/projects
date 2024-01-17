import streamlit as st

def main():
    st.title("Power BI Dashboard in Streamlit")

    # Replace the URL with your Power BI dashboard URL
    power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiOTg4ODViNzItMDczYi00NTMyLTgyM2MtY2I2OGYwZGZlMmQ5IiwidCI6ImI2NDE3Y2QwLTFmNzMtNDQ3MS05YTM5LTIwOTUzODIyYTM0YSIsImMiOjN9"
    
    # Embed Power BI dashboard using iframe
    st.markdown(f'<iframe width="800" height="600" src="{power_bi_url}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

