import streamlit as st
import sys
import os
import tempfile
from pipeline_ui import pipeline_ui

st.set_page_config(page_title='ALPR System', layout='wide', initial_sidebar_state='expanded')

# Set up the Streamlit app layout
st.title("ALPR System")
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    st.video(video_path)

    if st.button('Process Video'):
        with st.spinner('Processing...'):
            output_csv_path, output_video_path = pipeline_ui(video_path)
            st.success('Processing complete!')
            st.write("CSV results have been written to:", output_csv_path)
            st.write("Video results have been written to:", output_video_path)
            st.video(output_video_path)
            st.download_button('Download CSV', data=open(output_csv_path).read(), file_name='final_output.csv')
            st.download_button('Download Video', data=open(output_video_path).read(), file_name='final_output.mp4')