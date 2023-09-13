import streamlit as st
import os

# Get the current directory (where this script is located)
current_directory = os.path.dirname(os.path.abspath(__file__))

# List all folders in the current directory
folder_names = [folder for folder in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, folder))]

st.title("List of Folders in Current Directory")

if folder_names:
    selected_pdfs = []

    for folder in folder_names:
        # List PDF files in the current folder
        folder_path = os.path.join(current_directory, folder)
        pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]

        # Create a multiselect widget to choose PDF files for the current folder
        selected_pdfs.extend(st.multiselect(f"Select PDFs for '{folder}':", pdf_files))

    if selected_pdfs:
        st.write("Selected PDFs:")
        for pdf in selected_pdfs:
            st.write(pdf)
            
        # Display selected PDFs
        st.markdown("### Display Selected PDFs")
        for pdf_file in selected_pdfs:
            pdf_path = os.path.join(current_directory, pdf_file)
            try:
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                    st.markdown(pdf_bytes, unsafe_allow_html=True)
            except FileNotFoundError:
                st.error(f"File not found: {pdf_path}")
    else:
        st.write("No PDFs selected.")
else:
    st.write("No folders found in the current directory.")
