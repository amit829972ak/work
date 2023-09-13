import streamlit as st
import os
import PyPDF2
from io import BytesIO

# Get the current directory (where this script is located)
current_directory = os.path.dirname(os.path.abspath(__file__))

# List all folders in the current directory and sort them alphabetically
folder_names = sorted([folder for folder in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, folder))])

st.title("List of Folders in Current Directory")

if folder_names:
    selected_pdfs = []

    for folder in folder_names:
        # List PDF files in the current folder
        folder_path = os.path.join(current_directory, folder)
        pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".pdf")]

        # Extract only the file names for display
        pdf_file_names = [os.path.basename(pdf_file) for pdf_file in pdf_files]

        # Create a multiselect widget to choose PDF files for the current folder
        selected_pdfs.extend(st.multiselect(f"Select PDFs for '{folder}':", pdf_file_names))

    if selected_pdfs:
        st.write("Selected PDFs:")
        for pdf_name in selected_pdfs:
            st.write(pdf_name)

        # Merge the selected PDFs into a single PDF
        merged_pdf = PyPDF2.PdfMerger()
        for pdf_name in selected_pdfs:
            # Get the full path of the selected PDF using its name
            pdf_path = os.path.join(current_directory, folder_names[selected_pdfs.index(pdf_name)], pdf_name)
            merged_pdf.append(pdf_path)

        # Create a download link for the merged PDF
        st.markdown("### Download Merged PDF")
        with st.spinner("Merging PDFs..."):
            merged_pdf_bytes = BytesIO()
            merged_pdf.write(merged_pdf_bytes)
            st.download_button("Download Merged PDF", merged_pdf_bytes.getvalue(), file_name="merged_pdf.pdf", key="merged_pdf")

    else:
        st.write("No PDFs selected.")
else:
    st.write("No folders found in the current directory.")
