import streamlit as st
import os
import PyPDF2
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Get the current directory (where this script is located)
current_directory = os.path.dirname(os.path.abspath(__file__))

# List all folders in the current directory and sort them alphabetically
folder_names = sorted([folder for folder in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, folder))])

st.title("List of Folders in Current Directory")

if folder_names:
    selected_pdfs = []

    # Create a mapping of selected PDFs to their folder and file paths
    selected_pdf_paths = []

    for folder in folder_names:
        # List PDF files in the current folder
        folder_path = os.path.join(current_directory, folder)
        pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".pdf")]

        # Extract only the file names for display
        pdf_file_names = [os.path.basename(pdf_file) for pdf_file in pdf_files]

        # Create a multiselect widget to choose PDF files for the current folder
        selected_pdf_names = st.multiselect(f"Select PDFs for '{folder}':", pdf_file_names)

        # Map selected PDFs to their folder and file paths
        for selected_pdf_name in selected_pdf_names:
            selected_pdf_paths.append((folder, selected_pdf_name))

    if selected_pdf_paths:
        st.write("Selected PDFs:")
        for folder, pdf_name in selected_pdf_paths:
            st.write(pdf_name)

        # Create an "Index" PDF page with selected PDFs mentioned in order
        index_pdf_bytes = BytesIO()
        c = canvas.Canvas(index_pdf_bytes, pagesize=letter)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, 400, "Index")
        c.setFont("Helvetica", 12)
        y_position = 380
        for i, (_, pdf_name) in enumerate(selected_pdf_paths, start=1):
            c.drawString(100, y_position - i * 20, f"{i}. {pdf_name}")
        c.showPage()
        c.save()

        # Merge the selected PDFs and the "Index" PDF
        merged_pdf = PyPDF2.PdfMerger()

        # Append the first two selected PDFs
        for i in range(2):
            if i < len(selected_pdf_paths):
                folder, pdf_name = selected_pdf_paths[i]
                # Get the full path of the selected PDF using its name
                folder_path = os.path.join(current_directory, folder)
                pdf_path = os.path.join(folder_path, pdf_name)
                merged_pdf.append(pdf_path)

        # Insert the "Index" PDF as the third page
        index_pdf_bytes.seek(0)
        
        merged_pdf.append(index_pdf_bytes)

        # Append the rest of the selected PDFs
        for i in range(2, len(selected_pdf_paths)):
            folder, pdf_name = selected_pdf_paths[i]
            # Get the full path of the selected PDF using its name
            folder_path = os.path.join(current_directory, folder)
            pdf_path = os.path.join(folder_path, pdf_name)
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
