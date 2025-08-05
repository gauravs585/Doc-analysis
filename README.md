Doc-Analysis
Overview
Doc-analysis is a powerful and comprehensive document analysis tool that leverages large language models (LLMs) to provide in-depth insights into your documents. This application allows you to upload documents in various formats (PDF, DOCX, and TXT) and performs a wide range of analyses, including:

Basic Text Analysis: Get fundamental statistics about your document, such as word count, sentence count, character count, and unique words.

Sentiment Analysis: Determine the overall sentiment of the text (positive, negative, or neutral) with a confidence score.

Text Summarization: Generate a concise summary of the document, highlighting the key points.

Document Classification: Categorize your documents into predefined or custom categories.

AI-Powered Analysis with OpenAI: Utilize the power of OpenAI's GPT models for more advanced and nuanced analysis of your documents.

Data Visualization: Visualize the analysis results with interactive charts and graphs.

Features
File Upload: Supports PDF, DOCX, and TXT file formats.

Comprehensive Analysis: Performs a wide range of text analyses to provide a holistic understanding of your documents.

Customizable Classification: Allows you to define your own categories for document classification.

OpenAI Integration: Seamlessly integrates with the OpenAI API for advanced analysis.

Interactive Visualizations: Presents analysis results in an intuitive and easy-to-understand format.

Export Results: Export the analysis results to a JSON file for further use.

Technical Details
Dependencies
The project relies on a number of Python libraries to perform its analysis, including:

Streamlit: For creating the web application and user interface.

Pandas: For data manipulation and analysis.

PyPDF2: For extracting text from PDF files.

python-docx: For extracting text from DOCX files.

OpenAI: For interacting with the OpenAI API.

Transformers: For using pre-trained models for sentiment analysis, summarization, and classification.

Torch: As a backend for the Transformers library.

Plotly: For creating interactive visualizations.

NLTK: For natural language processing tasks such as tokenization and stopword removal.

Textstat: For calculating readability scores.

Development Environment
The project is configured to run in a Dev Container, which provides a consistent and reproducible development environment. The devcontainer.json file specifies the Docker image, extensions, and commands needed to set up the environment.

How to Run the Application
Clone the repository.

Install the required dependencies by running pip install -r requirements.txt.

Run the Streamlit application by executing the command streamlit run text.py.

Access the application in your web browser at the provided URL.

Future Enhancements
Support for more file formats: Add support for other document formats such as ODT and RTF.

Named Entity Recognition (NER): Implement NER to identify and extract entities such as people, organizations, and locations.

Topic Modeling: Use topic modeling techniques to identify the main topics in a document.

Improved error handling: Enhance the error handling to provide more informative messages to the user.

User authentication: Add user authentication to allow users to save their analysis results.
