import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import streamlit as st
from pathlib import Path
import PyPDF2
from docx import Document
import openai
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from textstat import flesch_reading_ease, flesch_kincaid_grade
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import re
from datetime import datetime

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data with error handling"""
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords')
    ]
    
    for data_path, data_name in required_data:
        try:
            nltk.data.find(data_path)
        except LookupError:
            try:
                print(f"Downloading {data_name}...")
                nltk.download(data_name, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download {data_name}: {e}")

# Download NLTK data
download_nltk_data()

class DocumentAnalyzer:
    """
    A comprehensive document analysis tool using various LLMs and NLP techniques
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the document analyzer with optional OpenAI API key"""
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
        
        # Initialize local models
        self.sentiment_analyzer = None
        self.summarizer = None
        self.classifier = None
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained models for analysis"""
        try:
            # Sentiment analysis model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            # Text summarization model
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn"
            )
            
            # Text classification model
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {e}")
    
    def extract_text_from_docx(self, docx_file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting DOCX text: {e}")
    
    def extract_text_from_txt(self, txt_file) -> str:
        """Extract text from TXT file"""
        try:
            return txt_file.read().decode('utf-8')
        except Exception as e:
            raise Exception(f"Error extracting TXT text: {e}")
    
    def basic_text_analysis(self, text: str) -> Dict[str, Any]:
        """Perform basic text analysis"""
        try:
            # Clean text
            clean_text = re.sub(r'[^\w\s]', '', text.lower())
            
            # Tokenization with fallback
            try:
                words = word_tokenize(clean_text)
                sentences = sent_tokenize(text)
            except LookupError:
                # Fallback tokenization if NLTK data is missing
                words = clean_text.split()
                sentences = text.split('.')
                sentences = [s.strip() for s in sentences if s.strip()]
            
            # Remove stopwords with fallback
            try:
                stop_words = set(stopwords.words('english'))
            except LookupError:
                # Basic English stopwords fallback
                stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                                'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'])
            
            filtered_words = [word for word in words if word.lower() not in stop_words and len(word) > 2]
            
            # Word frequency
            word_freq = Counter(filtered_words)
            
            # Readability scores with error handling
            try:
                readability_ease = flesch_reading_ease(text)
                readability_grade = flesch_kincaid_grade(text)
            except:
                readability_ease = 0
                readability_grade = 0
            
            return {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'character_count': len(text),
                'unique_words': len(set(words)),
                'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
                'most_common_words': word_freq.most_common(10),
                'readability_ease': readability_ease,
                'readability_grade': readability_grade
            }
        except Exception as e:
            return {"error": f"Basic analysis failed: {e}"}
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of the text"""
        if not self.sentiment_analyzer:
            return {"error": "Sentiment analyzer not available"}
        
        try:
            # Split text into chunks for analysis
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            sentiments = []
            
            for chunk in chunks[:10]:  # Limit to first 10 chunks
                result = self.sentiment_analyzer(chunk)
                sentiments.append(result[0])
            
            # Aggregate results
            labels = [s['label'] for s in sentiments]
            scores = [s['score'] for s in sentiments]
            
            # Map labels to standard format
            label_mapping = {'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'}
            mapped_labels = [label_mapping.get(label, label.lower()) for label in labels]
            
            sentiment_counts = Counter(mapped_labels)
            avg_score = sum(scores) / len(scores) if scores else 0
            
            return {
                'overall_sentiment': max(sentiment_counts, key=sentiment_counts.get),
                'sentiment_distribution': dict(sentiment_counts),
                'confidence_score': avg_score,
                'detailed_analysis': list(zip(mapped_labels, scores))
            }
            
        except Exception as e:
            return {"error": f"Sentiment analysis failed: {e}"}
    
    def summarize_text(self, text: str, max_length: int = 150) -> Dict[str, Any]:
        """Generate text summary"""
        if not self.summarizer:
            return {"error": "Summarizer not available"}
        
        try:
            # Truncate text if too long
            max_input_length = 1024
            if len(text) > max_input_length:
                text = text[:max_input_length]
            
            summary = self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)
            
            return {
                'summary': summary[0]['summary_text'],
                'original_length': len(text),
                'summary_length': len(summary[0]['summary_text']),
                'compression_ratio': len(summary[0]['summary_text']) / len(text)
            }
            
        except Exception as e:
            return {"error": f"Summarization failed: {e}"}
    
    def classify_document(self, text: str, candidate_labels: List[str]) -> Dict[str, Any]:
        """Classify document into categories"""
        if not self.classifier:
            return {"error": "Classifier not available"}
        
        try:
            # Truncate text if too long
            max_input_length = 1024
            if len(text) > max_input_length:
                text = text[:max_input_length]
            
            result = self.classifier(text, candidate_labels)
            
            return {
                'predicted_category': result['labels'][0],
                'confidence_scores': dict(zip(result['labels'], result['scores'])),
                'all_predictions': list(zip(result['labels'], result['scores']))
            }
            
        except Exception as e:
            return {"error": f"Classification failed: {e}"}
    
    def analyze_with_openai(self, text: str, analysis_type: str = "general") -> Dict[str, Any]:
        """Analyze text using OpenAI GPT"""
        if not self.openai_api_key:
            return {"error": "OpenAI API key not provided"}
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            prompts = {
                "general": f"Analyze the following document and provide insights about its content, tone, and key themes:\n\n{text[:2000]}",
                "legal": f"Analyze this legal document for key clauses, obligations, and potential risks:\n\n{text[:2000]}",
                "business": f"Analyze this business document for strategic insights, opportunities, and recommendations:\n\n{text[:2000]}",
                "academic": f"Analyze this academic text for methodology, findings, and theoretical contributions:\n\n{text[:2000]}"
            }
            
            prompt = prompts.get(analysis_type, prompts["general"])
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert document analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return {
                'analysis': response.choices[0].message.content,
                'model_used': 'gpt-3.5-turbo',
                'analysis_type': analysis_type
            }
            
        except Exception as e:
            return {"error": f"OpenAI analysis failed: {e}"}
    
    def generate_visualizations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualizations for analysis results"""
        visualizations = {}
        
        try:
            # Word frequency chart
            if 'basic_analysis' in analysis_results and 'most_common_words' in analysis_results['basic_analysis']:
                words, counts = zip(*analysis_results['basic_analysis']['most_common_words'])
                fig_words = px.bar(x=list(counts), y=list(words), orientation='h',
                                 title="Most Common Words", labels={'x': 'Frequency', 'y': 'Words'})
                visualizations['word_frequency'] = fig_words
            
            # Sentiment distribution
            if 'sentiment_analysis' in analysis_results and 'sentiment_distribution' in analysis_results['sentiment_analysis']:
                sentiment_data = analysis_results['sentiment_analysis']['sentiment_distribution']
                fig_sentiment = px.pie(values=list(sentiment_data.values()), 
                                     names=list(sentiment_data.keys()),
                                     title="Sentiment Distribution")
                visualizations['sentiment_pie'] = fig_sentiment
            
            # Text statistics
            if 'basic_analysis' in analysis_results:
                stats = analysis_results['basic_analysis']
                fig_stats = go.Figure(data=[
                    go.Bar(x=['Words', 'Sentences', 'Unique Words'], 
                          y=[stats.get('word_count', 0), 
                             stats.get('sentence_count', 0), 
                             stats.get('unique_words', 0)])
                ])
                fig_stats.update_layout(title="Document Statistics")
                visualizations['document_stats'] = fig_stats
            
        except Exception as e:
            st.error(f"Error generating visualizations: {e}")
        
        return visualizations
    
    def export_results(self, analysis_results: Dict[str, Any], filename: str = None, export_dir: str = None) -> str:
        """Export analysis results to JSON"""
        # Create exports directory if it doesn't exist
        if not export_dir:
            export_dir = os.path.join(os.getcwd(), "exports")
        
        os.makedirs(export_dir, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"document_analysis_{timestamp}.json"
        
        # Full path for the export file
        full_path = os.path.join(export_dir, filename)
        
        try:
            # Convert non-serializable objects to strings
            exportable_results = {
                'export_info': {
                    'timestamp': datetime.now().isoformat(),
                    'export_location': full_path,
                    'analysis_summary': {
                        'total_analyses': len(analysis_results),
                        'analyses_included': list(analysis_results.keys())
                    }
                }
            }
            
            for key, value in analysis_results.items():
                if isinstance(value, dict):
                    exportable_results[key] = {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict)) else v 
                                             for k, v in value.items()}
                else:
                    exportable_results[key] = str(value) if not isinstance(value, (str, int, float, bool, list)) else value
            
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(exportable_results, f, indent=2, ensure_ascii=False)
            
            return full_path
            
        except Exception as e:
            raise Exception(f"Export failed: {e}")


def create_streamlit_app():
    """Create Streamlit web application"""
    st.set_page_config(page_title="Document Analysis with LLMs", page_icon="📄", layout="wide")
    
    st.title("📄 Document Analysis with LLMs")
    st.markdown("Upload a document and get comprehensive AI-powered analysis including sentiment, summarization, and classification.")
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    openai_api_key = st.sidebar.text_input("OpenAI API Key (Optional)", type="password")
    
    # Initialize analyzer
    analyzer = DocumentAnalyzer(openai_api_key if openai_api_key else None)
    
    # File upload
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['pdf', 'docx', 'txt'],
        help="Upload PDF, DOCX, or TXT files for analysis"
    )
    
    if uploaded_file is not None:
        # Extract text based on file type
        try:
            if uploaded_file.type == "application/pdf":
                text = analyzer.extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = analyzer.extract_text_from_docx(uploaded_file)
            else:
                text = analyzer.extract_text_from_txt(uploaded_file)
            
            if text.strip():
                # Store extracted text in session state
                st.session_state.extracted_text = text
                
                # Display extracted text preview
                st.header("📝 Document Preview")
                with st.expander("View extracted text"):
                    st.text_area("Extracted Text", text[:1000] + "..." if len(text) > 1000 else text, height=200)
                
                # Analysis options
                st.header("🔍 Analysis Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    basic_analysis = st.checkbox("Basic Text Analysis", value=True, key="basic_cb")
                    sentiment_analysis = st.checkbox("Sentiment Analysis", value=True, key="sentiment_cb")
                    text_summarization = st.checkbox("Text Summarization", value=True, key="summary_cb")
                
                with col2:
                    document_classification = st.checkbox("Document Classification", value=True, key="class_cb")
                    openai_analysis = st.checkbox("OpenAI Analysis", value=bool(openai_api_key), key="openai_cb")
                    generate_visualizations = st.checkbox("Generate Visualizations", value=True, key="viz_cb")
                
                # Classification categories
                categories = []
                if document_classification:
                    st.subheader("Classification Categories")
                    default_categories = ["business", "legal", "academic", "technical", "personal", "news", "medical"]
                    categories = st.multiselect(
                        "Select categories for classification:",
                        default_categories,
                        default=default_categories[:4],
                        key="categories_select"
                    )
                
                # OpenAI analysis type
                analysis_type = "general"
                if openai_analysis and openai_api_key:
                    analysis_type = st.selectbox(
                        "OpenAI Analysis Type:",
                        ["general", "legal", "business", "academic"],
                        key="analysis_type_select"
                    )
                
                # Run analysis button
                run_analysis = st.button("🚀 Run Analysis", type="primary", key="run_analysis_btn")
                
                # Run analysis only when button is clicked
                if run_analysis and not st.session_state.analysis_complete:
                    analysis_results = {}
                    
                    with st.spinner("Analyzing document..."):
                        # Basic analysis
                        if basic_analysis:
                            with st.status("Running basic text analysis..."):
                                analysis_results['basic_analysis'] = analyzer.basic_text_analysis(text)
                        
                        # Sentiment analysis
                        if sentiment_analysis:
                            with st.status("Analyzing sentiment..."):
                                analysis_results['sentiment_analysis'] = analyzer.analyze_sentiment(text)
                        
                        # Text summarization
                        if text_summarization:
                            with st.status("Generating summary..."):
                                analysis_results['summary'] = analyzer.summarize_text(text)
                        
                        # Document classification
                        if document_classification and categories:
                            with st.status("Classifying document..."):
                                analysis_results['classification'] = analyzer.classify_document(text, categories)
                        
                        # OpenAI analysis
                        if openai_analysis and openai_api_key:
                            with st.status("Running OpenAI analysis..."):
                                analysis_results['openai_analysis'] = analyzer.analyze_with_openai(text, analysis_type)
                    
                    # Store results in session state
                    st.session_state.analysis_results = analysis_results
                    st.session_state.analysis_complete = True
                
                # Display results if analysis is complete
                if st.session_state.analysis_complete and st.session_state.analysis_results:
                    analysis_results = st.session_state.analysis_results
                    
                    # Display results
                    st.header("📊 Analysis Results")
                    
                    # Basic Analysis
                    if 'basic_analysis' in analysis_results and 'error' not in analysis_results['basic_analysis']:
                        st.subheader("📈 Basic Text Statistics")
                        basic = analysis_results['basic_analysis']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Word Count", basic['word_count'])
                        with col2:
                            st.metric("Sentences", basic['sentence_count'])
                        with col3:
                            st.metric("Unique Words", basic['unique_words'])
                        with col4:
                            st.metric("Readability Score", f"{basic['readability_ease']:.1f}" if isinstance(basic['readability_ease'], (int, float)) else "N/A")
                    
                    # Sentiment Analysis
                    if 'sentiment_analysis' in analysis_results and 'error' not in analysis_results['sentiment_analysis']:
                        st.subheader("😊 Sentiment Analysis")
                        sentiment = analysis_results['sentiment_analysis']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Overall Sentiment", sentiment['overall_sentiment'].title())
                            st.metric("Confidence", f"{sentiment['confidence_score']:.2f}")
                        with col2:
                            if 'sentiment_distribution' in sentiment:
                                st.write("**Sentiment Distribution:**")
                                for emotion, count in sentiment['sentiment_distribution'].items():
                                    st.write(f"- {emotion.title()}: {count}")
                    
                    # Summary
                    if 'summary' in analysis_results and 'error' not in analysis_results['summary']:
                        st.subheader("📋 Document Summary")
                        summary = analysis_results['summary']
                        st.write(summary['summary'])
                        st.caption(f"Compression ratio: {summary['compression_ratio']:.2%}")
                    
                    # Classification
                    if 'classification' in analysis_results and 'error' not in analysis_results['classification']:
                        st.subheader("🏷️ Document Classification")
                        classification = analysis_results['classification']
                        st.write(f"**Predicted Category:** {classification['predicted_category']}")
                        
                        st.write("**Confidence Scores:**")
                        for category, score in classification['confidence_scores'].items():
                            st.write(f"- {category}: {score:.2%}")
                    
                    # OpenAI Analysis
                    if 'openai_analysis' in analysis_results and 'error' not in analysis_results['openai_analysis']:
                        st.subheader("🤖 AI-Powered Analysis")
                        openai_result = analysis_results['openai_analysis']
                        st.write(openai_result['analysis'])
                    
                    # Visualizations
                    if generate_visualizations:
                        st.header("📊 Visualizations")
                        visualizations = analyzer.generate_visualizations(analysis_results)
                        
                        for viz_name, fig in visualizations.items():
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Export results section
                    st.header("💾 Export Results")
                    
                    # Export configuration
                    col1, col2 = st.columns(2)
                    with col1:
                        export_format = st.selectbox("Export Format", ["JSON", "CSV (Basic Stats)", "TXT (Summary)"], key="export_format_select")
                    with col2:
                        custom_filename = st.text_input("Custom Filename (optional)", placeholder="my_analysis", key="custom_filename_input")
                    
                    # Show current export directory
                    current_dir = os.getcwd()
                    export_dir = os.path.join(current_dir, "exports")
                    st.info(f"📁 **Export Location**: `{export_dir}`")
                    
                    # Export button
                    if st.button("📊 Export Analysis Results", key="export_btn"):
                        try:
                            if export_format == "JSON":
                                filename = custom_filename + ".json" if custom_filename else None
                                full_path = analyzer.export_results(analysis_results, filename)
                                st.success(f"✅ Results exported successfully!")
                                st.write(f"**File saved to**: `{full_path}`")
                                
                                # Provide download button
                                with open(full_path, 'r', encoding='utf-8') as f:
                                    json_data = f.read()
                                
                                st.download_button(
                                    label="⬇️ Download JSON File",
                                    data=json_data,
                                    file_name=os.path.basename(full_path),
                                    mime="application/json",
                                    key="download_json_btn"
                                )
                            
                            elif export_format == "CSV (Basic Stats)":
                                # Export basic stats as CSV
                                if 'basic_analysis' in analysis_results:
                                    basic_stats = analysis_results['basic_analysis']
                                    df = pd.DataFrame([basic_stats])
                                    
                                    filename = custom_filename + ".csv" if custom_filename else f"basic_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                    full_path = os.path.join(export_dir, filename)
                                    os.makedirs(export_dir, exist_ok=True)
                                    
                                    df.to_csv(full_path, index=False)
                                    st.success(f"✅ CSV exported to: `{full_path}`")
                                    
                                    st.download_button(
                                        label="⬇️ Download CSV File",
                                        data=df.to_csv(index=False),
                                        file_name=filename,
                                        mime="text/csv",
                                        key="download_csv_btn"
                                    )
                                else:
                                    st.error("No basic analysis data available for CSV export")
                            
                            elif export_format == "TXT (Summary)":
                                # Export as readable text summary
                                summary_text = f"Document Analysis Report\n"
                                summary_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                summary_text += "=" * 50 + "\n\n"
                                
                                for analysis_type, results in analysis_results.items():
                                    summary_text += f"{analysis_type.upper().replace('_', ' ')}\n"
                                    summary_text += "-" * 30 + "\n"
                                    
                                    if isinstance(results, dict) and 'error' not in results:
                                        for key, value in results.items():
                                            if isinstance(value, (str, int, float)):
                                                summary_text += f"{key}: {value}\n"
                                            elif isinstance(value, list) and key == 'most_common_words':
                                                summary_text += f"Top words: {', '.join([f'{w}({c})' for w, c in value[:5]])}\n"
                                    summary_text += "\n"
                                
                                filename = custom_filename + ".txt" if custom_filename else f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                                
                                st.download_button(
                                    label="⬇️ Download TXT File",
                                    data=summary_text,
                                    file_name=filename,
                                    mime="text/plain",
                                    key="download_txt_btn"
                                )
                                
                        except Exception as e:
                            st.error(f"❌ Export failed: {e}")
                    
                    # Reset analysis button
                    if st.button("🔄 Run New Analysis", key="reset_btn"):
                        st.session_state.analysis_results = None
                        st.session_state.analysis_complete = False
                        st.rerun()
                    
                    # Error handling for individual analyses
                    for analysis_name, result in analysis_results.items():
                        if isinstance(result, dict) and 'error' in result:
                            st.error(f"{analysis_name.title()} Error: {result['error']}")
            
            else:
                st.error("Could not extract text from the uploaded file.")
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    # Instructions
    st.sidebar.header("📖 Instructions")
    st.sidebar.markdown("""
    1. Upload a PDF, DOCX, or TXT file
    2. Configure analysis options
    3. Add OpenAI API key for advanced analysis
    4. Click 'Run Analysis' to start
    5. View results and visualizations
    6. Export results as JSON
    """)
    
    st.sidebar.header("🔧 Features")
    st.sidebar.markdown("""
    - **Basic Analysis**: Word count, readability
    - **Sentiment Analysis**: Emotion detection
    - **Summarization**: Auto-generated summaries
    - **Classification**: Document categorization
    - **AI Analysis**: OpenAI-powered insights
    - **Visualizations**: Interactive charts
    - **Export**: Save results as JSON
    """)


if __name__ == "__main__":
    # For running as Streamlit app
    create_streamlit_app()