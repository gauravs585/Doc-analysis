import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import streamlit as st
from pathlib import Path
import PyPDF2
from docx import Document
import plotly.express as px
import plotly.graph_objects as go
from textstat import flesch_reading_ease, flesch_kincaid_grade
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import re
from datetime import datetime
import requests
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Download required NLTK data
@st.cache_resource
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
    A comprehensive document analysis tool with improved error handling
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the document analyzer with optional OpenAI API key"""
        self.openai_api_key = openai_api_key
        
        # Initialize models with better error handling
        self.sentiment_analyzer = None
        self.summarizer = None
        self.classifier = None
        self.models_loaded = False
        self._load_models()
        
    @st.cache_resource
    def _load_models(_self):
        """Load pre-trained models for analysis with improved error handling"""
        models_status = {
            'sentiment': False,
            'summarizer': False,
            'classifier': False
        }
        
        try:
            # Try loading models with better error handling
            from transformers import pipeline
            import torch
            
            # Set device to CPU to avoid CUDA issues
            device = "cpu"
            
            # Sentiment analysis model - try multiple options
            try:
                _self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=device,
                    return_all_scores=True
                )
                models_status['sentiment'] = True
            except Exception as e:
                print(f"Failed to load sentiment model: {e}")
                try:
                    # Fallback to a simpler model
                    _self.sentiment_analyzer = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        device=device,
                        top_k=None  # Updated parameter instead of return_all_scores
                    )
                    models_status['sentiment'] = True
                except Exception as e2:
                    print(f"Failed to load fallback sentiment model: {e2}")
            
            # Text summarization model
            try:
                _self.summarizer = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-12-6",
                    device=device
                )
                models_status['summarizer'] = True
            except Exception as e:
                print(f"Failed to load summarization model: {e}")
                try:
                    # Fallback to smaller model
                    _self.summarizer = pipeline(
                        "summarization",
                        model="sshleifer/distilbart-cnn-12-6",
                        device=device
                    )
                    models_status['summarizer'] = True
                except Exception as e2:
                    print(f"Failed to load fallback summarization model: {e2}")
            
            # Text classification model
            try:
                _self.classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=device
                )
                models_status['classifier'] = True
            except Exception as e:
                print(f"Failed to load classification model: {e}")
                try:
                    # Fallback to smaller model
                    _self.classifier = pipeline(
                        "zero-shot-classification",
                        model="typeform/distilbert-base-uncased-mnli",
                        device=device
                    )
                    models_status['classifier'] = True
                except Exception as e2:
                    print(f"Failed to load fallback classification model: {e2}")
            
        except ImportError as e:
            st.error(f"Error importing transformers: {e}. Some features may not be available.")
        except Exception as e:
            st.error(f"Error loading models: {e}")
        
        _self.models_status = models_status
        _self.models_loaded = any(models_status.values())
        return models_status
    
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
            
            # Character frequency analysis
            char_freq = Counter(text.lower().replace(' ', ''))
            most_common_chars = char_freq.most_common(5)
            
            # Average word length
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            
            return {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'character_count': len(text),
                'unique_words': len(set(words)),
                'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
                'avg_word_length': round(avg_word_length, 2),
                'most_common_words': word_freq.most_common(10),
                'most_common_chars': most_common_chars,
                'readability_ease': round(readability_ease, 2) if isinstance(readability_ease, (int, float)) else 0,
                'readability_grade': round(readability_grade, 2) if isinstance(readability_grade, (int, float)) else 0,
                'lexical_diversity': len(set(words)) / len(words) if words else 0
            }
        except Exception as e:
            return {"error": f"Basic analysis failed: {e}"}
    
    def analyze_sentiment_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback sentiment analysis using simple word matching"""
        try:
            # Simple sentiment lexicon
            positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                                'happy', 'joy', 'love', 'like', 'best', 'perfect', 'awesome',
                                'brilliant', 'outstanding', 'superb', 'magnificent', 'marvelous'])
            
            negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'worst',
                                'terrible', 'disgusting', 'pathetic', 'useless', 'disappointing',
                                'sad', 'angry', 'frustrated', 'annoying', 'stupid', 'ridiculous'])
            
            words = re.findall(r'\b\w+\b', text.lower())
            
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            total_sentiment_words = positive_count + negative_count
            
            if total_sentiment_words == 0:
                sentiment = 'neutral'
                confidence = 0.5
            elif positive_count > negative_count:
                sentiment = 'positive'
                confidence = positive_count / total_sentiment_words
            elif negative_count > positive_count:
                sentiment = 'negative'
                confidence = negative_count / total_sentiment_words
            else:
                sentiment = 'neutral'
                confidence = 0.5
            
            return {
                'overall_sentiment': sentiment,
                'sentiment_distribution': {
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': len(words) - total_sentiment_words
                },
                'confidence_score': round(confidence, 2),
                'method': 'fallback_lexicon'
            }
        except Exception as e:
            return {"error": f"Fallback sentiment analysis failed: {e}"}
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of the text"""
        if not self.sentiment_analyzer:
            st.warning("Using fallback sentiment analysis method")
            return self.analyze_sentiment_fallback(text)
        
        try:
            # Split text into chunks for analysis
            max_length = 512  # Most models have this limit
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            sentiments = []
            
            for chunk in chunks[:5]:  # Limit to first 5 chunks to avoid timeout
                if len(chunk.strip()) > 10:  # Only analyze meaningful chunks
                    result = self.sentiment_analyzer(chunk)
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], list):
                            # Handle return_all_scores=True format
                            sentiments.extend(result[0])
                        else:
                            sentiments.append(result[0])
            
            if not sentiments:
                return self.analyze_sentiment_fallback(text)
            
            # Process results
            if 'score' in sentiments[0]:
                # Standard format
                labels = [s['label'] for s in sentiments]
                scores = [s['score'] for s in sentiments]
            else:
                # Alternative format
                labels = [s.get('label', 'unknown') for s in sentiments]
                scores = [s.get('score', 0.5) for s in sentiments]
            
            # Map labels to standard format
            label_mapping = {
                'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive',
                'NEGATIVE': 'negative', 'POSITIVE': 'positive', 'NEUTRAL': 'neutral'
            }
            mapped_labels = [label_mapping.get(label.upper(), label.lower()) for label in labels]
            
            sentiment_counts = Counter(mapped_labels)
            avg_score = sum(scores) / len(scores) if scores else 0.5
            
            return {
                'overall_sentiment': max(sentiment_counts, key=sentiment_counts.get),
                'sentiment_distribution': dict(sentiment_counts),
                'confidence_score': round(avg_score, 2),
                'detailed_analysis': list(zip(mapped_labels, scores)),
                'method': 'transformer_model'
            }
            
        except Exception as e:
            st.warning(f"Model-based sentiment analysis failed, using fallback: {e}")
            return self.analyze_sentiment_fallback(text)
    
    def summarize_text_fallback(self, text: str, max_sentences: int = 3) -> Dict[str, Any]:
        """Fallback text summarization using extractive method"""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= max_sentences:
                return {
                    'summary': text,
                    'original_length': len(text),
                    'summary_length': len(text),
                    'compression_ratio': 1.0,
                    'method': 'no_compression_needed'
                }
            
            # Simple extractive summarization - take first, middle, and last sentences
            indices = [0, len(sentences)//2, len(sentences)-1]
            summary_sentences = [sentences[i] for i in indices if i < len(sentences)]
            summary = ' '.join(summary_sentences)
            
            return {
                'summary': summary,
                'original_length': len(text),
                'summary_length': len(summary),
                'compression_ratio': len(summary) / len(text),
                'method': 'extractive_fallback'
            }
        except Exception as e:
            return {"error": f"Fallback summarization failed: {e}"}
    
    def summarize_text(self, text: str, max_length: int = 150) -> Dict[str, Any]:
        """Generate text summary"""
        if not self.summarizer:
            st.warning("Using fallback summarization method")
            return self.summarize_text_fallback(text)
        
        try:
            # Truncate text if too long
            max_input_length = 1024
            if len(text) > max_input_length:
                text = text[:max_input_length]
            
            # Ensure minimum length for summarization
            if len(text) < 100:
                return {
                    'summary': text,
                    'original_length': len(text),
                    'summary_length': len(text),
                    'compression_ratio': 1.0,
                    'method': 'too_short_to_summarize'
                }
            
            summary = self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)
            
            return {
                'summary': summary[0]['summary_text'],
                'original_length': len(text),
                'summary_length': len(summary[0]['summary_text']),
                'compression_ratio': len(summary[0]['summary_text']) / len(text),
                'method': 'transformer_model'
            }
            
        except Exception as e:
            st.warning(f"Model-based summarization failed, using fallback: {e}")
            return self.summarize_text_fallback(text)
    
    def classify_document_fallback(self, text: str, candidate_labels: List[str]) -> Dict[str, Any]:
        """Fallback document classification using keyword matching"""
        try:
            # Simple keyword-based classification
            keywords = {
                'business': ['business', 'company', 'market', 'profit', 'revenue', 'sales', 'customer', 'strategy'],
                'legal': ['law', 'legal', 'court', 'contract', 'agreement', 'clause', 'liability', 'terms'],
                'academic': ['research', 'study', 'analysis', 'methodology', 'results', 'conclusion', 'hypothesis'],
                'technical': ['system', 'software', 'algorithm', 'data', 'technical', 'implementation', 'code'],
                'medical': ['patient', 'treatment', 'medical', 'health', 'diagnosis', 'therapy', 'clinical'],
                'news': ['news', 'report', 'today', 'yesterday', 'breaking', 'update', 'announced'],
                'personal': ['I', 'me', 'my', 'personal', 'diary', 'letter', 'email', 'friend']
            }
            
            text_lower = text.lower()
            scores = {}
            
            for label in candidate_labels:
                if label.lower() in keywords:
                    keyword_count = sum(1 for keyword in keywords[label.lower()] if keyword.lower() in text_lower)
                    scores[label] = keyword_count / len(keywords[label.lower()])
                else:
                    # Generic scoring for unknown categories
                    scores[label] = text_lower.count(label.lower()) / len(text_lower.split())
            
            if not scores or all(score == 0 for score in scores.values()):
                # Default classification
                predicted_category = candidate_labels[0] if candidate_labels else 'unknown'
                scores = {label: 1/len(candidate_labels) for label in candidate_labels}
            else:
                predicted_category = max(scores, key=scores.get)
            
            return {
                'predicted_category': predicted_category,
                'confidence_scores': scores,
                'all_predictions': sorted(scores.items(), key=lambda x: x[1], reverse=True),
                'method': 'keyword_fallback'
            }
            
        except Exception as e:
            return {"error": f"Fallback classification failed: {e}"}
    
    def classify_document(self, text: str, candidate_labels: List[str]) -> Dict[str, Any]:
        """Classify document into categories"""
        if not self.classifier:
            st.warning("Using fallback classification method")
            return self.classify_document_fallback(text, candidate_labels)
        
        try:
            # Truncate text if too long
            max_input_length = 1024
            if len(text) > max_input_length:
                text = text[:max_input_length]
            
            result = self.classifier(text, candidate_labels)
            
            return {
                'predicted_category': result['labels'][0],
                'confidence_scores': dict(zip(result['labels'], result['scores'])),
                'all_predictions': list(zip(result['labels'], result['scores'])),
                'method': 'transformer_model'
            }
            
        except Exception as e:
            st.warning(f"Model-based classification failed, using fallback: {e}")
            return self.classify_document_fallback(text, candidate_labels)
    
    def analyze_with_openai(self, text: str, analysis_type: str = "general") -> Dict[str, Any]:
        """Analyze text using OpenAI GPT"""
        if not self.openai_api_key:
            return {"error": "OpenAI API key not provided"}
        
        try:
            import openai
            client = openai.OpenAI(api_key=self.openai_api_key)
            
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
                words_data = analysis_results['basic_analysis']['most_common_words']
                if words_data:
                    words, counts = zip(*words_data)
                    fig_words = px.bar(x=list(counts), y=list(words), orientation='h',
                                     title="Most Common Words", labels={'x': 'Frequency', 'y': 'Words'})
                    fig_words.update_layout(height=400)
                    visualizations['word_frequency'] = fig_words
            
            # Sentiment distribution
            if 'sentiment_analysis' in analysis_results and 'sentiment_distribution' in analysis_results['sentiment_analysis']:
                sentiment_data = analysis_results['sentiment_analysis']['sentiment_distribution']
                if sentiment_data:
                    fig_sentiment = px.pie(values=list(sentiment_data.values()), 
                                         names=list(sentiment_data.keys()),
                                         title="Sentiment Distribution")
                    visualizations['sentiment_pie'] = fig_sentiment
            
            # Text statistics
            if 'basic_analysis' in analysis_results:
                stats = analysis_results['basic_analysis']
                metrics = ['Words', 'Sentences', 'Unique Words', 'Characters']
                values = [
                    stats.get('word_count', 0), 
                    stats.get('sentence_count', 0), 
                    stats.get('unique_words', 0),
                    stats.get('character_count', 0)
                ]
                
                fig_stats = go.Figure(data=[go.Bar(x=metrics, y=values)])
                fig_stats.update_layout(title="Document Statistics", height=400)
                visualizations['document_stats'] = fig_stats
            
            # Classification confidence (if available)
            if 'classification' in analysis_results and 'confidence_scores' in analysis_results['classification']:
                conf_data = analysis_results['classification']['confidence_scores']
                if conf_data:
                    categories = list(conf_data.keys())
                    confidences = list(conf_data.values())
                    
                    fig_conf = px.bar(x=categories, y=confidences,
                                     title="Classification Confidence Scores",
                                     labels={'x': 'Categories', 'y': 'Confidence'})
                    fig_conf.update_layout(height=400)
                    visualizations['classification_confidence'] = fig_conf
            
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
    
    # Display model status
    if hasattr(analyzer, 'models_status'):
        st.sidebar.header("🤖 Model Status")
        for model_name, status in analyzer.models_status.items():
            icon = "✅" if status else "❌"
            st.sidebar.write(f"{icon} {model_name.title()}")
        
        if not analyzer.models_loaded:
            st.sidebar.warning("⚠️ Some models failed to load. Fallback methods will be used.")
    
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
            with st.spinner("Extracting text from document..."):
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
                run_analysis = st.button
                # Run analysis button
                run_analysis = st.button("🚀 Run Analysis", type="primary", key="run_analysis_btn")
                
                if run_analysis:
                    st.session_state.analysis_results = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    analyses_to_run = []
                    if basic_analysis:
                        analyses_to_run.append("basic")
                    if sentiment_analysis:
                        analyses_to_run.append("sentiment")
                    if text_summarization:
                        analyses_to_run.append("summarization")
                    if document_classification:
                        analyses_to_run.append("classification")
                    if openai_analysis and openai_api_key:
                        analyses_to_run.append("openai")
                    
                    total_analyses = len(analyses_to_run)
                    
                    for i, analysis in enumerate(analyses_to_run):
                        progress = (i + 1) / total_analyses
                        progress_bar.progress(progress)
                        
                        if analysis == "basic":
                            status_text.text("Running basic text analysis...")
                            st.session_state.analysis_results['basic_analysis'] = analyzer.basic_text_analysis(text)
                        
                        elif analysis == "sentiment":
                            status_text.text("Analyzing sentiment...")
                            st.session_state.analysis_results['sentiment_analysis'] = analyzer.analyze_sentiment(text)
                        
                        elif analysis == "summarization":
                            status_text.text("Generating summary...")
                            st.session_state.analysis_results['text_summary'] = analyzer.summarize_text(text)
                        
                        elif analysis == "classification":
                            status_text.text("Classifying document...")
                            if categories:
                                st.session_state.analysis_results['classification'] = analyzer.classify_document(text, categories)
                        
                        elif analysis == "openai":
                            status_text.text("Running OpenAI analysis...")
                            st.session_state.analysis_results['openai_analysis'] = analyzer.analyze_with_openai(text, analysis_type)
                    
                    status_text.text("Analysis complete!")
                    st.session_state.analysis_complete = True
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    st.rerun()
            
            else:
                st.error("No text could be extracted from the uploaded file.")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.analysis_results:
        st.header("📊 Analysis Results")
        
        # Create tabs for different analyses
        tabs = []
        tab_names = []
        
        if 'basic_analysis' in st.session_state.analysis_results:
            tab_names.append("📈 Basic Analysis")
        if 'sentiment_analysis' in st.session_state.analysis_results:
            tab_names.append("😊 Sentiment")
        if 'text_summary' in st.session_state.analysis_results:
            tab_names.append("📝 Summary")
        if 'classification' in st.session_state.analysis_results:
            tab_names.append("🏷️ Classification")
        if 'openai_analysis' in st.session_state.analysis_results:
            tab_names.append("🤖 AI Analysis")
        
        if tab_names:
            tabs = st.tabs(tab_names)
            tab_index = 0
            
            # Basic Analysis Tab
            if 'basic_analysis' in st.session_state.analysis_results:
                with tabs[tab_index]:
                    basic_results = st.session_state.analysis_results['basic_analysis']
                    if 'error' not in basic_results:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Word Count", basic_results.get('word_count', 0))
                            st.metric("Unique Words", basic_results.get('unique_words', 0))
                        
                        with col2:
                            st.metric("Sentence Count", basic_results.get('sentence_count', 0))
                            st.metric("Avg Words/Sentence", f"{basic_results.get('avg_words_per_sentence', 0):.1f}")
                        
                        with col3:
                            st.metric("Character Count", basic_results.get('character_count', 0))
                            st.metric("Readability Score", basic_results.get('readability_ease', 0))
                        
                        # Most common words
                        if basic_results.get('most_common_words'):
                            st.subheader("Most Common Words")
                            words_df = pd.DataFrame(basic_results['most_common_words'], columns=['Word', 'Frequency'])
                            st.dataframe(words_df, use_container_width=True)
                        
                        # Readability information
                        st.subheader("Readability Analysis")
                        readability_score = basic_results.get('readability_ease', 0)
                        if readability_score >= 90:
                            readability_level = "Very Easy"
                        elif readability_score >= 80:
                            readability_level = "Easy"
                        elif readability_score >= 70:
                            readability_level = "Fairly Easy"
                        elif readability_score >= 60:
                            readability_level = "Standard"
                        elif readability_score >= 50:
                            readability_level = "Fairly Difficult"
                        elif readability_score >= 30:
                            readability_level = "Difficult"
                        else:
                            readability_level = "Very Difficult"
                        
                        st.write(f"**Readability Level:** {readability_level}")
                        st.write(f"**Grade Level:** {basic_results.get('readability_grade', 0)}")
                        st.write(f"**Lexical Diversity:** {basic_results.get('lexical_diversity', 0):.3f}")
                    else:
                        st.error(f"Basic analysis failed: {basic_results['error']}")
                tab_index += 1
            
            # Sentiment Analysis Tab
            if 'sentiment_analysis' in st.session_state.analysis_results:
                with tabs[tab_index]:
                    sentiment_results = st.session_state.analysis_results['sentiment_analysis']
                    if 'error' not in sentiment_results:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            overall_sentiment = sentiment_results.get('overall_sentiment', 'unknown')
                            confidence = sentiment_results.get('confidence_score', 0)
                            
                            # Color code sentiment
                            if overall_sentiment.lower() == 'positive':
                                st.success(f"**Overall Sentiment:** {overall_sentiment.title()}")
                            elif overall_sentiment.lower() == 'negative':
                                st.error(f"**Overall Sentiment:** {overall_sentiment.title()}")
                            else:
                                st.info(f"**Overall Sentiment:** {overall_sentiment.title()}")
                            
                            st.metric("Confidence Score", f"{confidence:.2f}")
                            st.write(f"**Analysis Method:** {sentiment_results.get('method', 'unknown')}")
                        
                        with col2:
                            # Show sentiment distribution as text
                            if 'sentiment_distribution' in sentiment_results:
                                st.subheader("Sentiment Breakdown")
                                sentiment_dist = sentiment_results['sentiment_distribution']
                                for sentiment, count in sentiment_dist.items():
                                    st.write(f"**{sentiment.title()}:** {count}")
                        
                        # Detailed analysis if available
                        if 'detailed_analysis' in sentiment_results:
                            st.subheader("Detailed Sentiment Analysis")
                            detailed_df = pd.DataFrame(
                                sentiment_results['detailed_analysis'],
                                columns=['Sentiment', 'Score']
                            )
                            st.dataframe(detailed_df, use_container_width=True)
                    else:
                        st.error(f"Sentiment analysis failed: {sentiment_results['error']}")
                tab_index += 1
            
            # Text Summary Tab
            if 'text_summary' in st.session_state.analysis_results:
                with tabs[tab_index]:
                    summary_results = st.session_state.analysis_results['text_summary']
                    if 'error' not in summary_results:
                        st.subheader("Document Summary")
                        st.write(summary_results.get('summary', 'No summary available'))
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Length", f"{summary_results.get('original_length', 0):,} chars")
                        with col2:
                            st.metric("Summary Length", f"{summary_results.get('summary_length', 0):,} chars")
                        with col3:
                            compression_ratio = summary_results.get('compression_ratio', 0)
                            st.metric("Compression Ratio", f"{compression_ratio:.1%}")
                        
                        st.write(f"**Method Used:** {summary_results.get('method', 'unknown')}")
                    else:
                        st.error(f"Summarization failed: {summary_results['error']}")
                tab_index += 1
            
            # Classification Tab
            if 'classification' in st.session_state.analysis_results:
                with tabs[tab_index]:
                    classification_results = st.session_state.analysis_results['classification']
                    if 'error' not in classification_results:
                        predicted_category = classification_results.get('predicted_category', 'unknown')
                        st.success(f"**Predicted Category:** {predicted_category.title()}")
                        
                        # Confidence scores
                        if 'confidence_scores' in classification_results:
                            st.subheader("Category Confidence Scores")
                            conf_scores = classification_results['confidence_scores']
                            
                            # Show as dataframe too
                            conf_df = pd.DataFrame(
                                list(conf_scores.items()),
                                columns=['Category', 'Confidence']
                            ).sort_values('Confidence', ascending=False)
                            st.dataframe(conf_df, use_container_width=True)
                        
                        st.write(f"**Method Used:** {classification_results.get('method', 'unknown')}")
                    else:
                        st.error(f"Classification failed: {classification_results['error']}")
                tab_index += 1
            
            # OpenAI Analysis Tab
            if 'openai_analysis' in st.session_state.analysis_results:
                with tabs[tab_index]:
                    openai_results = st.session_state.analysis_results['openai_analysis']
                    if 'error' not in openai_results:
                        st.subheader("AI-Powered Analysis")
                        st.write(openai_results.get('analysis', 'No analysis available'))
                        
                        st.info(f"**Model Used:** {openai_results.get('model_used', 'unknown')}")
                        st.info(f"**Analysis Type:** {openai_results.get('analysis_type', 'general')}")
                    else:
                        st.error(f"OpenAI analysis failed: {openai_results['error']}")
        
        # Generate visualizations if requested
        if st.checkbox("Generate Additional Visualizations", key="gen_viz_cb"):
            with st.spinner("Generating visualizations..."):
                visualizations = analyzer.generate_visualizations(st.session_state.analysis_results)
                
                if visualizations:
                    st.header("📊 Additional Visualizations")
                    
                    for viz_name, fig in visualizations.items():
                        st.subheader(viz_name.replace('_', ' ').title())
                        st.plotly_chart(fig, use_container_width=True, key=f"viz_{viz_name}")

        # Export results
        st.header("💾 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            export_filename = st.text_input(
                "Export filename (optional):",
                placeholder="analysis_results.json",
                key="export_filename"
            )
        
        with col2:
            if st.button("📥 Export to JSON", key="export_btn"):
                try:
                    exported_path = analyzer.export_results(
                        st.session_state.analysis_results,
                        filename=export_filename if export_filename else None
                    )
                    st.success(f"Results exported to: {exported_path}")
                    
                    # Provide download link
                    with open(exported_path, 'r', encoding='utf-8') as f:
                        st.download_button(
                            label="📱 Download JSON File",
                            data=f.read(),
                            file_name=os.path.basename(exported_path),
                            mime='application/json',
                            key="download_json"
                        )
                except Exception as e:
                    st.error(f"Export failed: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Document Analysis with LLMs | Built with Streamlit, Transformers, and OpenAI</p>
            <p>📄 Supports PDF, DOCX, and TXT files | 🤖 AI-powered analysis | 📊 Interactive visualizations</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    """Main function to run the Streamlit app"""
    try:
        create_streamlit_app()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page and try again.")


if __name__ == "__main__":
    main()


