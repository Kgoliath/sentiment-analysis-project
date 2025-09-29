import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import time
import io
import os
from typing import Optional, Tuple, Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# -------------------------------
# API TOKEN (ORIGINAL METHOD)
# -------------------------------
#try:
    #API_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
#except:
    #API_TOKEN = "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"

# -------------------------------
# CONFIGURATION AND CONSTANTS
# -------------------------------
class Config:
    """Configuration class for the sentiment analysis application"""
    
    # API Configuration
    API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
    
    # File and Text Limits
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_TEXT_LENGTH = 5000
    SUPPORTED_FILE_TYPES = ["txt", "csv"]
    
    # UI Configuration
    PAGE_TITLE = "Sentiment Analysis Dashboard"
    PAGE_ICON = "📊"
    
    # Sentiment Colors for Visualization
    SENTIMENT_COLORS = {
        'Positive': '#4ECDC4',
        'Negative': '#FF6B6B', 
        'Neutral': '#FFE66D'
    }
    
    # Sentiment Words for Highlighting
    POSITIVE_WORDS = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
        'love', 'like', 'awesome', 'brilliant', 'best', 'perfect', 'outstanding',
        'superb', 'magnificent', 'exceptional', 'incredible', 'marvelous'
    ]
    
    NEGATIVE_WORDS = [
        'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 
        'dislike', 'worst', 'poor', 'sucks', 'disappointing', 'pathetic',
        'dreadful', 'atrocious', 'abysmal', 'horrendous', 'appalling'
    ]
    
    # Label Mapping
    LABEL_MAPPING = {
        'LABEL_0': 'Negative', 
        'LABEL_1': 'Neutral', 
        'LABEL_2': 'Positive'
    }

# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "results_df": None,
        "file_uploaded": False,
        "uploaded_file_name": None,
        "current_input_method": "Type Text",
        "comparison_results": None,
        "text_results": None,
        "comparison_df": None,
        "comparison_file_results_dfs": (None, None),
        "comparison_texts": (None, None),
        "comparison_files": (None, None),
        "comparison_file_names": (None, None),
        "text_input_analyzed": None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def preprocess_text(text: Any) -> Optional[str]:
    """Clean and preprocess text before analysis"""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Basic text validation
    if len(text.strip()) < 3:
        return None
        
    return text.strip()

def validate_text_input(text: str) -> bool:
    """Validate text input before analysis"""
    if not text or not text.strip():
        st.warning("⚠️ Please enter some text to analyze")
        return False
    
    if len(text) > Config.MAX_TEXT_LENGTH:
        st.warning(f"⚠️ Text too long (max {Config.MAX_TEXT_LENGTH:,} characters)")
        return False
        
    return True

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts"""
    if not text1 or not text2:
        return 0.0
    
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except Exception:
        return 0.0

def generate_text_insights(input_text: str, result: Dict) -> Dict:
    """Generate comprehensive insights for a single text"""
    insights = {
        'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'text_length': len(input_text),
        'word_count': len(input_text.split()),
        'character_count': len(input_text),
        'dominant_sentiment': result['sentiment_label'],
        'confidence_score': result['confidence'],
        'confidence_level': 'High' if result['confidence'] > 0.8 else 'Medium' if result['confidence'] > 0.5 else 'Low',
        'sentiment_distribution': {},
        'detected_keywords': [],
        'analysis_summary': ''
    }
    
    # Add detailed sentiment scores
    if 'scores' in result:
        for score in result['scores']:
            sentiment_name = Config.LABEL_MAPPING[score['label']]
            insights['sentiment_distribution'][sentiment_name] = score['score']
    
    # Detect sentiment keywords
    words = input_text.lower().split()
    positive_found = [w for w in words if w.strip(string.punctuation) in Config.POSITIVE_WORDS]
    negative_found = [w for w in words if w.strip(string.punctuation) in Config.NEGATIVE_WORDS]
    
    insights['detected_keywords'] = {
        'positive_words': positive_found,
        'negative_words': negative_found,
        'total_sentiment_words': len(positive_found) + len(negative_found)
    }
    
    # Generate summary
    confidence_desc = insights['confidence_level'].lower()
    keyword_count = insights['detected_keywords']['total_sentiment_words']
    
    insights['analysis_summary'] = (
        f"The text shows {result['sentiment_label'].lower()} sentiment with {confidence_desc} confidence "
        f"({result['confidence']:.1%}). Analysis detected {keyword_count} sentiment-bearing words. "
        f"Text contains {insights['word_count']} words across {insights['character_count']} characters."
    )
    
    return insights

def generate_file_insights(results_df: pd.DataFrame, filename: str) -> Dict:
    """Generate comprehensive insights for file analysis"""
    sentiment_counts = results_df['Sentiment'].value_counts()
    sentiment_percentages = results_df['Sentiment'].value_counts(normalize=True)
    
    insights = {
        'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_file': filename,
        'total_texts_analyzed': len(results_df),
        'sentiment_distribution': {
            'counts': sentiment_counts.to_dict(),
            'percentages': sentiment_percentages.to_dict()
        },
        'confidence_statistics': {
            'overall_average': results_df['Confidence'].mean(),
            'overall_median': results_df['Confidence'].median(),
            'overall_std': results_df['Confidence'].std(),
            'by_sentiment': results_df.groupby('Sentiment')['Confidence'].agg(['mean', 'median', 'std']).to_dict()
        },
        'dominant_sentiment': sentiment_counts.index[0],
        'sentiment_consistency': sentiment_percentages.iloc[0],
        'high_confidence_texts': len(results_df[results_df['Confidence'] > 0.8]),
        'low_confidence_texts': len(results_df[results_df['Confidence'] < 0.5]),
        'analysis_summary': ''
    }
    
    # Generate comprehensive summary
    dominant = insights['dominant_sentiment']
    consistency = insights['sentiment_consistency']
    avg_conf = insights['confidence_statistics']['overall_average']
    
    insights['analysis_summary'] = (
        f"Analysis of {insights['total_texts_analyzed']} texts from {filename} reveals "
        f"{dominant.lower()} sentiment as dominant ({consistency:.1%} of texts). "
        f"Overall confidence average is {avg_conf:.1%}. "
        f"{insights['high_confidence_texts']} texts show high confidence (>80%), "
        f"while {insights['low_confidence_texts']} show low confidence (<50%)."
    )
    
    return insights

def create_enhanced_csv_data(data_type: str, **kwargs) -> str:
    """Create enhanced CSV with insights"""
    if data_type == "single_text":
        result = kwargs['result']
        input_text = kwargs['input_text']
        insights = generate_text_insights(input_text, result)
        
        # Create comprehensive CSV
        rows = [
            ["=== SENTIMENT ANALYSIS REPORT ==="],
            ["Analysis Timestamp", insights['analysis_timestamp']],
            [""],
            ["=== TEXT INFORMATION ==="],
            ["Original Text", input_text],
            ["Character Count", insights['character_count']],
            ["Word Count", insights['word_count']],
            [""],
            ["=== SENTIMENT ANALYSIS ==="],
            ["Dominant Sentiment", insights['dominant_sentiment']],
            ["Confidence Score", f"{insights['confidence_score']:.4f}"],
            ["Confidence Level", insights['confidence_level']],
            [""],
            ["=== DETAILED SCORES ==="],
            ["Sentiment", "Score", "Percentage"],
        ]
        
        for sentiment, score in insights['sentiment_distribution'].items():
            rows.append([sentiment, f"{score:.4f}", f"{score:.1%}"])
        
        rows.extend([
            [""],
            ["=== KEYWORD ANALYSIS ==="],
            ["Positive Keywords Found", ", ".join(insights['detected_keywords']['positive_words']) or "None"],
            ["Negative Keywords Found", ", ".join(insights['detected_keywords']['negative_words']) or "None"],
            ["Total Sentiment Keywords", insights['detected_keywords']['total_sentiment_words']],
            [""],
            ["=== ANALYSIS SUMMARY ==="],
            ["Summary", insights['analysis_summary']]
        ])
        
        return "\n".join([",".join(map(str, row)) for row in rows])
    
    elif data_type == "single_file":
        results_df = kwargs['results_df']
        filename = kwargs['filename']
        insights = generate_file_insights(results_df, filename)
        
        # Create header with insights
        header_rows = [
            ["=== SENTIMENT ANALYSIS REPORT ==="],
            ["Analysis Timestamp", insights['analysis_timestamp']],
            ["Source File", insights['source_file']],
            ["Total Texts Analyzed", insights['total_texts_analyzed']],
            [""],
            ["=== OVERALL STATISTICS ==="],
            ["Dominant Sentiment", insights['dominant_sentiment']],
            ["Sentiment Consistency", f"{insights['sentiment_consistency']:.1%}"],
            ["Average Confidence", f"{insights['confidence_statistics']['overall_average']:.1%}"],
            ["Median Confidence", f"{insights['confidence_statistics']['overall_median']:.1%}"],
            ["High Confidence Texts (>80%)", insights['high_confidence_texts']],
            ["Low Confidence Texts (<50%)", insights['low_confidence_texts']],
            [""],
            ["=== SENTIMENT DISTRIBUTION ==="],
            ["Sentiment", "Count", "Percentage"],
        ]
        
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            count = insights['sentiment_distribution']['counts'].get(sentiment, 0)
            percentage = insights['sentiment_distribution']['percentages'].get(sentiment, 0)
            header_rows.append([sentiment, count, f"{percentage:.1%}"])
        
        header_rows.extend([
            [""],
            ["=== ANALYSIS SUMMARY ==="],
            ["Summary", insights['analysis_summary']],
            [""],
            ["=== DETAILED RESULTS ==="]
        ])
        
        # Combine header with detailed results
        header_csv = "\n".join([",".join(map(str, row)) for row in header_rows])
        detailed_csv = results_df.to_csv(index=False)
        
        return header_csv + "\n" + detailed_csv
    
    elif data_type == "comparison":
        comparison_df = kwargs['comparison_df']
        comparison_type = kwargs.get('comparison_type', 'text')
        
        # Add timestamp and analysis info
        header_rows = [
            ["=== COMPARATIVE SENTIMENT ANALYSIS REPORT ==="],
            ["Analysis Timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ["Comparison Type", comparison_type.title()],
            [""],
            ["=== COMPARISON RESULTS ==="]
        ]
        
        header_csv = "\n".join([",".join(map(str, row)) for row in header_rows])
        detailed_csv = comparison_df.to_csv(index=False)
        
        return header_csv + "\n" + detailed_csv

def create_enhanced_json_data(data_type: str, **kwargs) -> str:
    """Create enhanced JSON with insights"""
    if data_type == "single_text":
        result = kwargs['result']
        input_text = kwargs['input_text']
        insights = generate_text_insights(input_text, result)
        
        enhanced_data = {
            "report_type": "Single Text Sentiment Analysis",
            "analysis_timestamp": insights['analysis_timestamp'],
            "text_analysis": {
                "original_text": input_text,
                "character_count": insights['character_count'],
                "word_count": insights['word_count']
            },
            "sentiment_results": {
                "dominant_sentiment": insights['dominant_sentiment'],
                "confidence_score": insights['confidence_score'],
                "confidence_level": insights['confidence_level'],
                "detailed_scores": insights['sentiment_distribution']
            },
            "keyword_analysis": insights['detected_keywords'],
            "analysis_summary": insights['analysis_summary'],
            "model_information": {
                "api_used": "Hugging Face - cardiffnlp/twitter-roberta-base-sentiment",
                "model_type": "RoBERTa-based transformer",
                "classification_classes": ["Negative", "Neutral", "Positive"]
            }
        }
        
        return json.dumps(enhanced_data, indent=2)
    
    elif data_type == "single_file":
        results_df = kwargs['results_df']
        filename = kwargs['filename']
        insights = generate_file_insights(results_df, filename)
        
        enhanced_data = {
            "report_type": "File Sentiment Analysis",
            "analysis_timestamp": insights['analysis_timestamp'],
            "source_information": {
                "filename": insights['source_file'],
                "total_texts_analyzed": insights['total_texts_analyzed']
            },
            "overall_statistics": {
                "dominant_sentiment": insights['dominant_sentiment'],
                "sentiment_consistency": insights['sentiment_consistency'],
                "confidence_statistics": insights['confidence_statistics'],
                "sentiment_distribution": insights['sentiment_distribution']
            },
            "quality_metrics": {
                "high_confidence_texts": insights['high_confidence_texts'],
                "low_confidence_texts": insights['low_confidence_texts']
            },
            "analysis_summary": insights['analysis_summary'],
            "detailed_results": results_df.to_dict(orient='records'),
            "model_information": {
                "api_used": "Hugging Face - cardiffnlp/twitter-roberta-base-sentiment",
                "model_type": "RoBERTa-based transformer",
                "classification_classes": ["Negative", "Neutral", "Positive"]
            }
        }
        
        return json.dumps(enhanced_data, indent=2)
    
    elif data_type == "comparison":
        comparison_df = kwargs['comparison_df']
        comparison_type = kwargs.get('comparison_type', 'text')
        
        enhanced_data = {
            "report_type": "Comparative Sentiment Analysis",
            "analysis_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "comparison_type": comparison_type,
            "comparison_results": comparison_df.to_dict(orient='records'),
            "model_information": {
                "api_used": "Hugging Face - cardiffnlp/twitter-roberta-base-sentiment",
                "model_type": "RoBERTa-based transformer",
                "classification_classes": ["Negative", "Neutral", "Positive"]
            }
        }
        
        return json.dumps(enhanced_data, indent=2)

def create_pdf_report(data_type: str, **kwargs) -> bytes:
    """Create comprehensive PDF report"""
    buffer = io.BytesIO()
    
    with PdfPages(buffer) as pdf:
        if data_type == "single_text":
            result = kwargs['result']
            input_text = kwargs['input_text']
            insights = generate_text_insights(input_text, result)
            
            # Page 1: Summary
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle('Sentiment Analysis Report', fontsize=16, fontweight='bold')
            
            # Sentiment pie chart
            sentiments = list(insights['sentiment_distribution'].keys())
            scores = list(insights['sentiment_distribution'].values())
            colors = [Config.SENTIMENT_COLORS[s] for s in sentiments]
            
            ax1.pie(scores, labels=sentiments, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Sentiment Distribution')
            
            # Confidence bar chart
            ax2.bar(sentiments, scores, color=colors)
            ax2.set_title('Confidence Scores')
            ax2.set_ylabel('Confidence')
            ax2.set_ylim(0, 1)
            
            # Text statistics
            ax3.text(0.1, 0.8, f"Analysis Timestamp: {insights['analysis_timestamp']}", transform=ax3.transAxes)
            ax3.text(0.1, 0.7, f"Text Length: {insights['character_count']} characters", transform=ax3.transAxes)
            ax3.text(0.1, 0.6, f"Word Count: {insights['word_count']} words", transform=ax3.transAxes)
            ax3.text(0.1, 0.5, f"Dominant Sentiment: {insights['dominant_sentiment']}", transform=ax3.transAxes)
            ax3.text(0.1, 0.4, f"Confidence: {insights['confidence_score']:.1%}", transform=ax3.transAxes)
            ax3.text(0.1, 0.3, f"Confidence Level: {insights['confidence_level']}", transform=ax3.transAxes)
            ax3.set_title('Analysis Summary')
            ax3.axis('off')
            
            # Keywords found
            pos_keywords = ", ".join(insights['detected_keywords']['positive_words'][:5]) or "None"
            neg_keywords = ", ".join(insights['detected_keywords']['negative_words'][:5]) or "None"
            
            ax4.text(0.1, 0.8, "Keyword Analysis:", transform=ax4.transAxes, fontweight='bold')
            ax4.text(0.1, 0.6, f"Positive: {pos_keywords}", transform=ax4.transAxes, wrap=True)
            ax4.text(0.1, 0.4, f"Negative: {neg_keywords}", transform=ax4.transAxes, wrap=True)
            ax4.text(0.1, 0.2, f"Total Keywords: {insights['detected_keywords']['total_sentiment_words']}", transform=ax4.transAxes)
            ax4.set_title('Detected Keywords')
            ax4.axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Text content and summary
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.1, 0.95, 'Analyzed Text:', transform=ax.transAxes, fontsize=14, fontweight='bold')
            
            # Wrap text for display
            import textwrap
            wrapped_text = textwrap.fill(input_text, width=80)
            ax.text(0.1, 0.85, wrapped_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
            
            ax.text(0.1, 0.3, 'Analysis Summary:', transform=ax.transAxes, fontsize=14, fontweight='bold')
            wrapped_summary = textwrap.fill(insights['analysis_summary'], width=80)
            ax.text(0.1, 0.25, wrapped_summary, transform=ax.transAxes, fontsize=10, verticalalignment='top')
            
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
        elif data_type == "single_file":
            results_df = kwargs['results_df']
            filename = kwargs['filename']
            insights = generate_file_insights(results_df, filename)
            
            # Page 1: Overview
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle(f'File Analysis Report: {filename}', fontsize=16, fontweight='bold')
            
            # Sentiment distribution pie chart
            counts = list(insights['sentiment_distribution']['counts'].values())
            labels = list(insights['sentiment_distribution']['counts'].keys())
            colors = [Config.SENTIMENT_COLORS[label] for label in labels]
            
            ax1.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Sentiment Distribution')
            
            # Confidence histogram
            ax2.hist(results_df['Confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_title('Confidence Distribution')
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Frequency')
            
            # Statistics
            ax3.text(0.1, 0.9, f"Total Texts: {insights['total_texts_analyzed']}", transform=ax3.transAxes)
            ax3.text(0.1, 0.8, f"Dominant Sentiment: {insights['dominant_sentiment']}", transform=ax3.transAxes)
            ax3.text(0.1, 0.7, f"Average Confidence: {insights['confidence_statistics']['overall_average']:.1%}", transform=ax3.transAxes)
            ax3.text(0.1, 0.6, f"High Confidence: {insights['high_confidence_texts']}", transform=ax3.transAxes)
            ax3.text(0.1, 0.5, f"Low Confidence: {insights['low_confidence_texts']}", transform=ax3.transAxes)
            ax3.set_title('Key Statistics')
            ax3.axis('off')
            
            # Summary
            import textwrap
            wrapped_summary = textwrap.fill(insights['analysis_summary'], width=40)
            ax4.text(0.1, 0.8, wrapped_summary, transform=ax4.transAxes, fontsize=10)
            ax4.set_title('Analysis Summary')
            ax4.axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    buffer.seek(0)
    return buffer.read()

# -------------------------------
# SENTIMENT ANALYSIS CLASS
# -------------------------------
class SentimentAnalyzer:
    """Main sentiment analysis class using public Hugging Face API"""
    
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def _query_huggingface(_self, payload: Dict[str, Any]) -> Optional[Dict]:
        """Query Hugging Face API with retry logic - NO TOKEN REQUIRED"""
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Try without any headers first (public access)
                response = requests.post(_self.api_url, json=payload, timeout=60)
                
                # If we get a 503 (model loading), wait and retry
                if response.status_code == 503:
                    estimated_time = response.json().get('estimated_time', 10)
                    st.info(f"🔄 Model is loading, please wait {estimated_time} seconds...")
                    time.sleep(min(estimated_time, 30))
                    continue
                
                # If unauthorized, try with token as fallback (but don't require it)
                if response.status_code in [401, 403]:
                    try:
                        # Try with token if available, but don't fail if not
                        API_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN", "")
                        if API_TOKEN:
                            headers = {"Authorization": f"Bearer {API_TOKEN}"}
                            response = requests.post(_self.api_url, headers=headers, json=payload, timeout=60)
                        else:
                            # If no token, continue with public access
                            continue
                    except:
                        # If token fails, continue with public access
                        continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay)
                else:
                    st.error(f"❌ API request failed after {max_retries} attempts: {e}")
                    return None
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                return None
    
    def analyze_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze sentiment of a single text"""
        processed_text = preprocess_text(text)
        if not processed_text:
            return None
            
        result = self._query_huggingface({"inputs": processed_text})
        if not result:
            return None
            
        if isinstance(result, dict) and "error" in result:
            error_msg = result.get('error', 'Unknown error')
            if "loading" in error_msg.lower():
                st.warning("🔄 Model is still loading, please try again in 30 seconds")
            else:
                st.warning(f"⚠️ API Error: {error_msg}")
            return None
        elif isinstance(result, list) and result:
            try:
                max_score = max(result[0], key=lambda x: x['score'])
                sentiment_label = Config.LABEL_MAPPING[max_score['label']]
                
                return {
                    'sentiment': max_score['label'],
                    'sentiment_label': sentiment_label,
                    'confidence': max_score['score'],
                    'scores': result[0],
                    'sentiment_counts': {
                        sentiment_label: 1, 
                        **{k: 0 for k in ['Negative', 'Neutral', 'Positive'] if k != sentiment_label}
                    }
                }
            except (KeyError, IndexError, TypeError) as e:
                st.error(f"❌ Error parsing API response: {e}")
                return None
        return None
    
    def analyze_file_with_progress(self, file) -> Tuple[Optional[Dict], Optional[pd.DataFrame], Optional[str]]:
        """Analyze file with comprehensive error handling and progress tracking"""
        if file is None:
            return None, None, "No file provided"
        
        # Validate file size
        if file.size > Config.MAX_FILE_SIZE:
            return None, None, f"File too large (max {Config.MAX_FILE_SIZE // (1024*1024)}MB)"
        
        try:
            file_content = file.getvalue()
            
            if file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(io.BytesIO(file_content))
                except pd.errors.EmptyDataError:
                    return None, None, "CSV file is empty or corrupted"
                except pd.errors.ParserError as e:
                    return None, None, f"CSV parsing error: {str(e)}"
                
                if df.empty:
                    return None, None, "CSV file is empty"
                if 'text' not in df.columns:
                    return None, None, "CSV must have a 'text' column"
                texts = df['text'].astype(str).tolist()
            else:
                try:
                    content = file_content.decode("utf-8")
                except UnicodeDecodeError:
                    return None, None, "File encoding not supported. Please use UTF-8."
                
                texts = [line.strip() for line in content.splitlines() if line.strip()]
                if not texts:
                    return None, None, "Text file is empty"
            
            # Process texts with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            for i, text in enumerate(texts):
                progress_bar.progress((i + 1) / len(texts))
                status_text.text(f"Analyzing text {i + 1} of {len(texts)}...")
                
                res = self.analyze_text(text)
                if res:
                    results.append((text, res))
                # Add small delay to avoid rate limiting
                time.sleep(0.5)
            
            progress_bar.empty()
            status_text.empty()
            
            if not results:
                return None, None, "No valid text to analyze"
            
            # Calculate statistics
            sentiment_counts = {'Negative': 0, 'Neutral': 0, 'Positive': 0}
            total_conf = 0
            
            for _, res in results:
                sentiment_counts[res['sentiment_label']] += 1
                total_conf += res['confidence']
            
            dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            avg_conf = total_conf / len(results) if results else 0
            
            # Create detailed results dataframe
            detailed_results = []
            for text, res in results:
                detailed_results.append({
                    'Text': text,
                    'Sentiment': res['sentiment_label'],
                    'Confidence': res['confidence']
                })
            
            results_df = pd.DataFrame(detailed_results)
            
            summary = {
                'file_name': file.name,
                'total_texts': len(results),
                'sentiment_counts': sentiment_counts,
                'dominant_sentiment': dominant_sentiment,
                'avg_confidence': avg_conf,
                'sample_text': results[0][0] if results else "No text available",
                'sentiment_label': dominant_sentiment,
                'confidence': avg_conf
            }
            
            return summary, results_df, None
            
        except Exception as e:
            st.error(f"❌ Unexpected error analyzing file: {str(e)}")
            return None, None, f"Unexpected error: {str(e)}"

# -------------------------------
# UI HELPER FUNCTIONS
# -------------------------------
def create_comparison_dataframe(result1: Dict, result2: Dict, label1: str = "Input 1", 
                               label2: str = "Input 2", mode: str = "text", 
                               text1: str = None, text2: str = None) -> pd.DataFrame:
    """Create comparison dataframe with similarity calculation"""
    if not result1 or not result2:
        return None
    
    if mode == 'text':
        data = {
            'Source': [label1, label2],
            'Dominant Sentiment': [result1['sentiment_label'], result2['sentiment_label']],
            'Average Confidence': [result1['confidence'], result2['confidence']]
        }
    else:  # mode == 'file'
        data = {
            'Source': [label1, label2],
            'Dominant Sentiment': [result1['dominant_sentiment'], result2['dominant_sentiment']],
            'Average Confidence': [result1['avg_confidence'], result2['avg_confidence']],
            'Total Texts': [result1['total_texts'], result2['total_texts']]
        }
    
    if text1 and text2:
        similarity = calculate_text_similarity(text1, text2)
        data['Text Similarity'] = [f'{similarity:.2f}', f'{similarity:.2f}']
    
    return pd.DataFrame(data)

def create_comparison_chart(comparison_df: pd.DataFrame):
    """Create comparison chart"""
    fig = go.Figure()
    colors = [Config.SENTIMENT_COLORS.get(s, '#999999') for s in comparison_df['Dominant Sentiment']]
    
    fig.add_trace(go.Bar(
        x=comparison_df['Source'],
        y=comparison_df['Average Confidence'],
        text=comparison_df['Dominant Sentiment'] + '<br>(' + (comparison_df['Average Confidence'] * 100).round(1).astype(str) + '%)',
        textposition='auto',
        marker_color=colors
    ))
    
    fig.update_layout(
        title='Sentiment Comparison', 
        yaxis_title='Average Confidence', 
        yaxis=dict(range=[0, 1]), 
        showlegend=False
    )
    return fig

def create_sentiment_distribution_chart(result1: Dict, result2: Dict, label1: str, label2: str):
    """Create sentiment distribution chart"""
    fig = go.Figure()
    sentiments = ['Negative', 'Neutral', 'Positive']
    colors = [Config.SENTIMENT_COLORS[s] for s in sentiments]
    
    for i, s in enumerate(sentiments):
        fig.add_trace(go.Bar(
            name=s,
            x=[label1, label2],
            y=[result1['sentiment_counts'].get(s, 0), result2['sentiment_counts'].get(s, 0)],
            marker_color=colors[i],
            text=[result1['sentiment_counts'].get(s, 0), result2['sentiment_counts'].get(s, 0)],
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Sentiment Distribution Comparison', 
        yaxis_title='Number of Texts', 
        barmode='group', 
        showlegend=True
    )
    return fig

def highlight_sentiment_words(text: str) -> Tuple[str, List[str]]:
    """Highlight sentiment words in text"""
    highlighted_text = ""
    sentiment_words = []
    
    for word in text.split():
        clean_word = word.lower().strip(string.punctuation)
        if clean_word in Config.POSITIVE_WORDS:
            highlighted_text += f'<span style="color:green;font-weight:bold;background-color:#e8f5e8">{word}</span> '
            sentiment_words.append(f"✅ '{word}' → Positive")
        elif clean_word in Config.NEGATIVE_WORDS:
            highlighted_text += f'<span style="color:red;font-weight:bold;background-color:#ffebee">{word}</span> '
            sentiment_words.append(f"❌ '{word}' → Negative")
        else:
            highlighted_text += f'<span style="color:gray">{word}</span> '
    
    return highlighted_text, sentiment_words

def display_single_text_analysis(result: Dict, input_text: str, show_downloads: bool = True, key_suffix: str = ""):
    """Display single text analysis results"""
    sentiment = result['sentiment_label']
    confidence = result['confidence']
    
    # Display sentiment with appropriate styling
    if sentiment == "Positive":
        st.success(f"🎉 Sentiment: {sentiment} | Confidence: {confidence:.1%}")
    elif sentiment == "Negative":
        st.error(f"😞 Sentiment: {sentiment} | Confidence: {confidence:.1%}")
    else:
        st.info(f"😐 Sentiment: {sentiment} | Confidence: {confidence:.1%}")
    
    # Show detailed scores
    if 'scores' in result:
        with st.expander("📊 Model Scores"):
            for score in result['scores']:
                label = Config.LABEL_MAPPING[score['label']]
                st.write(f"- {label}: {score['score']:.1%}")
    
    st.subheader("📝 Analyzed Text")
    st.write(input_text)
    
    # Highlight sentiment words
    highlighted_text, sentiment_words = highlight_sentiment_words(input_text)
    st.markdown(highlighted_text, unsafe_allow_html=True)
    
    with st.expander("🔍 Detailed Explanation"):
        if sentiment_words:
            for word in sentiment_words:
                st.write(word)
        else:
            st.write("No strong sentiment words detected.")

def display_single_file_analysis(results_df: pd.DataFrame, uploaded_file_name: str, 
                                show_downloads: bool = True, key_suffix: str = ""):
    """Display single file analysis results"""
    st.dataframe(results_df, width='stretch')

def display_enhanced_download_section(data_type: str, **kwargs):
    """Display enhanced download section at bottom of page"""
    st.markdown("---")
    st.subheader("💾 Export Analysis Results")
    st.markdown("*Download your complete analysis with detailed insights and statistics*")
    
    col1, col2, col3 = st.columns(3)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if data_type == "single_text":
        result = kwargs['result']
        input_text = kwargs['input_text']
        key_suffix = kwargs.get('key_suffix', '')
        
        with col1:
            st.download_button(
                "📊 Comprehensive Spreadsheet Report",
                create_enhanced_csv_data("single_text", result=result, input_text=input_text),
                file_name=f"sentiment_analysis_report_{timestamp}.csv",
                mime="text/csv",
                help="Complete analysis with insights, statistics, and keyword analysis",
                key=f"enhanced_csv_{key_suffix}"
            )
        
        with col2:
            st.download_button(
                "⚙️ Detailed JSON Data Export",
                create_enhanced_json_data("single_text", result=result, input_text=input_text),
                file_name=f"sentiment_analysis_data_{timestamp}.json",
                mime="application/json",
                help="Structured data export with comprehensive analysis metadata",
                key=f"enhanced_json_{key_suffix}"
            )
        
        with col3:
            pdf_data = create_pdf_report("single_text", result=result, input_text=input_text)
            st.download_button(
                "📄 Professional PDF Report",
                pdf_data,
                file_name=f"sentiment_analysis_report_{timestamp}.pdf",
                mime="application/pdf",
                help="Publication-ready report with charts, statistics, and analysis",
                key=f"enhanced_pdf_{key_suffix}"
            )
    
    elif data_type == "single_file":
        results_df = kwargs['results_df']
        filename = kwargs['filename']
        key_suffix = kwargs.get('key_suffix', '')
        
        with col1:
            st.download_button(
                "📊 Complete File Analysis Report",
                create_enhanced_csv_data("single_file", results_df=results_df, filename=filename),
                file_name=f"{filename.split('.')[0]}_analysis_report_{timestamp}.csv",
                mime="text/csv",
                help="Full analysis with statistics, insights, and individual text results",
                key=f"enhanced_csv_{key_suffix}"
            )
        
        with col2:
            st.download_button(
                "⚙️ Structured Analysis Data",
                create_enhanced_json_data("single_file", results_df=results_df, filename=filename),
                file_name=f"{filename.split('.')[0]}_analysis_data_{timestamp}.json",
                mime="application/json",
                help="Machine-readable format with comprehensive statistics and metadata",
                key=f"enhanced_json_{key_suffix}"
            )
        
        with col3:
            pdf_data = create_pdf_report("single_file", results_df=results_df, filename=filename)
            st.download_button(
                "📄 Executive Summary Report",
                pdf_data,
                file_name=f"{filename.split('.')[0]}_analysis_report_{timestamp}.pdf",
                mime="application/pdf",
                help="Visual report with charts, statistics, and key insights",
                key=f"enhanced_pdf_{key_suffix}"
            )
    
    elif data_type == "comparison":
        comparison_df = kwargs['comparison_df']
        comparison_type = kwargs.get('comparison_type', 'text')
        key_suffix = kwargs.get('key_suffix', '')
        
        with col1:
            st.download_button(
                "📊 Comparative Analysis Report",
                create_enhanced_csv_data("comparison", comparison_df=comparison_df, comparison_type=comparison_type),
                file_name=f"comparative_analysis_{comparison_type}_{timestamp}.csv",
                mime="text/csv",
                help="Side-by-side comparison with detailed analysis",
                key=f"enhanced_csv_{key_suffix}"
            )
        
        with col2:
            st.download_button(
                "⚙️ Comparison Data Export",
                create_enhanced_json_data("comparison", comparison_df=comparison_df, comparison_type=comparison_type),
                file_name=f"comparative_analysis_{comparison_type}_{timestamp}.json",
                mime="application/json",
                help="Structured comparison data for further analysis",
                key=f"enhanced_json_{key_suffix}"
            )
        
        with col3:
            # For now, use a simplified PDF for comparisons
            st.download_button(
                "📄 Comparison Summary",
                comparison_df.to_csv(index=False),
                file_name=f"comparative_analysis_{comparison_type}_{timestamp}.csv",
                mime="text/csv",
                help="Quick comparison summary",
                key=f"simple_comparison_{key_suffix}"
            )

# -------------------------------
# MAIN STREAMLIT APPLICATION
# -------------------------------
def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title=Config.PAGE_TITLE, 
        page_icon=Config.PAGE_ICON, 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Main UI
    st.title("📊 Sentiment Analysis Dashboard")
    st.markdown("Analyze text sentiment with AI-powered insights. Supports Positive, Negative, and Neutral classifications.")
    
    # Sidebar for input selection
    with st.sidebar:
        st.header("⚙️ Input Options")
        input_method = st.radio(
            "Select input method:", 
            ("Type Text", "Upload File (CSV/TXT)", "Comparative Analysis"),
            help="Choose how you want to provide text for analysis"
        )
        
        # Input handling based on method
        input_text = None
        uploaded_file = None
        compare_text_1 = None
        compare_text_2 = None
        compare_file_1 = None
        compare_file_2 = None
        
        if input_method == "Type Text":
            input_text = st.text_area(
                "Enter text:", 
                height=150,
                max_chars=Config.MAX_TEXT_LENGTH,
                help=f"Maximum {Config.MAX_TEXT_LENGTH:,} characters"
            )
            
        elif input_method == "Upload File (CSV/TXT)":
            uploaded_file = st.file_uploader(
                "Upload a file", 
                type=Config.SUPPORTED_FILE_TYPES,
                help="CSV files must have a 'text' column"
            )
            
        elif input_method == "Comparative Analysis":
            st.subheader("🔄 Compare Two Inputs")
            compare_option = st.radio("Compare using:", ("Two Texts", "Two Files"))
            
            if compare_option == "Two Texts":
                col1, col2 = st.columns(2)
                with col1:
                    compare_text_1 = st.text_area("Text 1:", height=100, key="text1")
                with col2:
                    compare_text_2 = st.text_area("Text 2:", height=100, key="text2")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    compare_file_1 = st.file_uploader("File 1", type=Config.SUPPORTED_FILE_TYPES, key="file1")
                with col2:
                    compare_file_2 = st.file_uploader("File 2", type=Config.SUPPORTED_FILE_TYPES, key="file2")
        
        # Analyze button
        analyze_button = st.button("🚀 Analyze Sentiment", type="primary")
    
    # Clear session state when input method changes
    if input_method != st.session_state["current_input_method"]:
        st.session_state["current_input_method"] = input_method
        keys_to_clear = [
            "results_df", "file_uploaded", "uploaded_file_name", "comparison_results", 
            "text_results", "comparison_df", "comparison_file_results_dfs", 
            "comparison_texts", "comparison_files", "comparison_file_names", "text_input_analyzed"
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["📝 Analysis Results", "📈 Visualization"])
    
    with tab1:
        if input_method == "Type Text":
            if analyze_button and input_text and validate_text_input(input_text):
                with st.spinner("🔄 Analyzing text..."):
                    result = analyzer.analyze_text(input_text)
                    if result:
                        st.session_state["text_results"] = result
                        st.session_state["text_input_analyzed"] = input_text
                        display_single_text_analysis(result, input_text, show_downloads=False, key_suffix="single_text")
                        # Download section at bottom
                        display_enhanced_download_section("single_text", result=result, input_text=input_text, key_suffix="single_text")
                    else:
                        st.error("❌ Failed to analyze text. Please try again.")
            elif st.session_state.get("text_results") and st.session_state.get("text_input_analyzed") == input_text:
                display_single_text_analysis(
                    st.session_state["text_results"], 
                    st.session_state["text_input_analyzed"], 
                    show_downloads=False,
                    key_suffix="single_text"
                )
                # Download section at bottom
                display_enhanced_download_section("single_text", 
                    result=st.session_state["text_results"], 
                    input_text=st.session_state["text_input_analyzed"], 
                    key_suffix="single_text")
        
        elif input_method == "Upload File (CSV/TXT)":
            if analyze_button and uploaded_file:
                with st.spinner(f"🔄 Analyzing {uploaded_file.name}..."):
                    res, res_df, err = analyzer.analyze_file_with_progress(uploaded_file)
                    if err:
                        st.error(f"❌ {err}")
                    else:
                        st.session_state["results_df"] = res_df
                        st.session_state["uploaded_file_name"] = uploaded_file.name
                        st.subheader("📋 Detailed Results")
                        st.success(f"✅ Successfully analyzed {len(res_df)} texts from {uploaded_file.name}")
                        display_single_file_analysis(res_df, uploaded_file.name, show_downloads=False, key_suffix="single_file")
                        # Download section at bottom
                        display_enhanced_download_section("single_file", results_df=res_df, filename=uploaded_file.name, key_suffix="single_file")
            elif (st.session_state.get("results_df") is not None and uploaded_file and 
                  st.session_state.get("uploaded_file_name") == uploaded_file.name):
                st.subheader("📋 Detailed Results")
                display_single_file_analysis(
                    st.session_state["results_df"], 
                    st.session_state["uploaded_file_name"], 
                    show_downloads=False,
                    key_suffix="single_file"
                )
                # Download section at bottom
                display_enhanced_download_section("single_file", 
                    results_df=st.session_state["results_df"], 
                    filename=st.session_state["uploaded_file_name"], 
                    key_suffix="single_file")
        
        elif input_method == "Comparative Analysis":
            if analyze_button:
                if compare_option == "Two Texts" and compare_text_1 and compare_text_2:
                    if validate_text_input(compare_text_1) and validate_text_input(compare_text_2):
                        with st.spinner("🔄 Comparing texts..."):
                            res1 = analyzer.analyze_text(compare_text_1)
                            res2 = analyzer.analyze_text(compare_text_2)
                            
                            if res1 and res2:
                                comparison_df = create_comparison_dataframe(
                                    res1, res2, "Text 1", "Text 2", "text", compare_text_1, compare_text_2
                                )
                                st.session_state["comparison_df"] = comparison_df
                                st.session_state["comparison_results"] = (res1, res2)
                                st.session_state["comparison_texts"] = (compare_text_1, compare_text_2)
                                
                                st.subheader("📊 Comparison Results")
                                st.dataframe(comparison_df, width='stretch')
                                
                                # Display detailed analysis
                                st.subheader("🔍 Detailed Comparison")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Text 1 Analysis**")
                                    display_single_text_analysis(res1, compare_text_1, show_downloads=False, key_suffix="text1")
                                with col2:
                                    st.markdown("**Text 2 Analysis**")
                                    display_single_text_analysis(res2, compare_text_2, show_downloads=False, key_suffix="text2")
                                
                                if 'Text Similarity' in comparison_df.columns:
                                    st.markdown(f"**📏 Text Similarity:** {comparison_df['Text Similarity'].iloc[0]}")
                                
                                # Download section at bottom
                                display_enhanced_download_section("comparison", 
                                    comparison_df=comparison_df, 
                                    comparison_type="text", 
                                    key_suffix="text_comparison")
                            else:
                                st.error("❌ Failed to analyze one or both texts. Please try again.")
                
                elif compare_option == "Two Files" and compare_file_1 and compare_file_2:
                    with st.spinner("🔄 Comparing files..."):
                        res1, res_df1, err1 = analyzer.analyze_file_with_progress(compare_file_1)
                        res2, res_df2, err2 = analyzer.analyze_file_with_progress(compare_file_2)
                        
                        if err1:
                            st.error(f"❌ Error in File 1: {err1}")
                        if err2:
                            st.error(f"❌ Error in File 2: {err2}")
                        
                        if res1 and res2:
                            comparison_df = create_comparison_dataframe(
                                res1, res2, compare_file_1.name, compare_file_2.name, 
                                "file", res1['sample_text'], res2['sample_text']
                            )
                            st.session_state["comparison_df"] = comparison_df
                            st.session_state["comparison_results"] = (res1, res2)
                            st.session_state["comparison_files"] = (res_df1, res_df2)
                            st.session_state["comparison_file_names"] = (compare_file_1.name, compare_file_2.name)
                            
                            st.subheader("📊 Comparison Results")
                            st.dataframe(comparison_df, width='stretch')
                            
                            # Display detailed file results
                            st.subheader("🔍 Detailed File Analysis")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**{compare_file_1.name} Analysis**")
                                if res_df1 is not None:
                                    display_single_file_analysis(res_df1, compare_file_1.name, show_downloads=False, key_suffix="file1")
                            with col2:
                                st.markdown(f"**{compare_file_2.name} Analysis**")
                                if res_df2 is not None:
                                    display_single_file_analysis(res_df2, compare_file_2.name, show_downloads=False, key_suffix="file2")
                            
                            # Download section at bottom
                            display_enhanced_download_section("comparison", 
                                comparison_df=comparison_df, 
                                comparison_type="file", 
                                key_suffix="file_comparison")
            
            # Show stored comparison results
            elif st.session_state.get("comparison_df") is not None:
                comparison_df = st.session_state["comparison_df"]
                st.subheader("📊 Comparison Results")
                st.dataframe(comparison_df, width='stretch')
                
                if "Total Texts" in comparison_df.columns:
                    # File comparison
                    res_df1, res_df2 = st.session_state.get("comparison_files", (None, None))
                    name1, name2 = st.session_state.get("comparison_file_names", ("File 1", "File 2"))
                    
                    st.subheader("🔍 Detailed File Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**{name1} Analysis**")
                        if res_df1 is not None:
                            display_single_file_analysis(res_df1, name1, show_downloads=False, key_suffix="stored_file1")
                    with col2:
                        st.markdown(f"**{name2} Analysis**")
                        if res_df2 is not None:
                            display_single_file_analysis(res_df2, name2, show_downloads=False, key_suffix="stored_file2")
                    
                    display_enhanced_download_section("comparison", 
                        comparison_df=comparison_df, 
                        comparison_type="file", 
                        key_suffix="stored_file_comparison")
                else:
                    # Text comparison
                    res1, res2 = st.session_state.get("comparison_results", (None, None))
                    text1, text2 = st.session_state.get("comparison_texts", (None, None))
                    
                    if res1 and res2 and text1 and text2:
                        st.subheader("🔍 Detailed Comparison")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Text 1 Analysis**")
                            display_single_text_analysis(res1, text1, show_downloads=False, key_suffix="stored_text1")
                        with col2:
                            st.markdown("**Text 2 Analysis**")
                            display_single_text_analysis(res2, text2, show_downloads=False, key_suffix="stored_text2")
                        
                        if 'Text Similarity' in comparison_df.columns:
                            st.markdown(f"**📏 Text Similarity:** {comparison_df['Text Similarity'].iloc[0]}")
                        
                        display_enhanced_download_section("comparison", 
                            comparison_df=comparison_df, 
                            comparison_type="text", 
                            key_suffix="stored_text_comparison")
    
    with tab2:
        st.header("📈 Visualizations")
        
        if input_method == "Type Text":
            # Text Analysis Visualizations (Single or Comparative)
            if st.session_state.get("text_results"):
                # Single Text Analysis Visualization
                result = st.session_state["text_results"]
                
                # Create confidence chart for single text
                fig = go.Figure()
                
                # Single bar showing sentiment and confidence
                color = Config.SENTIMENT_COLORS.get(result['sentiment_label'], '#999999')
                fig.add_trace(go.Bar(
                    x=[result['sentiment_label']],
                    y=[result['confidence']],
                    text=[f"{result['sentiment_label']}<br>({result['confidence']:.1%})"],
                    textposition='auto',
                    marker_color=[color],
                    name="Sentiment Confidence"
                ))
                
                fig.update_layout(
                    title='Text Sentiment Analysis',
                    yaxis_title='Confidence Score',
                    yaxis=dict(range=[0, 1]),
                    showlegend=False,
                    xaxis_title='Sentiment'
                )
                st.plotly_chart(fig, width='stretch')
                
                # Show detailed scores breakdown
                if 'scores' in result:
                    fig2 = go.Figure()
                    sentiments = []
                    scores = []
                    colors = []
                    
                    for score in result['scores']:
                        sentiment_name = Config.LABEL_MAPPING[score['label']]
                        sentiments.append(sentiment_name)
                        scores.append(score['score'])
                        colors.append(Config.SENTIMENT_COLORS.get(sentiment_name, '#999999'))
                    
                    fig2.add_trace(go.Bar(
                        x=sentiments,
                        y=scores,
                        text=[f"{s:.1%}" for s in scores],
                        textposition='auto',
                        marker_color=colors
                    ))
                    
                    fig2.update_layout(
                        title='Detailed Model Scores',
                        yaxis_title='Confidence Score',
                        yaxis=dict(range=[0, 1]),
                        showlegend=False,
                        xaxis_title='Sentiment Categories'
                    )
                    st.plotly_chart(fig2, width='stretch')
            
            elif st.session_state.get("comparison_df") is not None and st.session_state.get("comparison_results"):
                # Comparative Text Analysis Visualization
                comparison_df = st.session_state["comparison_df"]
                res1, res2 = st.session_state["comparison_results"]
                
                # Use the same bar chart style as single text but with comparison
                fig = create_comparison_chart(comparison_df)
                st.plotly_chart(fig, width='stretch')
                
                # Show detailed scores comparison
                fig2 = go.Figure()
                sentiments = ['Negative', 'Neutral', 'Positive']
                colors = [Config.SENTIMENT_COLORS[s] for s in sentiments]
                
                text1_scores = [0, 0, 0]
                text2_scores = [0, 0, 0]
                
                if 'scores' in res1:
                    for score in res1['scores']:
                        sentiment_name = Config.LABEL_MAPPING[score['label']]
                        idx = sentiments.index(sentiment_name)
                        text1_scores[idx] = score['score']
                
                if 'scores' in res2:
                    for score in res2['scores']:
                        sentiment_name = Config.LABEL_MAPPING[score['label']]
                        idx = sentiments.index(sentiment_name)
                        text2_scores[idx] = score['score']
                
                for i, sentiment in enumerate(sentiments):
                    fig2.add_trace(go.Bar(
                        name=sentiment,
                        x=['Text 1', 'Text 2'],
                        y=[text1_scores[i], text2_scores[i]],
                        marker_color=colors[i],
                        text=[f"{text1_scores[i]:.1%}", f"{text2_scores[i]:.1%}"],
                        textposition='auto'
                    ))
                
                fig2.update_layout(
                    title='Detailed Model Scores Comparison',
                    yaxis_title='Confidence Score',
                    barmode='group',
                    showlegend=True,
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig2, width='stretch')
            
            else:
                st.info("🔍 Analyze some text to see visualizations.")
        
        elif input_method == "Upload File (CSV/TXT)":
            # File Analysis Visualizations (Single or Comparative)
            if st.session_state.get("results_df") is not None:
                # Single File Analysis Visualization
                results_df = st.session_state["results_df"]
                
                # Calculate sentiment distribution
                sentiment_counts = results_df['Sentiment'].value_counts()
                sentiment_percentages = results_df['Sentiment'].value_counts(normalize=True)
                
                # Create sentiment distribution chart
                fig = go.Figure()
                sentiments = ['Negative', 'Neutral', 'Positive']
                colors = [Config.SENTIMENT_COLORS[s] for s in sentiments]
                
                counts = [sentiment_counts.get(s, 0) for s in sentiments]
                percentages = [sentiment_percentages.get(s, 0) for s in sentiments]
                
                fig.add_trace(go.Bar(
                    x=sentiments,
                    y=counts,
                    text=[f"{count}<br>({pct:.1%})" for count, pct in zip(counts, percentages)],
                    textposition='auto',
                    marker_color=colors,
                    name="Sentiment Distribution"
                ))
                
                fig.update_layout(
                    title=f'Sentiment Distribution - {st.session_state.get("uploaded_file_name", "File")}',
                    yaxis_title='Number of Texts',
                    showlegend=False,
                    xaxis_title='Sentiment Categories'
                )
                st.plotly_chart(fig, width='stretch')
                
                # Add confidence distribution
                avg_confidence_by_sentiment = results_df.groupby('Sentiment')['Confidence'].mean()
                
                fig2 = go.Figure()
                avg_confidences = [avg_confidence_by_sentiment.get(s, 0) for s in sentiments]
                
                fig2.add_trace(go.Bar(
                    x=sentiments,
                    y=avg_confidences,
                    text=[f"{conf:.1%}" for conf in avg_confidences],
                    textposition='auto',
                    marker_color=colors,
                    name="Average Confidence"
                ))
                
                fig2.update_layout(
                    title='Average Confidence by Sentiment',
                    yaxis_title='Average Confidence',
                    yaxis=dict(range=[0, 1]),
                    showlegend=False,
                    xaxis_title='Sentiment Categories'
                )
                st.plotly_chart(fig2, width='stretch')
            
            elif st.session_state.get("comparison_df") is not None and st.session_state.get("comparison_results"):
                # Comparative File Analysis Visualization
                res1, res2 = st.session_state["comparison_results"]
                name1, name2 = st.session_state.get("comparison_file_names", ("File 1", "File 2"))
                
                # Use sentiment distribution chart (same style as single file)
                fig = create_sentiment_distribution_chart(res1, res2, name1, name2)
                st.plotly_chart(fig, width='stretch')
                
                # Add confidence comparison chart
                fig2 = go.Figure()
                sentiments = ['Negative', 'Neutral', 'Positive']
                colors = [Config.SENTIMENT_COLORS[s] for s in sentiments]
                
                # Calculate average confidence for each sentiment in both files
                res_df1, res_df2 = st.session_state.get("comparison_files", (None, None))
                
                if res_df1 is not None and res_df2 is not None:
                    avg_conf1 = res_df1.groupby('Sentiment')['Confidence'].mean()
                    avg_conf2 = res_df2.groupby('Sentiment')['Confidence'].mean()
                    
                    for i, sentiment in enumerate(sentiments):
                        conf1 = avg_conf1.get(sentiment, 0)
                        conf2 = avg_conf2.get(sentiment, 0)
                        
                        fig2.add_trace(go.Bar(
                            name=sentiment,
                            x=[name1, name2],
                            y=[conf1, conf2],
                            marker_color=colors[i],
                            text=[f"{conf1:.1%}", f"{conf2:.1%}"],
                            textposition='auto'
                        ))
                    
                    fig2.update_layout(
                        title='Average Confidence Comparison by Sentiment',
                        yaxis_title='Average Confidence',
                        barmode='group',
                        showlegend=True,
                        yaxis=dict(range=[0, 1])
                    )
                    st.plotly_chart(fig2, width='stretch')
            
            else:
                st.info("📊 Upload and analyze a file to see visualizations.")
        
        elif input_method == "Comparative Analysis":
            # Show appropriate visualization based on comparison type
            if st.session_state.get("comparison_df") is not None and st.session_state.get("comparison_results"):
                comparison_df = st.session_state["comparison_df"]
                
                # Check if it's text or file comparison
                if "Total Texts" in comparison_df.columns:
                    # File comparison - use file visualization style
                    res1, res2 = st.session_state["comparison_results"]
                    name1, name2 = st.session_state.get("comparison_file_names", ("File 1", "File 2"))
                    
                    fig = create_sentiment_distribution_chart(res1, res2, name1, name2)
                    st.plotly_chart(fig, width='stretch')
                    
                    # Add confidence comparison
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=[name1, name2],
                        y=[res1['avg_confidence'], res2['avg_confidence']],
                        text=[f"{res1['avg_confidence']:.1%}", f"{res2['avg_confidence']:.1%}"],
                        textposition='auto',
                        marker_color=['#4ECDC4', '#FF6B6B'],
                        name="Average Confidence"
                    ))
                    
                    fig2.update_layout(
                        title='Overall Average Confidence Comparison',
                        yaxis_title='Average Confidence',
                        yaxis=dict(range=[0, 1]),
                        showlegend=False
                    )
                    st.plotly_chart(fig2, width='stretch')
                else:
                    # Text comparison - use text visualization style
                    fig = create_comparison_chart(comparison_df)
                    st.plotly_chart(fig, width='stretch')
                    
                    # Add detailed scores if available
                    res1, res2 = st.session_state["comparison_results"]
                    if 'scores' in res1 and 'scores' in res2:
                        fig2 = go.Figure()
                        sentiments = ['Negative', 'Neutral', 'Positive']
                        colors = [Config.SENTIMENT_COLORS[s] for s in sentiments]
                        
                        text1_scores = [0, 0, 0]
                        text2_scores = [0, 0, 0]
                        
                        for score in res1['scores']:
                            sentiment_name = Config.LABEL_MAPPING[score['label']]
                            idx = sentiments.index(sentiment_name)
                            text1_scores[idx] = score['score']
                        
                        for score in res2['scores']:
                            sentiment_name = Config.LABEL_MAPPING[score['label']]
                            idx = sentiments.index(sentiment_name)
                            text2_scores[idx] = score['score']
                        
                        for i, sentiment in enumerate(sentiments):
                            fig2.add_trace(go.Bar(
                                name=sentiment,
                                x=['Text 1', 'Text 2'],
                                y=[text1_scores[i], text2_scores[i]],
                                marker_color=colors[i],
                                text=[f"{text1_scores[i]:.1%}", f"{text2_scores[i]:.1%}"],
                                textposition='auto'
                            ))
                        
                        fig2.update_layout(
                            title='Detailed Model Scores Comparison',
                            yaxis_title='Confidence Score',
                            barmode='group',
                            showlegend=True,
                            yaxis=dict(range=[0, 1])
                        )
                        st.plotly_chart(fig2, width='stretch')
            else:
                st.info("🔄 Perform a comparative analysis to see visualizations.")

if __name__ == "__main__":

    main()


