import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import matplotlib.font_manager as fm
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('vader_lexicon')
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('stopwords')

# Page Configuration
st.set_page_config(
    page_title="Call Transcript Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("📞 Call Transcript Sentiment Analysis")
st.markdown("""
This application analyzes call transcripts to extract sentiment, key topics, and emotional patterns.
Upload your transcript in text format to get started.
""")

# Functions for analysis
def clean_text(text):
    """Clean text by removing special characters and extra whitespace"""
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()  # Convert to lowercase for better analysis

def extract_speaker_content(text):
    """Extract content by speaker (Agent/Customer) if possible"""
    # Simple pattern matching for "Speaker: Text" format
    patterns = [
        r'(Agent|Customer|Representative|Rep|Support|Client|Customer Service|CSR):\s*(.*?)(?=\n\s*(?:Agent|Customer|Representative|Rep|Support|Client|Customer Service|CSR):|$)',
        r'([A-Z][a-z]+):\s*(.*?)(?=\n\s*[A-Z][a-z]+:|$)'  # Fallback for names (e.g., "Sarah: Hello")
    ]
    
    speakers = []
    contents = []
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            for match in matches:
                speakers.append(match[0].strip())
                contents.append(match[1].strip())
            break  # Use first successful pattern
    
    # If no pattern matches, return empty lists
    return speakers, contents

def split_into_sentences(text):
    """Split text into sentences"""
    return nltk.sent_tokenize(text)

def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER"""
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob"""
    analysis = TextBlob(text)
    return {
        'polarity': analysis.sentiment.polarity,
        'subjectivity': analysis.sentiment.subjectivity
    }

def get_emotion_scores(text):
    """Get more granular emotion scores based on lexical analysis"""
    # Simple emotion lexicons
    emotion_lexicons = {
        'joy': ['happy', 'glad', 'delighted', 'pleased', 'satisfied', 'enjoy', 'exciting', 'excited', 'thank', 'thanks', 'appreciate', 'wonderful', 'great', 'perfect', 'excellent'],
        'anger': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated', 'upset', 'outraged', 'hate', 'resent'],
        'sadness': ['sad', 'unhappy', 'disappointed', 'regret', 'sorry', 'unfortunate', 'depressed', 'miserable'],
        'fear': ['afraid', 'scared', 'worried', 'anxious', 'concerned', 'nervous', 'terrified', 'panic'],
        'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'unexpected', 'wow', 'oh'],
        'confusion': ['confused', 'unsure', 'unclear', 'perplexed', 'misunderstood', 'don\'t understand', 'what do you mean']
    }
    
    # Tokenize and clean words
    words = word_tokenize(clean_text(text))
    words = [word for word in words if word.isalpha()]  # Keep only alphabetic words
    
    # Score each emotion
    emotion_scores = {}
    total_words = len(words)
    
    for emotion, lexicon in emotion_lexicons.items():
        count = sum(1 for word in words if word in lexicon)
        # Normalize by text length to get percentage
        emotion_scores[emotion] = count / max(1, total_words) * 100
    
    return emotion_scores

def plot_sentiment_flow(sentences_df):
    """Plot the flow of sentiment throughout the conversation"""
    fig = px.line(
        sentences_df, 
        x=sentences_df.index, 
        y=['compound', 'pos', 'neg', 'neu'],
        title="Sentiment Flow Throughout the Call",
        labels={'index': 'Sentence Number', 'value': 'Sentiment Score'},
        color_discrete_map={
            'compound': 'black',
            'pos': 'green',
            'neg': 'red',
            'neu': 'gray'
        },
        height=400
    )
    
    # Add horizontal reference lines
    fig.add_shape(
        type="line", 
        x0=0, y0=0.05, x1=len(sentences_df), y1=0.05,
        line=dict(color="green", width=1, dash="dash"),
    )
    
    fig.add_shape(
        type="line", 
        x0=0, y0=-0.05, x1=len(sentences_df), y1=-0.05,
        line=dict(color="red", width=1, dash="dash"),
    )
    
    fig.update_layout(
        legend_title_text='Sentiment Type',
        hovermode="x unified",
        xaxis=dict(title='Sentence Number'),
        yaxis=dict(title='Sentiment Score')
    )
    
    return fig

def plot_speaker_sentiment(speakers, sentiments):
    """Plot sentiment by speaker"""
    if not speakers or len(speakers) == 0:
        return None
    
    # Create dataframe with speaker and sentiment data
    data = {
        'Speaker': speakers,
        'Compound': [s['compound'] for s in sentiments],
        'Positive': [s['pos'] for s in sentiments],
        'Negative': [s['neg'] for s in sentiments],
        'Neutral': [s['neu'] for s in sentiments]
    }
    
    df = pd.DataFrame(data)
    
    # Group by speaker and calculate means
    speaker_sentiment = df.groupby('Speaker').mean().reset_index()
    
    # Create grouped bar chart
    fig = px.bar(
        speaker_sentiment,
        x='Speaker',
        y=['Positive', 'Negative', 'Neutral'],
        title="Average Sentiment by Speaker",
        barmode='group',
        color_discrete_map={
            'Positive': 'green',
            'Negative': 'red',
            'Neutral': 'gray'
        },
        height=400
    )
    
    return fig

def generate_wordcloud(text, title="Word Cloud", mask_color=None):
    """Generate a wordcloud from text"""
    # Try to locate a common font
    common_fonts = ['Arial', 'DejaVuSans', 'Tahoma', 'Verdana', 'Helvetica', 'Times New Roman']
    font_path = None
    
    for font in common_fonts:
        font_files = fm.findSystemFonts()
        for file in font_files:
            if font.lower() in os.path.basename(file).lower():
                font_path = file
                break
        if font_path:
            break
    
    # Filter out stopwords
    stopwords_list = set(stopwords.words('english'))
    words = text.split()
    filtered_text = " ".join([word for word in words if word.lower() not in stopwords_list])
    
    try:
        # Create WordCloud
        if font_path:
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100,
                font_path=font_path,
                colormap=mask_color
            ).generate(filtered_text)
        else:
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100,
                colormap=mask_color
            ).generate(filtered_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title)
        return fig
    except ValueError:
        # Fallback to bar chart
        st.warning("WordCloud generation failed. Displaying top words instead.")
        words = filtered_text.lower().split()
        word_count = {}
        for word in words:
            if len(word) > 3:  # Ignore very short words
                word_count[word] = word_count.get(word, 0) + 1
        
        # Sort and get top words
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:20]
        words = [w[0] for w in sorted_words]
        counts = [w[1] for w in sorted_words]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(words, counts)
        ax.set_title(f"Top Words in {title}")
        ax.set_xlabel("Word")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig

def create_sentiment_specific_wordclouds(sentences_df):
    """Create separate wordclouds for positive, negative, and neutral sentences"""
    positive_df = sentences_df[sentences_df['compound'] >= 0.05]
    negative_df = sentences_df[sentences_df['compound'] <= -0.05]
    neutral_df = sentences_df[(sentences_df['compound'] > -0.05) & (sentences_df['compound'] < 0.05)]
    
    positive_text = " ".join(positive_df['sentence'])
    negative_text = " ".join(negative_df['sentence'])
    neutral_text = " ".join(neutral_df['sentence'])
    
    pos_wordcloud = generate_wordcloud(positive_text, "Positive Sentiment Words", "Greens")
    neg_wordcloud = generate_wordcloud(negative_text, "Negative Sentiment Words", "Reds")
    neu_wordcloud = generate_wordcloud(neutral_text, "Neutral Sentiment Words", "Blues")
    
    return pos_wordcloud, neg_wordcloud, neu_wordcloud

def analyze_transcript(text):
    """Complete analysis of transcript"""
    # Clean text
    cleaned_text = clean_text(text)
    
    # Split into sentences
    sentences = split_into_sentences(text)  # Use original text to maintain case
    
    # Try to extract speaker information
    speakers, contents = extract_speaker_content(text)
    has_speakers = len(speakers) > 0
    
    # Analyze each sentence
    sentence_data = []
    for i, sentence in enumerate(sentences):
        vader_sentiment = analyze_sentiment_vader(sentence)
        textblob_sentiment = analyze_sentiment_textblob(sentence)
        
        # Determine sentence speaker if possible
        speaker = "Unknown"
        if has_speakers:
            # Find which speaker's content contains this sentence
            for idx, content in enumerate(contents):
                if sentence in content:
                    speaker = speakers[idx]
                    break
        
        emotion_scores = get_emotion_scores(sentence)
        
        sentence_data.append({
            'sentence': sentence,
            'speaker': speaker,
            'compound': vader_sentiment['compound'],
            'pos': vader_sentiment['pos'],
            'neg': vader_sentiment['neg'],
            'neu': vader_sentiment['neu'],
            'polarity': textblob_sentiment['polarity'],
            'subjectivity': textblob_sentiment['subjectivity'],
            'joy': emotion_scores['joy'],
            'anger': emotion_scores['anger'],
            'sadness': emotion_scores['sadness'],
            'fear': emotion_scores['fear'],
            'surprise': emotion_scores['surprise'],
            'confusion': emotion_scores['confusion']
        })
    
    # Create DataFrame
    sentences_df = pd.DataFrame(sentence_data)
    
    # Overall sentiment
    overall_sentiment = analyze_sentiment_vader(text)
    textblob_overall = analyze_sentiment_textblob(text)
    overall_emotion = get_emotion_scores(text)
    
    # Analyze by speaker if possible
    speaker_sentiments = None
    if has_speakers:
        speaker_sentiments = []
        for speaker, content in zip(speakers, contents):
            sentiment = analyze_sentiment_vader(content)
            speaker_sentiments.append(sentiment)
    
    return {
        'full_text': text,
        'cleaned_text': cleaned_text,
        'sentences_df': sentences_df,
        'overall_sentiment': overall_sentiment,
        'textblob_overall': textblob_overall,
        'overall_emotion': overall_emotion,
        'has_speakers': has_speakers,
        'speakers': speakers,
        'speaker_sentiments': speaker_sentiments
    }

def determine_sentiment_color_scale(value):
    """Return color based on sentiment value"""
    if value >= 0.05:
        # Green with intensity based on value
        intensity = min(255, int(200 + (value * 55)))
        return f"background-color: rgba(0, {intensity}, 0, 0.2)"
    elif value <= -0.05:
        # Red with intensity based on value
        intensity = min(255, int(200 + (abs(value) * 55)))
        return f"background-color: rgba({intensity}, 0, 0, 0.2)"
    else:
        return "background-color: rgba(200, 200, 200, 0.2)"

def display_sentiment_summary(analysis_results):
    """Display overall sentiment summary with enhanced visualizations"""
    # Determine overall sentiment label
    compound = analysis_results['overall_sentiment']['compound']
    if compound >= 0.05:
        sentiment_label = "Positive 😊"
        sentiment_color = "green"
    elif compound <= -0.05:
        sentiment_label = "Negative ☹️"
        sentiment_color = "red"
    else:
        sentiment_label = "Neutral 😐"
        sentiment_color = "gray"
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Sentiment", sentiment_label)
    with col2:
        st.metric("Compound Score", f"{compound:.2f}")
    with col3:
        st.metric("Positivity", f"{analysis_results['overall_sentiment']['pos']:.2f}")
    with col4:
        st.metric("Negativity", f"{analysis_results['overall_sentiment']['neg']:.2f}")
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Sentiment Gauge", "Emotion Analysis"])
    
    with tab1:
        # Sentiment gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = compound,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment Gauge"},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': sentiment_color},
                'steps': [
                    {'range': [-1, -0.05], 'color': 'lightcoral'},
                    {'range': [-0.05, 0.05], 'color': 'lightgray'},
                    {'range': [0.05, 1], 'color': 'lightgreen'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': compound
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Emotion analysis chart
        emotions = analysis_results['overall_emotion']
        emotions_df = pd.DataFrame({
            'Emotion': list(emotions.keys()),
            'Score': list(emotions.values())
        }).sort_values('Score', ascending=False)
        
        fig = px.bar(
            emotions_df,
            x='Emotion',
            y='Score',
            title="Emotional Tone Analysis",
            color='Emotion',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            labels={'Score': 'Intensity (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)

def display_detailed_analysis(analysis_results):
    """Display detailed sentiment analysis with enhanced visualizations"""
    st.header("Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Flow")
        sentiment_flow_fig = plot_sentiment_flow(analysis_results['sentences_df'])
        st.plotly_chart(sentiment_flow_fig, use_container_width=True)
        
        # Create a heatmap of emotions throughout the conversation
        if len(analysis_results['sentences_df']) > 0:
            st.subheader("Emotion Flow")
            emotion_cols = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'confusion']
            emotion_df = analysis_results['sentences_df'][emotion_cols].copy()
            
            # Plot heatmap
            fig = px.imshow(
                emotion_df.T,
                labels=dict(x="Sentence Number", y="Emotion", color="Intensity"),
                title="Emotional Flow Throughout Conversation",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sentiment Distribution")
        # Create sentiment categories
        sentiment_cats = []
        for score in analysis_results['sentences_df']['compound']:
            if score >= 0.05:
                sentiment_cats.append('Positive')
            elif score <= -0.05:
                sentiment_cats.append('Negative')
            else:
                sentiment_cats.append('Neutral')
        
        # Add to dataframe
        temp_df = analysis_results['sentences_df'].copy()
        temp_df['sentiment_category'] = sentiment_cats
        
        # Create a more detailed pie chart with absolute numbers
        counts = temp_df['sentiment_category'].value_counts()
        labels = [f"{k} ({v})" for k, v in zip(counts.index, counts.values)]
        
        fig = px.pie(
            names=counts.index,
            values=counts.values,
            title="Sentiment Distribution",
            color=counts.index,
            color_discrete_map={'Positive':'green', 'Neutral':'gray', 'Negative':'red'},
            labels={'names': 'Sentiment Category', 'values': 'Count'}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Display key statistics
        st.subheader("Key Statistics")
        stats_df = pd.DataFrame({
            'Metric': ['Total Sentences', 'Average Sentence Length', 'Positive Sentences', 'Negative Sentences', 'Neutral Sentences'],
            'Value': [
                len(analysis_results['sentences_df']),
                round(np.mean([len(s) for s in analysis_results['sentences_df']['sentence']]), 1),
                len(analysis_results['sentences_df'][analysis_results['sentences_df']['compound'] >= 0.05]),
                len(analysis_results['sentences_df'][analysis_results['sentences_df']['compound'] <= -0.05]),
                len(analysis_results['sentences_df'][(analysis_results['sentences_df']['compound'] > -0.05) & 
                                                    (analysis_results['sentences_df']['compound'] < 0.05)])
            ]
        })
        st.table(stats_df)
    
    # Display word clouds for positive, negative, and neutral sentiment
    st.subheader("Sentiment-Specific Word Analysis")
    pos_cloud, neg_cloud, neu_cloud = create_sentiment_specific_wordclouds(analysis_results['sentences_df'])
    
    pw_col, nw_col, neuw_col = st.columns(3)
    with pw_col:
        st.pyplot(pos_cloud)
    with nw_col:
        st.pyplot(neg_cloud)
    with neuw_col:
        st.pyplot(neu_cloud)
    
    # Display speaker analysis if available
    if analysis_results['has_speakers']:
        st.subheader("Speaker Analysis")
        speaker_fig = plot_speaker_sentiment(
            analysis_results['speakers'], 
            analysis_results['speaker_sentiments']
        )
        if speaker_fig:
            st.plotly_chart(speaker_fig, use_container_width=True)

def display_sentence_breakdown(analysis_results):
    """Display sentence-by-sentence breakdown with enhanced styling"""
    st.header("Sentence-by-Sentence Breakdown")
    
    # Add sentiment categories for coloring
    sentences_df = analysis_results['sentences_df'].copy()
    
    # Display dataframe with styled rows
    sentences_df['sentence_num'] = range(1, len(sentences_df) + 1)
    
    # Select columns based on whether we have speaker information
    if 'Unknown' not in sentences_df['speaker'].unique() or len(sentences_df['speaker'].unique()) > 1:
        sentences_display = sentences_df[['sentence_num', 'speaker', 'sentence', 'compound', 'pos', 'neg', 'neu']].copy()
        sentences_display.columns = ['#', 'Speaker', 'Sentence', 'Compound', 'Positive', 'Negative', 'Neutral']
    else:
        sentences_display = sentences_df[['sentence_num', 'sentence', 'compound', 'pos', 'neg', 'neu']].copy()
        sentences_display.columns = ['#', 'Sentence', 'Compound', 'Positive', 'Negative', 'Neutral']
    
    # Define a function for styled dataframe
    def highlight_sentiment(row):
        compound = row['Compound']
        
        if compound >= 0.05:
            intensity = min(1.0, 0.2 + (compound * 0.8))
            return [f'background-color: rgba(0, 255, 0, {intensity})'] * len(row)
        elif compound <= -0.05:
            intensity = min(1.0, 0.2 + (abs(compound) * 0.8))
            return [f'background-color: rgba(255, 0, 0, {intensity})'] * len(row)
        else:
            return ['background-color: rgba(200, 200, 200, 0.2)'] * len(row)
    
    # Display with a filter
    sentiment_filter = st.selectbox(
        "Filter sentences by sentiment:",
        ["All Sentences", "Positive Only", "Negative Only", "Neutral Only"]
    )
    
    if sentiment_filter == "Positive Only":
        filtered_df = sentences_display[sentences_display['Compound'] >= 0.05]
    elif sentiment_filter == "Negative Only":
        filtered_df = sentences_display[sentences_display['Compound'] <= -0.05]
    elif sentiment_filter == "Neutral Only":
        filtered_df = sentences_display[(sentences_display['Compound'] > -0.05) & (sentences_display['Compound'] < 0.05)]
    else:
        filtered_df = sentences_display
    
    st.dataframe(filtered_df.style.apply(highlight_sentiment, axis=1), height=400)

def find_extreme_sentences(analysis_results, n=3):
    """Find and display the most extreme sentiment sentences"""
    st.header("Sentiment Highlights")
    
    sentences_df = analysis_results['sentences_df']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Most Positive Sentences")
        most_positive = sentences_df.sort_values('compound', ascending=False).head(n)
        for i, row in most_positive.iterrows():
            sentiment_color = "rgba(0, 255, 0, 0.2)"
            intensity = min(0.5, 0.1 + (row['compound'] * 0.4))
            st.markdown(f"<div style='padding: 10px; border-radius: 5px; background-color: rgba(0, 255, 0, {intensity});'>", unsafe_allow_html=True)
            if row['speaker'] != 'Unknown':
                st.markdown(f"**{row['speaker']}:** *{row['sentence']}*")
            else:
                st.markdown(f"*{row['sentence']}*")
            st.markdown(f"**Sentiment Score: {row['compound']:.2f}** (Pos: {row['pos']:.2f}, Neg: {row['neg']:.2f}, Neu: {row['neu']:.2f})")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("")
    
    with col2:
        st.subheader("Most Negative Sentences")
        most_negative = sentences_df.sort_values('compound', ascending=True).head(n)
        for i, row in most_negative.iterrows():
            intensity = min(0.5, 0.1 + (abs(row['compound']) * 0.4))
            st.markdown(f"<div style='padding: 10px; border-radius: 5px; background-color: rgba(255, 0, 0, {intensity});'>", unsafe_allow_html=True)
            if row['speaker'] != 'Unknown':
                st.markdown(f"**{row['speaker']}:** *{row['sentence']}*")
            else:
                st.markdown(f"*{row['sentence']}*")
            st.markdown(f"**Sentiment Score: {row['compound']:.2f}** (Pos: {row['pos']:.2f}, Neg: {row['neg']:.2f}, Neu: {row['neu']:.2f})")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("")

def find_sentiment_transitions(analysis_results):
    """Find significant sentiment transitions in the conversation"""
    st.header("Significant Sentiment Shifts")
    
    sentences_df = analysis_results['sentences_df'].copy()
    
    if len(sentences_df) < 3:
        st.info("Not enough sentences to analyze sentiment transitions.")
        return
    
    # Calculate sentiment shifts
    sentences_df['prev_compound'] = sentences_df['compound'].shift(1)
    sentences_df['sentiment_shift'] = sentences_df['compound'] - sentences_df['prev_compound']
    
    # Find significant transitions (positive to negative or vice versa)
    significant_shifts = sentences_df[
        (abs(sentences_df['sentiment_shift']) > 0.5) & 
        (~sentences_df['prev_compound'].isna())
    ].copy()
    
    if len(significant_shifts) == 0:
        st.info("No significant sentiment shifts detected in this conversation.")
        return
    
    # Display the transitions
    for i, row in significant_shifts.iterrows():
        prev_idx = i - 1
        if prev_idx >= 0:
            prev_row = sentences_df.iloc[prev_idx]
            
            # Determine direction
            if row['sentiment_shift'] > 0:
                direction = "🔼 Positive Shift"
                color1 = "rgba(255, 0, 0, 0.2)"
                color2 = "rgba(0, 255, 0, 0.2)"
            else:
                direction = "🔽 Negative Shift"
                color1 = "rgba(0, 255, 0, 0.2)"
                color2 = "rgba(255, 0, 0, 0.2)"
                
            st.subheader(f"{direction} (Δ {row['sentiment_shift']:.2f})")
            
            st.markdown(f"<div style='padding: 10px; border-radius: 5px; background-color: {color1};'>", unsafe_allow_html=True)
            if prev_row['speaker'] != 'Unknown':
                st.markdown(f"**{prev_row['speaker']}:** *{prev_row['sentence']}*")
            else:
                st.markdown(f"*{prev_row['sentence']}*")
            st.markdown(f"**Sentiment Score: {prev_row['compound']:.2f}**")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div style='text-align: center'>⬇️</div>", unsafe_allow_html=True)
            
            st.markdown(f"<div style='padding: 10px; border-radius: 5px; background-color: {color2};'>", unsafe_allow_html=True)
            if row['speaker'] != 'Unknown':
                st.markdown(f"**{row['speaker']}:** *{row['sentence']}*")
            else:
                st.markdown(f"*{row['sentence']}*")
            st.markdown(f"**Sentiment Score: {row['compound']:.2f}**")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("---")

# Example transcript for the app
example_text = """
Agent: Good morning! Thank you for calling customer support. How may I assist you today?

Customer: Hi, I've been having problems with my internet connection. It keeps dropping every few minutes.

Agent: I'm sorry to hear about your connection issues. That must be frustrating. Let me help you troubleshoot this.

Customer: Thanks, it's been happening for three days now and it's really annoying. I work from home so I need reliable internet.

Agent: I completely understand your concern, especially since you work from home. Let's get this resolved for you right away.

Customer: Great, I appreciate your help.
"""

# Sidebar
st.sidebar.header("Input Options")
input_method = st.sidebar.radio(
    "Choose input method:",
    ["Upload Text File", "Paste Example Text", "Enter Your Own Text"]
)

# File uploader for text files
uploaded_file = None
transcript_text = ""

if input_method == "Upload Text File":
    uploaded_file = st.sidebar.file_uploader("Upload transcript text file", type=['txt'])
    if uploaded_file is not None:
        transcript_text = uploaded_file.getvalue().decode("utf-8")
        st.sidebar.success("File uploaded successfully!")
elif input_method == "Paste Example Text":
    transcript_text = example_text
    st.sidebar.info("Using example transcript")
elif input_method == "Enter Your Own Text":
    transcript_text = st.sidebar.text_area(
        "Enter call transcript text",
        height=300,
        placeholder="Paste your transcript here..."
    )

# Additional settings
st.sidebar.header("Analysis Settings")
show_sentence_breakdown = st.sidebar.checkbox("Show Sentence Breakdown", value=True)
show_extreme_sentences = st.sidebar.checkbox("Show Sentiment Highlights", value=True)
show_sentiment_shifts = st.sidebar.checkbox("Show Sentiment Shifts", value=True)
num_extreme = st.sidebar.slider("Number of highlights to show", 1, 10, 3)

# About section
st.sidebar.header("About")
st.sidebar.info(
    """
    This app analyzes call transcripts to extract sentiment, 
    trends, and key emotional patterns. It works best with transcripts 
    that identify speakers (e.g., 'Agent:' or 'Customer:').
    """
)

# Main analysis
if transcript_text:
    # Display transcript
    with st.expander("View Transcript", expanded=False):
        st.text_area("Transcript", transcript_text, height=200)
    
    # Analyze transcript
    with st.spinner("Analyzing transcript..."):
        analysis_results = analyze_transcript(transcript_text)
    
    # Display results
    display_sentiment_summary(analysis_results)
    display_detailed_analysis(analysis_results)
    
    # Conditional displays
    if show_sentence_breakdown:
        display_sentence_breakdown(analysis_results)
    
    if show_extreme_sentences:
        find_extreme_sentences(analysis_results, n=num_extreme)
    
    if show_sentiment_shifts:
        find_sentiment_transitions(analysis_results)
    
    # Download results
    st.header("Export Results")
    
    def convert_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    if len(analysis_results['sentences_df']) > 0:
        csv = convert_to_csv(analysis_results['sentences_df'])
        st.download_button(
            "Download Sentence Analysis (CSV)",
            csv,
            "call_transcript_analysis.csv",
            "text/csv",
            key='download-csv'
        )
else:
    st.info("Please upload a transcript file or enter text to begin analysis.")

# Add footer
st.markdown("---")
st.markdown("Call Transcript Sentiment Analysis Tool | Built with Streamlit")