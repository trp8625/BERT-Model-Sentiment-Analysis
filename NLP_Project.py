# This study examines whether sentiment expressed in self-reported motivational strategies is linked to psychological traits such as anxiety, rumination, pleasure, and self-esteem. Using SiEBERT (Hugging Face, 2021), a BERT-based sentiment analysis model, the emotional tone of motivational responses was assessed and its relationship with these psychological measures.

### Preprocessing Steps
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import pandas as pd
import scikit_posthocs as sp
import matplotlib.pyplot as plt

# Download necessary resources for tokenization and stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords (English)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocesses a given text string by performing the following steps:
    1. Convert to lowercase (optional, depends on BERT case sensitivity)
    2. Remove special characters, extra whitespace, and excessive punctuation
    3. Remove stopwords
    4. Tokenize long responses into separate sentences

    Returns:
        str: The cleaned and preprocessed text.
    """
    if not isinstance(text, str) or text.strip() == "":
        return None  # Handle missing or empty values

    # Convert to lowercase (optional, helps with consistency)
    text = text.lower()

    # Remove special characters, HTML tags, and excessive whitespace
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Keep only alphanumeric characters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    # Tokenize into sentences
    sentences = sent_tokenize(text)

    # Remove stopwords (optional, useful for improving clustering)
    cleaned_sentences = []
    for sentence in sentences:
        words = sentence.split()
        filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords
        cleaned_sentence = " ".join(filtered_words)
        cleaned_sentences.append(cleaned_sentence)

    # Return cleaned text as a single string (or a list of sentences if needed)
    return " ".join(cleaned_sentences)

# Load dataset
file_path = '/Users/tanvipatel/Desktop/updated_data_with_bert_sentiment_2.csv'
df = pd.read_csv(file_path)

# Apply preprocessing to the "motivate_txt" column
df["cleaned_motivate_txt"] = df["motivate_txt"].dropna().apply(preprocess_text)

# Save the cleaned dataset (optional)
df.to_csv("/Users/tanvipatel/Desktop/cleaned_data.csv", index=False)

# Display first few rows of cleaned data
print(df[["motivate_txt", "cleaned_motivate_txt"]].head(10))


### Apply SiEBERT Model to the Text Data
import pandas as pd
from transformers import pipeline

# Load the dataset
file_path = '/Users/tanvipatel/Desktop/online_project_data.csv'
data = pd.read_csv(file_path)

# Ensure the text column exists and drop rows with missing text
if 'motivate_txt' in data.columns:
    text_data = data[['id', 'motivate_txt']].dropna()  # Adjust column names if necessary
else:
    raise ValueError("The dataset does not contain a column named 'motivate_txt'.")

# Load the pre-trained Siebert sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

# function to calculate sentiment using the model
def calculate_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

# Apply the function to the text data
text_data[['sentiment', 'sentiment_score']] = text_data['motivate_txt'].apply(
    lambda x: pd.Series(calculate_sentiment(x))
)

# Merge the results back into the original dataset
data_with_sentiment = pd.merge(
    data,
    text_data[['id', 'sentiment', 'sentiment_score']],
    on='id',
    how='left'
)

# Save the updated dataset to a new CSV file
output_file_path = 'updated_data_with_siebert_sentiment.csv'
data_with_sentiment.to_csv(output_file_path, index=False)

print(f"Sentiment analysis completed using Siebert RoBERTa. Results saved to {output_file_path}")


### Calculate Confusion Matrices to Find Most Accurate Sentiment BERT Model for Sentiment Analysis
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
file_path = "/Users/tanvipatel/Desktop/sie_distil_model_sentiment_comparison_results.csv"
df = pd.read_csv(file_path)

# Convert columns to categorical (needed for confusion matrix)
df['human_label'] = df['human_label'].astype('category')
df['siebert_sentiment'] = df['siebert_sentiment'].astype('category')
df['distilbert_sentiment'] = df['distilbert_sentiment'].astype('category')

# Confusion matrix for SieBERT model
conf_matrix_siebert = confusion_matrix(df['human_label'], df['siebert_sentiment'])
print("Confusion Matrix - SieBERT Model:")
print(conf_matrix_siebert)

# Visualize SieBERT's Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix_siebert, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix for SieBERT Model')
plt.xlabel('Predicted Sentiment')
plt.ylabel('Actual Sentiment')
plt.show()

# Extract and print accuracy
accuracy_siebert = accuracy_score(df['human_label'], df['siebert_sentiment'])
print(f"SieBERT Accuracy: {accuracy_siebert:.4f}")

# Confusion matrix for DistilBERT model
conf_matrix_distilbert = confusion_matrix(df['human_label'], df['distilbert_sentiment'])
print("Confusion Matrix - DistilBERT Model:")
print(conf_matrix_distilbert)

# Visualize DistilBERT's Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix_distilbert, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix for DistilBERT Model')
plt.xlabel('Predicted Sentiment')
plt.ylabel('Actual Sentiment')
plt.show()

# Extract and print accuracy
accuracy_distilbert = accuracy_score(df['human_label'], df['distilbert_sentiment'])
print(f"DistilBERT Accuracy: {accuracy_distilbert:.4f}")

# Perform McNemar's Test for significant difference between models
mcnemar_table = np.array([[21, 12], [35, 231]])
mcnemar_test = mcnemar(mcnemar_table)
print("McNemar Test for SieBERT vs. DistilBERT:")
print(mcnemar_test)

# Plot for SieBERT confusion matrix
siebert_confusion_matrix = np.array([[21, 12], [35, 231]])
plt.figure(figsize=(6, 6))
sns.heatmap(siebert_confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('SieBERT Model Accuracy')
plt.xlabel('Actual Sentiment')
plt.ylabel('Predicted Sentiment')
plt.show()

# Plot for DistilBERT confusion matrix
distilbert_confusion_matrix = np.array([[41, 70], [15, 173]])
plt.figure(figsize=(6, 6))
sns.heatmap(distilbert_confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('DistilBERT Model Accuracy')
plt.xlabel('Actual Sentiment')
plt.ylabel('Predicted Sentiment')
plt.show()

# Create a data frame with model comparison metrics
model_comparison = pd.DataFrame({
    'Model': ['SieBERT', 'DistilBERT'],
    'Accuracy': [0.84, 0.72],
    'Kappa': [0.3867, 0.3223],
})

# Plot the accuracy, kappa, and McNemar's test p-value
model_comparison_melt = model_comparison.melt(id_vars='Model', var_name='Metric', value_name='Value')
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Value', hue='Metric', data=model_comparison_melt, palette=['skyblue', 'pink'])
plt.title('Model Performance Comparison')
plt.xlabel('Model')
plt.ylabel('Metric Value')
plt.legend(title='Metric')
plt.show()


### Compute Kruskal-Wallis Test and Post-hoc Dunn Teset
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = "/Users/tanvipatel/Desktop/updated_data_with_siebert_sentiment1.csv"
df = pd.read_csv(file_path)

# Remove any unnamed columns that may have been added during file saving
df = df.loc[:, ~df.columns.str.contains('Unnamed')]

# Recode sentiment column as numeric values
df['sentiment_numeric'] = df['sentiment'].map({'POSITIVE': 1, 'NEGATIVE': -1})

# Define the behavioral variables for correlation
behavioral_scores = [
    'sticsa_total', 'rrs_total', 'selfesteem_score',
    'teps_anticipatory_score', 'teps_consummatory_score'
]
sentiment_column = 'sentiment_numeric'

# Drop rows with missing values in relevant columns
df_cleaned = df.dropna(subset=behavioral_scores + [sentiment_column])

# Perform Kruskal-Wallis test for each behavioral variable against sentiment score
kruskal_results = {}
for var in behavioral_scores:
    groups = [df_cleaned[sentiment_column][df_cleaned[var] == val] for val in df_cleaned[var].unique()]
    kruskal_stat, kruskal_p = stats.kruskal(*groups)
    kruskal_results[var] = {'H-Statistic': kruskal_stat, 'Kruskal-Wallis p-value': kruskal_p}

# Create a summary dataframe for Kruskal-Wallis results
kruskal_df = pd.DataFrame.from_dict(kruskal_results, orient='index')
kruskal_df = kruskal_df.round(2)  # Round to two decimal places

# Function to save a dataframe as a table image with Times New Roman font
def save_table_as_image(df, title, filename):
    plt.rcParams['font.family'] = 'Times New Roman'  # Set font to Times New Roman
    fig, ax = plt.subplots(figsize=(10, 4))  # Adjust size as needed
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)  # Adjust font size as needed
    table.scale(1.2, 1.2)  # Adjust scaling as needed
    plt.title(title, fontsize=14, pad=10, fontfamily='Times New Roman')  # Add title close to the table
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Save Kruskal-Wallis results as an image
save_table_as_image(kruskal_df, "Kruskal-Wallis Test Results", "kruskal_wallis_results.png")

# Create boxplots for sentiment vs behavioral variables
for var in behavioral_scores:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df_cleaned[sentiment_column], y=df_cleaned[var])
    plt.title(f"Distribution of {var} Across Sentiment Ratings", fontfamily='Times New Roman')
    plt.xlabel("Sentiment Rating", fontfamily='Times New Roman')
    plt.ylabel(var.replace("_", " ").title(), fontfamily='Times New Roman')
    # Use a custom palette for -1 (red) and +1 (green)
    sns.boxplot(x=df_cleaned[sentiment_column], y=df_cleaned[var],
                hue=df_cleaned[sentiment_column], palette={-1: 'darkred', 1: 'lightgreen'}, legend=False)
    plt.xticks(rotation=45)
    plt.show()

# Dictionary to store all Dunn test results
dunn_results = {}

# Loop through behavioral variables and run Dunn's test
for var in behavioral_scores:
    df_clean = df[[var, 'sentiment_numeric']].dropna()
    dunn_test = sp.posthoc_dunn(df_clean, val_col=var, group_col='sentiment_numeric', p_adjust='holm')
    dunn_results[var] = dunn_test
    print(f"\nPost-hoc Dunn Test with Holm correction for {var}")
    print(dunn_test)