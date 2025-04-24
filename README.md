# Investigating the Role of Sentiment in Motivational Strategies Using Transformer-Based Models: Associations With Psychiatric Traits​

This project examines whether sentiment expressed in self-reported motivational strategies is linked to psychiatric traits such as anxiety, rumination, pleasure, and self-esteem. Using SiEBERT, a BERT-based sentiment analysis model, the emotional tone of motivational responses was assessed and its relationship with these psychiatric measures.

Presented at the **13th Annual Biomedical Engineering Imaging Institute Symposium (AI/ML Program)** at Icahn School of Medicine at Mount Sinai.

## Methodology

**Participants**:  
312 participants were recruited via Prolific and asked:  
> *"What do you do to motivate yourself?"*

Each participant also completed standardized psychological self-report scales.

**Preprocessing**:  
- Lowercasing, special character removal (regex), stopword removal
- Sentence tokenization (NLTK)
- Final text cleaning applied to each response

**Sentiment Analysis**:
- Used Hugging Face model: `siebert/sentiment-roberta-large-english`
- Labels: `POSITIVE` or `NEGATIVE`
- Generated `sentiment_score` and merged back into dataset

**Statistical Testing**:
- Kruskal-Wallis tests for non-parametric group comparisons
- Post-hoc Dunn’s test with Holm correction for multiple comparisons
- Visualized distributions with seaborn boxplots

**Tools/Libraries**: Python, pandas, seaborn, scipy, statsmodels, scikit-learn, Hugging Face Transformers

## Model Performance

**SiEBERT vs. DistilBERT Comparison**:
- **Accuracy**: SiEBERT = 0.84, DistilBERT = 0.72
- **Cohen's Kappa**: SiEBERT = 0.3867, DistilBERT = 0.3223
- **McNemar’s Test**: p = 0.0013 (statistically significant difference)

**SiEBERT Model Accuracy Measures**:
- **Precision**: 0.87 | **Recall**: 0.95 | **F1 Score**: 0.91

Confusion matrices were used to compare model predictions against human-labeled sentiment.

## Statistical Results / Discussion

### Kruskal-Wallis Test Findings:
| Variable               | p-value | Interpretation                                        |
|------------------------|---------|-------------------------------------------------------|
| Anxiety                | 0.02    | Higher in negative sentiment group                    |
| Rumination             | 0.05    | Higher in negative sentiment group                    |
| Anticipatory Pleasure  | 0.0075  | Higher in positive sentiment group                    |
| Consummatory Pleasure  | 0.0075  | Higher in positive sentiment group                    |
| Self-Esteem            | 0.78    | Not statistically different across sentiment groups   |

### Dunn’s Post-Hoc Test:
- **Anxiety (p = 0.026)** → Negative sentiment linked with higher anxiety
- **Rumination (p = 0.007)** → Higher rumination in negative sentiment
- **Anticipatory Pleasure (p = 0.004)** → Positive sentiment linked to greater anticipation
- **Consummatory Pleasure (p = 0.026)** → Positive sentiment linked to greater enjoyment

Overall these results suggest that individuals with higher sentiment ratings tend to have higher pleasure anticipation and lower anxiety scores.

## Future Directions

- Add **neutral sentiment category** for more nuanced classification
- Further fine-tune SiEBERT on larger text data
- Analyze **self- vs. other-directed strategies** (e.g., “I keep going” vs. “my family helps me”)
- Perform **topic modeling** to extract common motivational themes

