

# Topic Modeling Methodology

## BERTopic Workflow

Topic modeling was conducted using BERTopic, a model that integrates transformer-based embeddings with density-based clustering and class-based TF–IDF representations. The model was fitted using the cleaned text segments together with the precomputed GTE-large embeddings generated in the previous stage.

Within BERTopic, dimensionality reduction was performed using Uniform Manifold Approximation and Projection (UMAP), followed by clustering with the density-based algorithm HDBSCAN. The parameters were set to maintain consistency with the clustering configuration established earlier in the analysis. UMAP was applied with *n_components = 5*, *n_neighbors = 15*, *min_dist = 0.05*, and *metric = cosine*, while HDBSCAN clustering used *min_cluster_size = 30*, *min_samples = None*, *metric = euclidean*, and *cluster_selection_method = "eom"*. This configuration ensures that the topic modeling stage operates on the same semantic structure identified during the earlier clustering analysis.

Topic representations were generated using a unigram CountVectorizer (*ngram_range = (1, 1)*, *min_df = 10*, *max_df = 0.6*). In order to improve interpretability, an Early Modern English stopword list was applied prior to topic modeling. This list extends the standard English stopword set with archaic function words and orthographic variants common in Early Modern texts (such as *thou*, *thee*, *thy*, *hath*, *doth*, *haue*, and *hee*), as well as corpus-specific high-frequency dialogic expressions identified during exploratory inspection. Words that primarily reflect orthographic variation rather than semantic content were also excluded.

After the initial model fitting, topic representations were further refined using two complementary representation models: **KeyBERTInspired** and **Maximal Marginal Relevance (MMR)**. This stacked representation strategy improves topic coherence by selecting keywords that are both semantically representative of the topic and minimally redundant with each other.

For interpretive clarity, multiple keyword sets were exported for each topic (top 6, 8, 10, and 20 terms). The ten-keyword representation was used as the primary format for exploratory interpretation, as it provides sufficient semantic context while avoiding the introduction of low-salience terms that often appear in longer keyword lists.

The resulting topic structure captures a range of thematic domains characteristic of Early Modern drama, including domestic and social interaction, political discourse, religious language, classical mythology, and representations of imperial power. Each topic was associated with a set of representative text segments and keyword lists, which were used in the subsequent stage of qualitative interpretation and topic labeling.