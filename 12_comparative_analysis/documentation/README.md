# Shakespeare vs Non-Shakespeare Comparative Analysis

This stage constructs separate genre-topic distribution tables for Shakespearean and non-Shakespearean drama based on the chunk-level master table generated in the previous stages of the pipeline.

## Comparative Corpus Construction

The corpus is divided into two groups using author attribution:

- Shakespeare
- Non-Shakespeare

Texts whose author field contains the string `"Shakespeare"` (case-insensitive) are assigned to the Shakespeare group. All remaining texts, including anonymous plays, are assigned to the Non-Shakespeare group.

## Genre Normalization

To ensure consistency across the comparative analysis, genre labels are normalized using a priority-based mapping system. Because many Early Modern plays contain multiple genre descriptors, the following hierarchy is applied:

1. Tragedy
2. Tragicomedy
3. Comedy
4. History

For example, plays labeled with both “tragedy” and “history” are classified as *Tragedy* according to the priority rule.

Only the three primary dramatic genres used in the dissertation analysis are retained:

- Comedy
- Tragedy
- History

## Topic Distribution Tables

For each corpus group, the script generates:

- genre-topic count tables
- within-genre percentage tables
- pivot tables for visualization
- diagnostic summaries of corpus composition

Topic distributions are normalized within each genre in order to support comparison independent of corpus size. The resulting tables form the basis for the subsequent comparative visualizations and thematic analysis of Shakespearean and non-Shakespearean drama.

## Outputs

The following outputs are generated separately for both Shakespeare and Non-Shakespeare corpora:

- `genre_topic_counts_long.csv`
- `genre_topic_pivot_percent.csv`

Additional diagnostic tables include:

- `debug_group_counts.csv`
- `debug_genres_by_group.csv`