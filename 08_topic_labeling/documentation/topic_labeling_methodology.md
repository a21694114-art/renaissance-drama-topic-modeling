

# Topic Interpretation and Labeling Methodology

## LLM-Assisted Topic Labeling

To facilitate interpretation of the discovered topics, descriptive labels were generated for each topic based on the extracted keyword sets. Because keyword lists alone can be difficult to interpret—particularly in large topic models—an automated labeling procedure was implemented to produce concise semantic descriptions of each topic.

Topic labels were generated using a language model through the OpenAI Responses API. For each topic, the model received the list of extracted keywords and was instructed to infer the underlying semantic domain represented by those terms. The prompt explicitly required the model to produce short descriptive labels rather than summaries, and to rely only on information directly supported by the keywords.

To reduce interpretive bias, the prompt prohibited the introduction of historical period labels, geographic descriptors, or author references unless such information was explicitly present in the keyword list. This constraint ensures that the generated labels reflect the statistical structure of the topic model rather than external scholarly assumptions about the corpus.

The language model was used only as a deterministic labeling tool rather than as a generative interpretive agent. The model temperature was set to a low value (*temperature = 0.1*) in order to minimize variation between runs and produce stable outputs for identical keyword inputs. Because the labels are derived directly from the exported keyword lists, the entire labeling procedure can be reproduced by rerunning the labeling script on the same topic keyword file.

The resulting labels provide concise descriptions of the semantic domains represented by each topic while preserving the reproducibility of the computational pipeline. These labels serve as interpretive guides for the thematic structure identified by the topic model and are used in the subsequent analysis of thematic distributions across the corpus.

## Model Configuration

- Model: `gpt-4.1-mini`
- API: OpenAI Responses API
- Temperature: `0.1`
- Input: exported topic keyword lists
- Output: concise semantic topic labels