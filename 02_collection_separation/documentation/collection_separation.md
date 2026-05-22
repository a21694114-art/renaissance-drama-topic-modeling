# Collection Separation

Some EEBO-TCP XML files contain multiple plays within a single collection volume rather than representing a single dramatic work. To construct a play-level corpus suitable for computational analysis, these collection texts were manually separated into individual play-level XML files before extraction and topic modeling.

The metadata table was updated accordingly to reflect the revised XML structure and the newly separated play files.

## Workflow

1. Download EEBO-TCP XML files from the TCP GitHub repositories.
2. Identify collection volumes containing multiple dramatic works.
3. Manually separate individual plays into standalone XML files.
4. Update the metadata table to reflect the revised play-level corpus.
5. Use the separated XML files for subsequent extraction and topic modeling.
