# Master Table Construction

This step integrates the outputs generated during the previous stages of the computational pipeline into a unified chunk-level analytical dataset.

The master table combines:

- chunk-level text segments
- topic assignments
- topic summaries
- clustering coordinates
- metadata fields

The integration process aligns all tables using `chunk_id` as the primary key and standardizes column naming across intermediate outputs.

Rows assigned to topic `-1` may optionally be excluded from the final analytical dataset. The resulting table serves as the primary dataset for subsequent visualization and comparative genre analysis.