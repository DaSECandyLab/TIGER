WITH classified AS MATERIALIZED (
SELECT movie.title AS title, TRY_CAST(json_extract_string(TRY_CAST(CAST(llm_complete(
  {'model_name': '<<model_name>>', 'model_parameters': '{"max_tokens":64,"temperature":0.0,"seed":0,"response_format":{"type":"json_schema","strict":true,"json_schema":{"name":"mmqa_matches","schema":{"type":"object","properties":{"matches":{"type":"boolean"}},"required":["matches"],"additionalProperties":false}}}}'},
  {
    'prompt': 'Is this movie a comedy according to its description? Return an object with the single boolean field matches; use true when the condition holds and false otherwise. Do not include explanation. Movie: {{title}} Description: {{description}}',
    'context_columns': [
      {'data': movie.title, 'name': 'title'},
      {'data': movie.text, 'name': 'description'}
    ]
  }
) AS VARCHAR) AS JSON), '$.matches') AS BOOLEAN) AS prediction
FROM lizzy_caplan_text_data AS movie

),
quality_decisions AS MATERIALIZED (
SELECT title AS sample_id, prediction FROM classified
),
query_result AS (
SELECT title FROM classified WHERE prediction
)
SELECT query_result.*, NULL::VARCHAR AS __quality_samples FROM query_result
UNION ALL
SELECT NULL,
  CAST(to_json(COALESCE(list(struct_pack(id := CAST(sample_id AS VARCHAR), prediction := CAST(prediction AS VARCHAR))), [])) AS VARCHAR) AS __quality_samples
FROM quality_decisions;
