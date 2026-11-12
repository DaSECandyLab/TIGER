WITH selected_pairs AS MATERIALIZED (SELECT * FROM read_parquet('__ECOMM_DATA__/q7_sample.parquet')),
classified AS MATERIALIZED (
SELECT p1.id || '-' || p2.id AS id, CAST(
    llm_filter(
      {'model_name': '<<model_name>>', 'model_parameters': '{"temperature":0,"max_tokens":64,"seed":0,"response_format":{"type":"json_schema","strict":true,"json_schema":{"schema":{"type":"string","enum":["true","false"]}}}}'},
      {
        'prompt': 'Do these descriptions identify products of the same category and the same brand? Answer true or false. First title: {{title1}} First description: {{description1}} Second title: {{title2}} Second description: {{description2}}',
        'context_columns': [
          {'data': p1.productDisplayName, 'name': 'title1'},
          {'data': COALESCE(p1.productDescriptors.description.value, ''), 'name': 'description1'},
          {'data': p2.productDisplayName, 'name': 'title2'},
          {'data': COALESCE(p2.productDescriptors.description.value, ''), 'name': 'description2'}
        ]
      }
    ) AS BOOLEAN
  ) AS prediction
FROM selected_pairs AS pairs
JOIN styles_details AS p1 ON p1.id = pairs.left_id
JOIN styles_details AS p2 ON p2.id = pairs.right_id
),
quality_decisions AS MATERIALIZED (
SELECT id AS sample_id, prediction FROM classified
),
query_result AS (
SELECT id FROM classified WHERE prediction
)
SELECT query_result.*, NULL::VARCHAR AS __quality_samples FROM query_result
UNION ALL
SELECT NULL,
  CAST(to_json(COALESCE(list(struct_pack(id := CAST(sample_id AS VARCHAR), prediction := CAST(prediction AS VARCHAR))), [])) AS VARCHAR) AS __quality_samples
FROM quality_decisions;
