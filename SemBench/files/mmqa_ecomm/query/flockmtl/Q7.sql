WITH classified AS MATERIALIZED (
SELECT sd.id AS id, CAST(
  llm_filter(
    {'model_name': '<<model_name>>', 'model_parameters': '{"temperature":0,"max_tokens":64,"seed":0,"response_format":{"type":"json_schema","strict":true,"json_schema":{"schema":{"type":"string","enum":["true","false"]}}}}'},
    {
      'prompt': 'Is this product a backpack from Reebok? Answer true or false. Title: {{title}} Description: {{description}}',
      'context_columns': [
        {'data': sd.productDisplayName, 'name': 'title'},
        {'data': COALESCE(sd.productDescriptors.description.value, ''), 'name': 'description'}
      ]
    }
  ) AS BOOLEAN
) AS prediction
FROM styles_details AS sd JOIN read_parquet('__ECOMM_DATA__/q1_sample.parquet') chosen ON sd.id=chosen.id

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
