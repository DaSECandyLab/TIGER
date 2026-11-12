WITH extracted_brands AS MATERIALIZED (
  SELECT
    sd.id,
    CAST(
    llm_complete(
      {
        'model_name': '<<model_name>>',
        'model_parameters': '{"max_tokens":128,"temperature":0.0,"seed":0,"response_format":{"type":"json_schema","json_schema":{"name":"product_brand","schema":{"type":"object","properties":{"brand":{"type":"string","minLength":1,"maxLength":64,"pattern":"^[A-Za-z0-9][A-Za-z0-9 .&''()+/_-]{0,63}$"}},"required":["brand"],"additionalProperties":false}},"strict":true}}'
      },
      {
        'prompt': 'Extract the brand name from the product title and description. Set the brand field to only the brand name. Title: {{title}} Description: {{description}}',
        'context_columns': [
          {'data': sd.productDisplayName, 'name': 'title'},
          {'data': COALESCE(sd.productDescriptors.description.value, ''), 'name': 'description'}
        ]
      }
    ) AS VARCHAR) AS raw_brand
  FROM styles_details AS sd JOIN read_parquet('__ECOMM_DATA__/q3_sample.parquet') chosen ON sd.id=chosen.id
)
SELECT
  id,
  json_extract_string(CAST(raw_brand AS JSON), '$.brand') AS category
FROM extracted_brands;
