WITH product_selection AS (
  SELECT *
  FROM styles_details
  WHERE id IN (SELECT id FROM read_parquet('__ECOMM_DATA__/q5_sample.parquet'))
), classified_products AS MATERIALIZED (
  SELECT
    sd.id,
    CAST(
      llm_complete(
        {
          'model_name': '<<model_name>>',
          'model_parameters': '{"max_tokens":128,"temperature":0.0,"seed":0,"response_format":{"type":"json_schema","json_schema":{"name":"product_category","schema":{"type":"object","properties":{"category":{"type":"string","enum":["Dress","Bottomwear","Socks","Topwear","Innerwear"]}},"required":["category"],"additionalProperties":false}},"strict":true}}'
        },
        {
          'prompt': 'Classify this product into exactly one category: Dress, Bottomwear, Socks, Topwear, Innerwear. Dress is a one-piece outer garment worn on the torso and hanging over the legs. Bottomwear includes trousers, jeans, skirts, shorts, and leggings. Socks are soft clothing worn on the feet. Topwear includes shirts, blouses, t-shirts, and jackets. Innerwear is clothing worn beneath outer garments. Set the category field to the selected category. Title: {{title}} Description: {{description}}',
          'context_columns': [
            {'data': sd.productDisplayName, 'name': 'title'},
            {'data': COALESCE(sd.productDescriptors.description.value, ''), 'name': 'description'}
          ]
        }
      ) AS VARCHAR
    ) AS raw_category
  FROM product_selection AS sd
)
SELECT
  id,
  json_extract_string(CAST(raw_category AS JSON), '$.category') AS category
FROM classified_products;
