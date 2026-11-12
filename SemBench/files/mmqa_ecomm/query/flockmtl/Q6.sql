WITH classified_routes AS MATERIALIZED (
  SELECT
    airport.Airlines,
    CAST(llm_complete(
      {
        'model_name': '<<model_name>>',
        'model_parameters': '{"max_tokens":64,"temperature":0.0,"response_format":{"type":"json_schema","json_schema":{"name":"frankfurt_destination_match","schema":{"type":"object","properties":{"matches":{"type":"boolean"}},"required":["matches"],"additionalProperties":false}},"strict":true},"seed":0}'
      },
      {
        'prompt': 'Inspect only the comma-separated Destinations field. Set matches=true if and only if one complete destination token, after trimming spaces, is exactly Frankfurt. Otherwise set matches=false. Substring similarity does not count: Fort Lauderdale, Frankfurt (Oder), and any other city are false. Germany in general and the airline name are not evidence. Examples: Frankfurt, Boston => true; Fort Lauderdale, Boston => false; Berlin, Boston => false. Airline: {{airline}} Destinations: {{destinations}}',
        'context_columns': [
          {'data': airport.Airlines, 'name': 'airline'},
          {'data': airport.Destinations, 'name': 'destinations'}
        ]
      }
    ) AS VARCHAR) AS raw_verdict
  FROM tampa_international_airport AS airport
),
quality_decisions AS MATERIALIZED (
SELECT Airlines AS sample_id, TRY_CAST(json_extract_string(TRY_CAST(raw_verdict AS JSON), '$.matches') AS BOOLEAN) AS prediction FROM classified_routes
),
query_result AS (
SELECT sample_id AS Airlines FROM quality_decisions WHERE prediction
)
SELECT query_result.*, NULL::VARCHAR AS __quality_samples FROM query_result
UNION ALL
SELECT NULL,
  CAST(to_json(COALESCE(list(struct_pack(id := CAST(sample_id AS VARCHAR), prediction := CAST(prediction AS VARCHAR))), [])) AS VARCHAR) AS __quality_samples
FROM quality_decisions;
