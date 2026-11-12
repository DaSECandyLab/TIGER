SELECT
  sentiment_label AS scoreSentiment,
  COUNT(*) AS count
FROM (
  SELECT
    CASE WHEN CAST(llm_filter(
      {'model_name': '<<model_name>>'},
      {
        'prompt': 'Is the following movie review positive? Answer true or false. Review: {{review}}',
        'context_columns': [{'data': r.reviewText, 'name': 'review'}]
      }
    ) AS BOOLEAN)
      THEN 'POSITIVE'
      ELSE 'NEGATIVE'
    END AS sentiment_label
  FROM reviews_2000 AS r
  WHERE r.id = 'taken_3'
) AS sentiment_results
GROUP BY sentiment_label;
