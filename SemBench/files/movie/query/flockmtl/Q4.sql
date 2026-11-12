SELECT AVG(
    CASE WHEN CAST(llm_filter(
        {'model_name': '<<model_name>>'},
        {
          'prompt': 'Is the following movie review positive? Answer true or false. Review: {{review}}',
          'context_columns': [{'data': r.reviewText, 'name': 'review'}]
        }
    ) AS BOOLEAN)
    THEN 1 ELSE 0
    END
) AS positive_reviews_ratio
FROM reviews_2000 AS r
WHERE id = 'taken_3';
