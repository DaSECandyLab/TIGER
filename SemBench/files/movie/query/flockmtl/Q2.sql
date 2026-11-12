SELECT reviewId
FROM reviews_2000 AS r
WHERE r.id = 'taken_3' AND CAST(llm_filter(
      {'model_name': '<<model_name>>'},
      {
        'prompt': 'Task: decide whether the following movie review is positive. Answer true or false. # # Review text: # # {{review}}',
        'cacheblend': true,
        'blend_special_str': ' # # ',
        'context_columns': [{'data': r.reviewText, 'name': 'review'}]
      }
  ) AS BOOLEAN);
