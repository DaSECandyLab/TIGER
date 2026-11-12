SELECT r1.id as id, r1.reviewId as reviewId1, r2.reviewId as reviewId2
FROM reviews_2000 AS r1
JOIN reviews_2000 AS r2
ON r1.id = r2.id AND r1.reviewId <> r2.reviewId
WHERE r1.id = 'ant_man_and_the_wasp_quantumania' AND CAST(llm_filter(
      {'model_name': '<<model_name>>'},
      {
        'prompt': 'Task: decide whether two movie reviews express opposite sentiments, meaning one positive and one negative. Answer true or false. # # Review 1 text: # # {{review1}} # # Review 2 text: # # {{review2}} # #',
        'cacheblend': true,
        'blend_special_str': ' # # ',
        'context_columns': [
          {'data': r1.reviewText, 'name': 'review1'},
          {'data': r2.reviewText, 'name': 'review2'}
        ]
      }
  ) AS BOOLEAN);
