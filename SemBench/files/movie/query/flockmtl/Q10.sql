WITH scored_reviews AS (
  SELECT
    r.id AS movieId,
    CAST(
      llm_complete(
        {'model_name': '<<model_name>>'},
        {
          'prompt': 'Classify how much the reviewer liked the movie. Output one numeric score from 1 to 5; decimal scores are allowed. Never output sentiment words. Mapping: 5 means very positive; 4 means positive; 3 means neutral; 2 means negative; 1 means very negative. Your entire answer for this row must be only the numeric score. # # Review: {{review}} # # The numeric score is',
          'cacheblend': true,
          'blend_special_str': ' # # ',
          'context_columns': [{'data': r.reviewText, 'name': 'review'}]
        }
      ) AS VARCHAR
    ) AS rawReviewScore
  FROM reviews_2000 AS r
),
parsed_scores AS (
  SELECT
    movieId,
    TRY_CAST(
      NULLIF(
        TRIM(regexp_extract(
          rawReviewScore,
          '[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?'
        )),
        ''
      ) AS DOUBLE
    ) AS parsedReviewScore
  FROM scored_reviews
),
numeric_scores AS (
  SELECT
    movieId,
    CASE
      WHEN parsedReviewScore BETWEEN 1.0 AND 5.0 THEN parsedReviewScore
      ELSE NULL
    END AS reviewScore
  FROM parsed_scores
)
SELECT
  movieId,
  AVG(reviewScore) AS movieScore
FROM numeric_scores
GROUP BY movieId;
