# Dataset explanation
We have 8 files in total. 4 of these files are demographics (age, gender, nationality, political leaning), and 4 are MBTI dimensions. MBTI stands for Myers-Briggs Type Indicator, which is a popular personality framework that assigns people a personality type based on four binary indicators.

### Birth_year
Number of rows: 41873
Number of unique posts: 2236

### extrovert_introvert
Number of rows: 40452
Number of unique posts: 1992

### feeling_thinking
Number of rows: 44635
Number of unique posts: 2401 

### gender
Number of rows: 41873
Number of unique posts: 2236

### judging_perceiving
Number of rows: 41365 
Number of unique posts: 1972 

### nationality
Number of rows: 82616 
Number of unique posts: 2363 

### political_leaning
Number of rows: 57231 
Number of unique posts: 2178 

### sensing_intuitive
Number of rows: 41839 
Number of unique posts: 1954 



# Side notes
There are author_IDs that repeat across datasets. Depending on our scope (do we want to predict demographics or personality traits?), we could only use the authors for which we have a complete profile. 