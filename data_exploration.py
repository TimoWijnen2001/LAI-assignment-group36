import pandas as pd

df_birth_year = pd.read_csv("data/birth_year.csv")
df_extrovert_introvert = pd.read_csv("data/extrovert_introvert.csv")
df_feeling_thinking = pd.read_csv("data/feeling_thinking.csv")
df_gender = pd.read_csv("data/gender.csv")
df_judging_perceiving = pd.read_csv("data/judging_perceiving.csv")
df_nationality = pd.read_csv("data/nationality.csv")
df_political_leaning = pd.read_csv("data/political_leaning.csv")
df_sensing_intuitive = pd.read_csv("data/sensing_intuitive.csv")

# Count number of entries and authors per dataset.
def describe_dataset(df):
    print(f"There are {len(df)} entries in this dataset.")
    print(f"Out of these {len(df)}, there are {df['auhtor_ID'].nunique()} unique authors.\n")

print("df_birth_year")
describe_dataset(df_birth_year)
print("df_extrovert_introvert")
describe_dataset(df_extrovert_introvert)
print("df_feeling_thinking")
describe_dataset(df_feeling_thinking)
print("df_gender")
describe_dataset(df_gender)
print("df_judging_perceiving")
describe_dataset(df_judging_perceiving)
print("df_nationality")
describe_dataset(df_nationality)
print("df_political_leaning")
describe_dataset(df_political_leaning)
print("df_sensing_intuitive")
describe_dataset(df_sensing_intuitive)

total_nr_of_entries = (
    len(df_birth_year) +
    len(df_extrovert_introvert) +
    len(df_feeling_thinking) +
    len(df_gender) +
    len(df_judging_perceiving) +
    len(df_nationality) +
    len(df_political_leaning) +
    len(df_sensing_intuitive)
)
print("Total number of entries across all datasets:", total_nr_of_entries)

all_posts = []
all_posts.extend(df_birth_year["post"])
all_posts.extend(df_extrovert_introvert["post"])
all_posts.extend(df_feeling_thinking["post"])
all_posts.extend(df_gender["post"])
all_posts.extend(df_judging_perceiving["post"])
all_posts.extend(df_nationality["post"])
all_posts.extend(df_political_leaning["post"])
all_posts.extend(df_sensing_intuitive["post"])
print(f"Total number of unique posts: {len(set(all_posts))}")

# Count the number of unique authors
all_authors = []
all_authors.extend(df_birth_year["auhtor_ID"])
all_authors.extend(df_extrovert_introvert["auhtor_ID"])
all_authors.extend(df_feeling_thinking["auhtor_ID"])
all_authors.extend(df_gender["auhtor_ID"])
all_authors.extend(df_judging_perceiving["auhtor_ID"])
all_authors.extend(df_nationality["auhtor_ID"])
all_authors.extend(df_political_leaning["auhtor_ID"])
all_authors.extend(df_sensing_intuitive["auhtor_ID"])
print(f"Total number of unique authors: {len(set(all_authors))}")

# show df distributions
print(df_birth_year.describe())
print(df_extrovert_introvert.describe())
print(df_feeling_thinking.describe())
print(df_gender.describe())
print(df_judging_perceiving.describe())
print(df_nationality.describe())
print(df_political_leaning.describe())
print(df_sensing_intuitive.describe())

print(df_political_leaning["political_leaning"].value_counts())

# Proportions of political leaning dataset
center = 25201
right = 17454
left = 14576
sum = center+right+left
center_ratio = round(center/sum, 3)
right_ratio = round(right/sum, 3)
left_ratio = round(left/sum, 3)

print(f"The ratios for left-center-right are: {left_ratio}-{center_ratio}-{right_ratio}")