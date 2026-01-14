import pandas as pd
import numpy as np
import gc, os
import spacy
from tqdm.auto import tqdm
from sklearn.model_selection import GroupShuffleSplit

def main():
    def mask_doc_fast(doc, mode):
        tokens = []
        target_vec = trait_vectors.get(mode)

        target_norm = None
        if target_vec is not None:
            target_norm = np.linalg.norm(target_vec)

        for token in doc:
            if token.ent_type_:
                ent = token.ent_type_
                if (mode == 'nationality' and ent in ("NORP", "GPE")) or \
                (mode == 'birth' and ent == "DATE") or \
                (mode == 'political' and ent == "NORP") or \
                (mode == 'gender' and ent == "GENDER") or \
                (ent == "MBTI_TYPE"):
                    tokens.append(f"[{ent}]")
                    continue

            if target_vec is not None and token.has_vector:
                sim = np.dot(token.vector, target_vec) / (token.vector_norm * target_norm)
                if sim >= 0.5:
                    tokens.append(trait_labels[mode])
                    continue

            tokens.append(token.text_with_ws)
        return "".join(tokens)


    # Create output folders
    os.makedirs("cleaned_data", exist_ok=True)
    for folder in ["train", "val", "test"]:
        os.makedirs(os.path.join("cleaned_data", folder), exist_ok=True)

    # Define location of the csv files
    file_map = {
        'birth': 'data/birth_year.csv',
        'extrovert': 'data/extrovert_introvert.csv',
        'feeling': 'data/feeling_thinking.csv',
        'gender': 'data/gender.csv',
        'judging': 'data/judging_perceiving.csv',
        'nationality': 'data/nationality.csv',
        'political': 'data/political_leaning.csv',
        'sensing': 'data/sensing_intuitive.csv'
    }

    # Define patterns/labels
    patterns = [
        {"label": "GENDER", "pattern": [{"LOWER": {"IN": ["he", "him", "his", "she", "her", "hers", "they", "them", "boy", "girl", "man", "woman"]}}]},
        {"label": "GENDER", "pattern": [{"LOWER": "he/him"}]},
        {"label": "GENDER", "pattern": [{"LOWER": "she/her"}]},
    ]

    mbti_labels = ["INFJ", "ENTP", "INTJ", "ENFP", "INFP", "ENFJ", "ISTJ", "ISFJ", 
                "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP", "INTP", "ENTJ"]
    patterns += [{"label": "MBTI_TYPE", "pattern": [{"LOWER": m.lower()}]} for m in mbti_labels]


    # Initialize spacy
    nlp = spacy.load("en_core_web_md")
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(patterns)

    trait_labels = {
        'extrovert': '[EXTROVERT_TRAIT]',
        'feeling': '[EMPATHY_TRAIT]',
        'judging': '[JUDGE_TRAIT]',
        'sensing': '[SENSE_TRAIT]'
    }

    trait_vectors = {
        'extrovert': nlp("extrovert").vector,
        'feeling': nlp("empathy").vector,
        'judging': nlp("judgemental").vector,
        'sensing': nlp("sensitive").vector
    }

    # Loop over CSV files
    for mode, file_path in file_map.items():
        print(f"\n--- Processing {mode.upper()} ---")
        df = pd.read_csv(file_path)
        texts = df['post'].fillna("").astype(str).tolist()
        
        processed_texts = []
        with nlp.select_pipes(enable=["entity_ruler", "ner"]):
            pipe = nlp.pipe(texts, n_process=os.cpu_count() - 1, batch_size=128)
            
            for doc in tqdm(pipe, total=len(texts), desc=f"Scrubbing", mininterval=0.5):
                processed_texts.append(mask_doc_fast(doc, mode))

        df['post_masked'] = processed_texts

        # Split the data (no author leakage)
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        tr_idx, tmp_idx = next(gss1.split(df, groups=df["auhtor_ID"]))
        train_df = df.iloc[tr_idx].copy()
        tmp_df = df.iloc[tmp_idx].copy()

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
        va_idx, te_idx = next(gss2.split(tmp_df, groups=tmp_df["auhtor_ID"]))
        val_df = tmp_df.iloc[va_idx].copy()
        test_df = tmp_df.iloc[te_idx].copy()

        # Remove unnecessary columns and save dfs
        train_df = train_df.drop(columns=["auhtor_ID"])
        test_df = test_df.drop(columns=["auhtor_ID"])
        val_df = val_df.drop(columns=["auhtor_ID"])

        train_df.to_csv(f'cleaned_data/train/{mode}_train.csv', index=False)
        val_df.to_csv(f'cleaned_data/val/{mode}_val.csv', index=False)
        test_df.to_csv(f'cleaned_data/test/{mode}_test.csv', index=False)

        del df, train_df, val_df, test_df, texts, processed_texts
        gc.collect()

if __name__ == "__main__":
    main()