"""
preprocess.py — Data Preprocessing Pipeline
Loads raw data, cleans it, engineers features, and saves preprocessed artifacts.

Usage:
    python src/preprocess.py --data_path data/tv-shows.csv --output_path outputs/preprocessed_data.pkl
"""

import argparse
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from collections import Counter



# GENRE MAPPING: Normalize 84 genres -> 41


GENRE_MAPPING = {
    # Drama family
    'Dramas': 'Drama', 'Drama': 'Drama', 'TV Dramas': 'Drama',
    # Comedy family
    'Comedies': 'Comedy', 'Comedy': 'Comedy', 'TV Comedies': 'Comedy',
    'Stand-Up Comedy': 'Stand-Up Comedy',
    'Stand-Up Comedy & Talk Shows': 'Stand-Up Comedy',
    # Documentary family
    'Documentaries': 'Documentary', 'Documentary': 'Documentary',
    'Docuseries': 'Documentary',
    'Science & Nature TV': 'Science & Nature TV',
    'Animals & Nature': 'Animals & Nature',
    # Action family
    'Action & Adventure': 'Action & Adventure',
    'Action-Adventure': 'Action & Adventure',
    'TV Action & Adventure': 'Action & Adventure',
    # Thriller family
    'Thrillers': 'Thrillers', 'Thriller': 'Thrillers', 'TV Thrillers': 'Thrillers',
    # Horror family
    'Horror Movies': 'Horror', 'TV Horror': 'Horror',
    # Romance family
    'Romance': 'Romance', 'Romantic Movies': 'Romance',
    'Romantic TV Shows': 'Romance', 'Romantic Comedy': 'Romance',
    # Sci-Fi family (kept separate)
    'Sci-Fi & Fantasy': 'Sci-Fi & Fantasy',
    'TV Sci-Fi & Fantasy': 'Sci-Fi & Fantasy',
    'Science Fiction': 'Science Fiction',
    'Fantasy': 'Fantasy',
    'Superhero': 'Superhero',
    # Kids/Family family
    'Children & Family Movies': 'Family', 'Family': 'Family',
    'Kids': "Kids' TV", "Kids' TV": "Kids' TV",
    'Teen TV Shows': 'Teen TV Shows', 'Coming of Age': 'Coming of Age',
    # Animation family
    'Animation': 'Animation',
    'Anime': 'Anime', 'Anime Features': 'Anime', 'Anime Series': 'Anime',
    # Music family
    'Music & Musicals': 'Music & Musicals', 'Musical': 'Music & Musicals',
    'Music': 'Music & Musicals', 'Concert Film': 'Music & Musicals',
    'Dance': 'Dance',
    # Reality family
    'Reality TV': 'Reality TV', 'Reality': 'Reality TV',
    'Game Show / Competition': 'Reality TV', 'Talk Show': 'Reality TV',
    'Variety': 'Reality TV',
    # Sports family
    'Sports Movies': 'Sports', 'Sports': 'Sports',
    # Crime/Mystery family
    'Crime TV Shows': 'Crime', 'Crime': 'Crime',
    'Police/Cop': 'Crime', 'Spy/Espionage': 'Crime',
    'TV Mysteries': 'Mystery', 'Mystery': 'Mystery',
    # International family (kept separate)
    'International Movies': 'International Movies',
    'International TV Shows': 'International TV Shows',
    'British TV Shows': 'British TV Shows',
    'Korean TV Shows': 'Korean TV Shows',
    'Spanish-Language TV Shows': 'Spanish-Language TV Shows',
    # Classic/Cult
    'Classic Movies': 'Classic', 'Classic & Cult TV': 'Classic',
    'Cult Movies': 'Cult Movies',
    # Singletons
    'Independent Movies': 'Independent Movies',
    'LGBTQ Movies': 'LGBTQ Movies',
    'Faith & Spirituality': 'Faith & Spirituality',
    'Historical': 'Historical', 'Biographical': 'Biographical',
    'Anthology': 'Anthology', 'Western': 'Western', 'Buddy': 'Buddy',
    # Drop (None = remove)
    'Movies': None, 'TV Shows': None, 'Series': None,
    'Travel': None, 'Disaster': None, 'Soap Opera / Melodrama': None,
    'Lifestyle': None, 'Medical': None, 'Parody': None, 'Survival': None,
}



# PREPROCESSING FUNCTIONS

def load_data(data_path):
    """Load the raw CSV file."""
    df = pd.read_csv(data_path)
    print("Loaded:", len(df), "rows")
    return df


def fix_data_quality(df):
    """Fix known data quality issues."""
    # Fix 1: 3 rows have rating/duration swapped
    bad_mask = df['rating'].str.contains('min', na=False)
    df.loc[bad_mask, 'duration'] = df.loc[bad_mask, 'rating']
    df.loc[bad_mask, 'rating'] = np.nan
    print("Fixed", bad_mask.sum(), "rows with rating/duration swap")

    # Fix 2: Trailing commas in country
    df['country'] = df['country'].str.strip().str.rstrip(',')
    print("Fixed trailing commas in country")

    return df


def handle_missing_values(df):
    """Fill or drop missing values."""
    df['country'] = df['country'].fillna('Unknown')
    df['rating'] = df['rating'].fillna('Other')
    df['date_added'] = df['date_added'].fillna('Unknown')
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')

    before = len(df)
    df = df.dropna(subset=['duration']).reset_index(drop=True)
    print("Missing values handled. Dropped", before - len(df), "rows")
    return df


def normalize_genres(df):
    """Normalize genre labels from 84 to 41 using the mapping."""
    def apply_mapping(genre_string):
        genres = genre_string.split(', ')
        normalized = []
        for g in genres:
            g = g.strip()
            mapped = GENRE_MAPPING.get(g)
            if mapped is not None and mapped not in normalized:
                normalized.append(mapped)
        return normalized

    df['normalized_genres'] = df['listed_in'].apply(apply_mapping)

    # Drop titles with 0 genres after normalization
    empty_mask = df['normalized_genres'].apply(len) == 0
    before = len(df)
    df = df[~empty_mask].reset_index(drop=True)
    print("Genre normalization: 84 -> " + str(df['normalized_genres'].explode().nunique()) + " genres")
    print("Dropped", before - len(df), "titles with no valid genres")
    return df


def engineer_features(df):
    """Create new features from existing columns."""
    # Split duration into movie_minutes and tv_seasons
    df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float)
    df['movie_minutes'] = np.where(df['type'] == 'Movie', df['duration_num'], 0)
    df['tv_seasons'] = np.where(df['type'] == 'TV Show', df['duration_num'], 0)

    # Group rare ratings
    rare_ratings = ['NC-17', 'UR', 'NR', 'TV-Y7-FV']
    df['rating_clean'] = df['rating'].apply(
        lambda x: 'Other' if x in rare_ratings else x
    )

    # Parse country into lists
    df['country_list'] = df['country'].apply(
        lambda s: [c.strip() for c in s.split(',') if c.strip() != '']
    )

    print("Features engineered: movie_minutes, tv_seasons, rating_clean, country_list")
    return df


def build_lemmatizer():
    """Setup lemmatizer — try NLTK first, fallback to simple rules."""
    try:
        import nltk
        nltk.download('wordnet', quiet=True)
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize('test')  # Test it works

        def lemmatize_text(text):
            words = text.lower().split()
            return ' '.join([lemmatizer.lemmatize(w) for w in words])

        print("Using NLTK WordNetLemmatizer")
        return lemmatize_text

    except Exception:
        def lemmatize_text(text):
            words = text.lower().split()
            result = []
            for w in words:
                if w.endswith('ies') and len(w) > 4:
                    w = w[:-3] + 'y'
                elif w.endswith('s') and not w.endswith('ss') and len(w) > 3:
                    w = w[:-1]
                result.append(w)
            return ' '.join(result)

        print("Using simple rule-based lemmatizer")
        return lemmatize_text


def encode_features(df, lemmatize_fn):
    """Encode all features into a sparse matrix."""
    # 1. Lemmatize descriptions
    df['desc_lemmatized'] = df['description'].apply(lemmatize_fn)

    # 2. TF-IDF on description
    tfidf_desc = TfidfVectorizer(
        max_features=3000, stop_words='english', ngram_range=(1, 2),
        min_df=3, max_df=0.95, sublinear_tf=True
    )
    X_desc = tfidf_desc.fit_transform(df['desc_lemmatized'])

    # 3. TF-IDF on title
    tfidf_title = TfidfVectorizer(
        max_features=500, stop_words='english', ngram_range=(1, 1),
        min_df=2, max_df=0.95, sublinear_tf=True
    )
    X_title = tfidf_title.fit_transform(df['title'])

    # 4. Multi-hot encode country (top 15 + Other)
    all_countries = df['country_list'].explode()
    top15 = all_countries.value_counts().head(15).index.tolist()

    def bucket_countries(country_list):
        result = []
        has_other = False
        for c in country_list:
            if c in top15:
                result.append(c)
            else:
                has_other = True
        if has_other:
            result.append('Other')
        return result

    df['country_list'] = df['country_list'].apply(bucket_countries)
    mlb_country = MultiLabelBinarizer()
    country_matrix = mlb_country.fit_transform(df['country_list'])
    X_country = csr_matrix(country_matrix.astype(float))
    country_col_names = ['country_' + c for c in mlb_country.classes_]

    # 5. One-hot encode categorical columns
    single_cat_dummies = pd.get_dummies(df[['type', 'rating_clean', 'platform']])
    X_cat = csr_matrix(single_cat_dummies.values.astype(float))

    # 6. Numeric features
    X_numeric = csr_matrix(
        df[['release_year', 'movie_minutes', 'tv_seasons']].values.astype(float)
    )

    # 7. Combine all
    X = hstack([X_desc, X_title, X_country, X_numeric, X_cat])

    # Build feature names list
    feature_names = (
        list(tfidf_desc.get_feature_names_out()) +
        list(tfidf_title.get_feature_names_out()) +
        country_col_names +
        ['release_year', 'movie_minutes', 'tv_seasons'] +
        list(single_cat_dummies.columns)
    )

    print("Feature matrix:", X.shape)

    transformers = {
        'tfidf_desc': tfidf_desc,
        'tfidf_title': tfidf_title,
        'mlb_country': mlb_country,
        'top15_countries': top15,
        'single_cat_columns': list(single_cat_dummies.columns),
    }

    return X, feature_names, transformers


def encode_target(df):
    """Encode normalized genres into a binary matrix."""
    mlb_target = MultiLabelBinarizer()
    y = mlb_target.fit_transform(df['normalized_genres'])
    genre_names = list(mlb_target.classes_)
    print("Target:", y.shape[1], "genres")
    return y, genre_names, mlb_target


def split_data(X, y, test_size=0.2, seed=42):
    """Split into train and test with stratification."""
    dominant_label = y.argmax(axis=1)
    counts = Counter(dominant_label)
    safe_labels = np.array([
        lbl if counts[lbl] >= 2 else -1
        for lbl in dominant_label
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=safe_labels
    )

    print("Train:", X_train.shape[0], "| Test:", X_test.shape[0])
    return X_train, X_test, y_train, y_test


def save_artifacts(output_path, **kwargs):
    """Save all artifacts to a pickle file."""
    with open(output_path, 'wb') as f:
        pickle.dump(kwargs, f)
    print("Saved to:", output_path)


 
 

def main():
    parser = argparse.ArgumentParser(description='Preprocess TV/Movie genre data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to raw CSV')
    parser.add_argument('--output_path', type=str, default='outputs/preprocessed_data.pkl',
                        help='Path to save preprocessed data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    print("=" * 50)
    print("PREPROCESSING PIPELINE")
    print("=" * 50)

    # Step 1: Load
    df = load_data(args.data_path)

    # Step 2: Fix quality
    df = fix_data_quality(df)

    # Step 3: Missing values
    df = handle_missing_values(df)

    # Step 4: Normalize genres
    df = normalize_genres(df)

    # Step 5: Engineer features
    df = engineer_features(df)

    # Step 6: Build lemmatizer
    lemmatize_fn = build_lemmatizer()

    # Step 7: Encode features
    X, feature_names, transformers = encode_features(df, lemmatize_fn)

    # Step 8: Encode target
    y, genre_names, mlb_target = encode_target(df)

    # Step 9: Split
    X_train, X_test, y_train, y_test = split_data(X, y, seed=args.seed)

    # Step 10: Save
    save_artifacts(
        args.output_path,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        genre_names=genre_names, feature_names=feature_names,
        mlb_target=mlb_target, genre_mapping=GENRE_MAPPING,
        **transformers
    )

    print("")
    print("Done!")


if __name__ == '__main__':
    main()
