# -*- coding: utf-8 -*-
"""
data_proT5.py - Preprocessing per modelli T5/GEMMA

Questo script sostituisce la parte di preprocessing classica orientata agli embedding statici
con una versione ottimizzata per modelli transformer come T5 e GEMMA che lavorano direttamente
con testi naturali.

DIFFERENZE dal data_pro.py classico:
- Non genera word_index, padding, o embedding statici
- Mantiene pulizia moderata compatibile con tokenizer transformer
- Preserva split train/val/test per coerenza con pipeline esistente

FLUSSO:
1. Carica dataset JSON 
2. Esegue split train/val/test (80/10/10)
3. Pulisce testi con clean_str() moderata
4. Raggruppa recensioni per user_id e item_id
5. Salva user_texts.json e item_texts.json

OUTPUT:
../dataset/{nome_dataset}_data/
├── user_texts.json   # {user_id: [review1, review2, ...]}
├── item_texts.json   # {item_id: [review1, review2, ...]}
└── train/val/test/   # cartelle vuote per compatibilità


"""

import json
import pandas as pd
import re
import sys
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Configurazione preprocessing
P_REVIEW = 0.85
DOC_LEN = 500  # massimo numero di parole per documento (per eventuale troncamento)

def now():
    """Timestamp formattato per logging"""
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))

def clean_str(string):
    """
    Pulizia moderata del testo per T5/GEMMA
    Mantiene struttura linguistica naturale ma rimuove:
    - Caratteri non alfanumerici problematici
    - Spazi multipli
    - Pattern comuni di rumore

    NOTA: Meno aggressiva del preprocessing classico per preservare
    il contesto semantico che T5 utilizza meglio.
    """
    string = re.sub(r"[^A-Za-z0-9.,!?'\s]", " ", string)  # mantiene punteggiatura base
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)  # spazi multipli -> singolo
    return string.strip()

def get_count(data, id_col):
    """Estrae set univoco di ID da colonna dataset"""
    return set(data[id_col].tolist())

def build_user_item_texts(data):
    """
    Costruisce dizionari di aggregazione testi per utenti e item

    Args:
        data: DataFrame con colonne user_id, item_id, reviewText

    Returns:
        tuple: (user_reviews_dict, item_reviews_dict, user_iid_dict, item_uid_dict)
        - user_reviews_dict: {user_id: [review1, review2, ...]}
        - item_reviews_dict: {item_id: [review1, review2, ...]}
        - user_iid_dict: {user_id: [item_id1, item_id2, ...]}
        - item_uid_dict: {item_id: [user_id1, user_id2, ...]}
    """
    user_reviews_dict = defaultdict(list)
    item_reviews_dict = defaultdict(list)
    user_iid_dict = defaultdict(list)
    item_uid_dict = defaultdict(list)

    print(f"{now()} Costruzione dizionari testi...")

    for i, row in enumerate(data.values):
        uid = row[0]  # user_id numerico
        iid = row[1]  # item_id numerico
        # review_text = str(row[3]) if len(row) > 3 else ""  # reviewText
        review_text = str(data.iloc[i]["reviewText"])

        # Pulizia e validazione testo
        cleaned_review = clean_str(review_text) if review_text.strip() else "<unk>"

        # Filtro lunghezza minima (opzionale)
        if len(cleaned_review.split()) < 3:
            cleaned_review = "<unk>"

        # Aggregazione per user e item
        user_reviews_dict[uid].append(cleaned_review)
        item_reviews_dict[iid].append(cleaned_review)
        user_iid_dict[uid].append(iid)
        item_uid_dict[iid].append(uid)

        # Progress logging
        if (i + 1) % 10000 == 0:
            print(f"{now()} Processate {i + 1} recensioni...")

    return user_reviews_dict, item_reviews_dict, user_iid_dict, item_uid_dict

def save_texts_json(save_folder, user_reviews_dict, item_reviews_dict):
    """
    Salva i dizionari di testi in formato JSON per T5/GEMMA

    Args:
        save_folder: cartella di destinazione
        user_reviews_dict: dizionario user_id -> [reviews]
        item_reviews_dict: dizionario item_id -> [reviews]
    """
    user_path = os.path.join(save_folder, "user_texts.json")
    item_path = os.path.join(save_folder, "item_texts.json")

    # Salvataggio con encoding UTF-8 e indentazione per debugging
    with open(user_path, 'w', encoding='utf-8') as f:
        json.dump(user_reviews_dict, f, ensure_ascii=False, indent=2)

    with open(item_path, 'w', encoding='utf-8') as f:
        json.dump(item_reviews_dict, f, ensure_ascii=False, indent=2)

    print(f"{now()}  Salvati dizionari testi:")
    print(f"    {user_path}")
    print(f"    {item_path}")
    print(f"    {len(user_reviews_dict)} utenti")
    print(f"    {len(item_reviews_dict)} item")

if __name__ == '__main__':

    # Validazione argomenti
    if len(sys.argv) < 2:
        print(" Errore: specificare il file dataset")
        print(" Uso: python data_proT5.py <file_dataset.json>")
        print(" Esempio: python data_proT5.py reviews_Movies_and_TV_5.json")
        sys.exit(1)

    start_time = time.time()
    filename = sys.argv[1]

    print(f" {now()} Avvio preprocessing T5-GEMMA per: {filename}")

    # Setup cartelle output
    save_folder = '../dataset/' + filename.split('.')[0] + "_data"
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'val'), exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'test'), exist_ok=True)

    print(f" {now()} Cartella output: {save_folder}")

    # Caricamento dataset
    try:
        data = pd.read_json(filename, lines=True)
        print(f" {now()} Dataset caricato: {data.shape[0]} recensioni")
    except Exception as e:
        print(f" Errore caricamento dataset: {e}")
        sys.exit(1)

    # Verifica colonne necessarie
    required_cols = ['reviewerID', 'asin', 'reviewText']
    for col in required_cols:
        if col not in data.columns:
            print(f" Colonna mancante: {col}")
            sys.exit(1)

    # Mapping user/item a ID numerici (per compatibilità)
    print(f" {now()} Creazione mapping ID numerici...")
    uid_list = get_count(data, 'reviewerID')
    iid_list = get_count(data, 'asin')
    user2id = {uid: i for i, uid in enumerate(uid_list)}
    item2id = {iid: i for i, iid in enumerate(iid_list)}

    # Applicazione mapping
    data['user_id'] = data['reviewerID'].map(user2id)
    data['item_id'] = data['asin'].map(item2id)

    print(f" {now()} Utenti univoci: {len(uid_list)}")
    print(f" {now()} Item univoci: {len(iid_list)}")

    # Split train/val/test (80/10/10)
    print(f" {now()} Esecuzione split dataset...")
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=1234)
    data_val, data_test = train_test_split(data_test, test_size=0.5, random_state=1234)

    print(f" {now()} Split completato:")
    print(f"    Train: {len(data_train)} ({len(data_train)/len(data)*100:.1f}%)")
    print(f"    Val: {len(data_val)} ({len(data_val)/len(data)*100:.1f}%)")
    print(f"    Test: {len(data_test)} ({len(data_test)/len(data)*100:.1f}%)")

    # Costruzione dizionari testi (solo da training set)
    print(f" {now()} Costruzione dizionari testi dal training set...")
    user_reviews_dict, item_reviews_dict, user_iid_dict, item_uid_dict = build_user_item_texts(data_train)

    # Salvataggio JSON per T5/GEMMA
    print(f" {now()} Salvataggio dizionari in formato JSON...")
    save_texts_json(save_folder, user_reviews_dict, item_reviews_dict)

    # Statistiche finali
    total_user_reviews = sum(len(reviews) for reviews in user_reviews_dict.values())
    total_item_reviews = sum(len(reviews) for reviews in item_reviews_dict.values())

    print(f"\n {now()} Statistiche finali:")
    print(f"    Recensioni per utenti: {total_user_reviews}")
    print(f"    Recensioni per item: {total_item_reviews}")
    print(f"   ️ Tempo totale: {time.time() - start_time:.2f}s")
    print(f"\n {now()} Preprocessing T5-GEMMA completato con successo!")
    print(f" I file sono pronti per data_review.py nel nuovo flusso T5-GEMMA")
