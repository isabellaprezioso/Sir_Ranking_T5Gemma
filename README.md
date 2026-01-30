# Sir_Ranking_T5Gemma – Reproducible pipeline (Baby dataset)

Questo repository contiene una pipeline completa per **review-based recommendation** basata su **DeepCoNN**, con supporto a **embeddings T5Gemma precomputati** (modalità *NO CNN*).

Il README descrive **tutti i passaggi per riprodurre gli esperimenti sul dataset Baby**, dalla preparazione dei dati fino al training e al test.

---

## 1. Installazione ambiente

L’ambiente Python può essere ricreato a partire dal file:

```bash
pip install -r ambiente.txt
```


## 2. Download del modello T5Gemma (cache)

Prima di generare embeddings è necessario scaricare il modello T5Gemma nella cache locale di Hugging Face
(serve un **Hugging Face access token** configurato con `huggingface-cli login`).

Script:

```bash
python cache.py
```

Nel file `cache.py` si seleziona il modello da scaricare, ad esempio:

* `google/t5gemma-2b-2b-prefixlm-it`
* `google/t5gemma-2-1b-1b`
* `google/t5gemma-2-270m-270m`

**Output**:
Il modello e il tokenizer vengono salvati in `~/.cache/huggingface/`.

---

## 3. Preprocessing classico (DeepCoNN – CNN)

Script:

```bash
python pro_data/data_pro.py Baby_5.json
```

### Cosa fa `data_pro.py`

* Legge il dataset Amazon Baby in formato JSON
* Mappa `user_id` e `item_id` in ID numerici (`user2id.json`, `item2id.json`)
* Divide i dati in **train / validation / test**
* Costruisce i documenti testuali per utenti e item
* Genera:

  * rappresentazioni indicizzate (`userDoc2Index.npy`, `itemDoc2Index.npy`, ecc.)
  * embedding Word2Vec (`w2v_*.npy`)

### Cartelle create

```
dataset/Baby_5_data/
├── train/
├── val/
├── test/
├── user2id.json
├── item2id.json
```

Questa fase è necessaria per il **training DeepCoNN classico con CNN**.

---

## 4. Preprocessing per T5/Gemma (testi grezzi)

Script:

```bash
python pro_data/data_pro_t5PRO.py Baby_5.json
```

### Cosa fa `data_pro_t5PRO.py`

* Usa lo stesso split train/val/test
* Pulisce i testi in modo **compatibile con i transformer**
* Aggrega le recensioni per utente e per item
* **Non** genera Word2Vec o padding

### Output principali

```
dataset/Baby_5_data/
├── user_texts.json
├── item_texts.json
```

Questi file contengono i testi che verranno codificati da **T5Gemma**.

---

## 5. Generazione embeddings T5Gemma (offline)

Script:

```bash
python precompute_embeddings_mean.py \
  --data_path ./dataset/Baby_5_data \
  --model_path t5gemma_offline \
  --batch_size 16 \
  --max_len 500 \
  --save_dtype fp16
```

### Cosa fa `precompute_embeddings_mean.py`

* Carica `user_texts.json` e `item_texts.json`
* Codifica i testi con l’encoder T5Gemma
* Applica **mean pooling sulla sequenza**
* Genera un embedding per ogni utente e item

### Collegamento corretto user/item → embedding

L’allineamento è garantito da:

* `user2id.json`
* `item2id.json`

La riga `i` dell’embedding corrisponde **esattamente** all’ID `i` usato nel training.

### Output

```
dataset/Baby_5_data/
├── user_embeddings.npy
├── item_embeddings.npy
├── embeddings_meta.json
```

---

## 6. Training DeepCoNN classico (CNN)

Script:

```bash
python main_t5PRO.py train \
  --dataset Baby_5_data \
  --model DeepCoNN \
  --setup BPR \
  --use_precomputed False \
  --use_t5 False \
  --num_epochs 100 \
  --lr 0.002 \
  --kernel_size 4 \
  --word_dim 300
```

### Cosa succede

* Usa la pipeline DeepCoNN originale
* Testi → Word2Vec → CNN
* Nessun embedding precomputato

---

## 7. Training con embeddings T5Gemma (NO CNN)

Script:

```bash
python main_t5PRO.py train \
  --dataset Baby_5_data \
  --model DeepCoNN \
  --setup BPR \
  --use_precomputed True \
  --precomputed_mode deepconn \
  --use_t5 False \
  --num_epochs 100 \
  --lr 0.002
```

### Cosa succede

* `ReviewData` carica `user_embeddings.npy` e `item_embeddings.npy`
* Le CNN e le embedding layer **non vengono usate**
* Il modello lavora direttamente sugli embeddings T5Gemma

Questa modalità permette di confrontare **CNN vs Transformer embeddings** a parità di architettura di ranking.

---

## 8. Test del modello (precomputed)

Script:

```bash
python main_t5PRO.py test \
  --dataset Baby_5_data \
  --model DeepCoNN \
  --setup BPR \
  --use_precomputed True \
  --precomputed_mode deepconn
```

Il test utilizza gli stessi embeddings precomputati del training.

---

## 9. Note sulle modifiche al codice

Per supportare la modalità precomputed sono state introdotte modifiche in:

* `ReviewData`: caricamento automatico embeddings 2D/3D
* `main_t5PRO.py`: gestione `use_precomputed`
* `BPR_t5PRO.py`: supporto a input già embedded
* `config`: aggiunta parametri per embeddings

Queste modifiche permettono di usare **la stessa pipeline di training** sia con CNN sia con embeddings T5Gemma .
