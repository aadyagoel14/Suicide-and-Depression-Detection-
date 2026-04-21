# 🧠 Detecting Suicide Ideation from Text

## Why this project exists

Every day, people express distress online — sometimes subtly, sometimes directly.  
The challenge is that these signals are easy to miss at scale.

This project explores how deep learning can help identify potentially harmful or suicidal intent in text, with the goal of supporting early intervention systems.

It is not about replacing human judgment — but about assisting it.

---

## 💡 What this project does

Given a piece of text, the model predicts whether it reflects:
- Suicide-related intent
- Non-suicidal content

Example:

Input:
"Through these past years thoughts of suicide, fear, anxiety I’m so close to my limit"

Output:
⚠️ Potential Suicide Post (confidence ~0.96)

---

## ⚙️ How it works (high-level)

Instead of treating text as raw words, the model:

1. Converts words into vectors using pretrained embeddings (GloVe)
2. Processes the sequence using an LSTM to capture context
3. Extracts important signals using pooling
4. Outputs a probability score

In simple terms:
It learns patterns in how distress is expressed in language.

---

## 📊 Performance

- ~93% accuracy on a balanced test set  
- Strong precision and recall across both classes  

Class-wise performance:

- Non-suicide → Precision: 0.92, Recall: 0.95  
- Suicide → Precision: 0.95, Recall: 0.91  

---

## 🚀 Try it yourself

```python
twt = ["i feel like giving up"]
twt = tokenizer.texts_to_sequences(twt)
twt = pad_sequences(twt, maxlen=50)

prediction = model.predict(twt)[0][0]

if prediction > 0.5:
    print("⚠️ Potential Suicide Post")
else:
    print("✅ Non Suicide Post")
```

---

## 🧠 What I learned

- Language around mental health is nuanced and context-heavy  
- Pretrained embeddings help, but lack deep contextual understanding  
- False negatives are especially critical in this domain  
- Building such systems requires both technical and ethical awareness  

---

## ⚠️ Important note

This is a research project, not a production-ready system.

It should not be used as a standalone decision-maker.

Any real-world deployment must include:
- Human oversight  
- Privacy safeguards  
- Responsible intervention mechanisms  

---

## 🔮 Future improvements

- Use transformer-based models (e.g., BERT)
- Fine-tune embeddings instead of freezing
- Improve recall for suicide-related class
- Handle subtle and indirect expressions better

---

## 👤 Author

Aadya Goel  
AI/ML | Deep Learning | Bioinformatics
