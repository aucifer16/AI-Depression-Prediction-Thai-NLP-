# 🧠 AI Depression Prediction (Thai NLP) — Naive Bayes

โค้ดและเอกสารประกอบงานวิจัย **นวัตกรรมสื่อด้วยปัญญาประดิษฐ์เพื่อทำนายภาวะซึมเศร้า** สำหรับภาษาไทย โดยใช้เทคนิค **Natural Language Processing (NLP)** และ **Naive Bayes Classifier**

> **Main script:** `runPredictAll.py`  
> สคริปต์นี้โหลดโมเดลที่เทรนไว้ (`finalized_model.pickle`) และ `vocabulary_model.pickle` จากนั้นอ่านไฟล์ CSV คอลัมน์ `text` แล้วทำนายผลเป็นคอลัมน์ `Result`

---

## ✨ คุณสมบัติเด่น
- ตัดคำภาษาไทยด้วย `PyThaiNLP`
- แปลงข้อความเป็น feature โดยอ้างอิงคำศัพท์ (vocabulary) จากโมเดลที่เทรนไว้
- จัดจำแนกข้อความเป็นเชิงบวก/ลบ/กลาง (ตามที่โมเดลถูกสอนมา)
- รับ/ส่งข้อมูลเป็น CSV ใช้งานง่าย

---

## 📂 โครงสร้างโปรเจกต์ (แนะนำ)
```
.
├── runPredictAll.py                   # สคริปต์หลักสำหรับรันทำนาย
├── finalized_model.pickle             # โมเดล Naive Bayes ที่เทรนไว้
├── vocabulary_model.pickle            # คำศัพท์/ฟีเจอร์ที่ใช้กับโมเดล
├── usertwitter_iamsobad15feed.csv     # ข้อมูลอินพุต ต้องมีคอลัมน์ 'text'
├── usertwitter_iamsobad15feed_Predit.csv  # ไฟล์ผลลัพธ์หลังรันทำนาย (auto-gen)
└── README.md
```

> หมายเหตุ: ชื่อไฟล์ CSV อินพุต/เอาต์พุตในสคริปต์ตั้งตายตัว หากต้องการใช้ชื่ออื่นให้แก้ใน `runPredictAll.py`

---

## ⚙️ การติดตั้ง (Installation)

แนะนำให้ใช้ Python 3.9+

```bash
# 1) สร้างและเปิดใช้งาน virtual env (แนะนำ)
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
# source .venv/bin/activate

# 2) ติดตั้งไลบรารีที่จำเป็น
pip install nltk pythainlp pandas

# 3) ดาวน์โหลดข้อมูล tokenizers ของ NLTK (เช่น punkt)
python -m nltk.downloader punkt
```

> หากระบบถามหาโมดูลเพิ่มเติมของ PyThaiNLP (เช่น `pythainlp.corpus`) ให้ติดตั้งเพิ่มตามข้อความแจ้ง

---

## 🚀 การใช้งาน (Usage)

เตรียมไฟล์ต่อไปนี้ไว้ในโฟลเดอร์เดียวกัน:
- `finalized_model.pickle` (โมเดลที่เทรนแล้ว)
- `vocabulary_model.pickle` (ชุดคำศัพท์ที่ใช้ทำฟีเจอร์)
- `usertwitter_iamsobad15feed.csv` (ไฟล์อินพุต ต้องมีคอลัมน์ `text` ที่เป็นข้อความภาษาไทย)

ตัวอย่างสั่งรัน:
```bash
python runPredictAll.py
```

สคริปต์จะ:
1) โหลดโมเดลและ vocabulary  
2) ตัดคำและแปลงข้อความเป็นฟีเจอร์  
3) ทำนายผลและบันทึกลงไฟล์ `usertwitter_iamsobad15feed_Predit.csv` (จะมีคอลัมน์ `Result` เพิ่มขึ้น)

ตัวอย่างโครง CSV อินพุต:
```csv
text
"รู้สึกเหนื่อยมากเลยวันนี้"
"วันนี้อากาศดีจัง"
```

ตัวอย่างผลลัพธ์ (ตัดให้ดูบางส่วน):
```csv
text,Result
"รู้สึกเหนื่อยมากเลยวันนี้",negative
"วันนี้อากาศดีจัง",positive
```

> หมายเหตุ: ชนิดป้ายกำกับ (`positive/negative/neutral`) จะขึ้นกับวิธีที่โมเดลถูกฝึกและแมปปิ้งในขั้นตอนเทรนเดิม

---

## 🧪 หมายเหตุด้านโมเดล
- โมเดลนี้ถูกเทรนและบันทึกไว้แล้วใน `finalized_model.pickle` และใช้คำศัพท์จาก `vocabulary_model.pickle`
- หากจะฝึกใหม่ ควรจัดเตรียมชุดข้อมูลที่มีความสมดุลของ **positive/neutral/negative** และระวัง **class imbalance**  
- เนื่องจากเป็น **Bag-of-Words (BoW)** + Naive Bayes: ความหมายจากบริบทซับซ้อน, เสียดสี, อีโมจิ ฯลฯ อาจทำนายได้ยาก ควรพิจารณาเพิ่มข้อมูลฝึกหรือเทคนิคเสริม

---

## 🔐 จริยธรรมและความเป็นส่วนตัว (Ethics & Privacy)
- ข้อมูลโซเชียลมีเดียควรถูกทำให้ไม่สามารถระบุตัวตนได้ (anonymized)
- ใช้เพื่อการวิจัย/การศึกษา/คัดกรองเบื้องต้นเท่านั้น **ไม่ใช่เครื่องมือวินิจฉัยทางการแพทย์**
- ปฏิบัติตามกฎหมายคุ้มครองข้อมูลส่วนบุคคล (เช่น PDPA)

---

## 👩‍💻 ผู้เขียน (Authors)
- Patcharin Boonsomthop (NIDA)  
- Chutisant Kerdvibulvech (NIDA)

---

## 📄 License
โค้ดและโมเดลนี้เผยแพร่เพื่อ **การศึกษาและวิจัย** เท่านั้น ไม่อนุญาตให้นำไปใช้เชิงพาณิชย์หรือระบุตัวบุคคลจากข้อมูลโซเชียลมีเดีย

---

### ✅ Quick Checklist ก่อนเปิดใช้งานจริง
- [ ] วางไฟล์ `finalized_model.pickle` และ `vocabulary_model.pickle` ถูกที่
- [ ] ตรวจสอบว่า CSV อินพุตมีคอลัมน์ `text`
- [ ] ลง `nltk`, `pythainlp`, `pandas` ครบถ้วน
- [ ] ดาวน์โหลด `punkt` สำหรับ NLTK เรียบร้อย
- [ ] ไฟล์ผลลัพธ์จะชื่อ `usertwitter_iamsobad15feed_Predit.csv`
