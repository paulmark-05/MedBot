# =============================================================
# rag.py — ChromaDB knowledge base for Medical Health FAQ Bot
# 15 documents — added dedicated docs for stroke, vaccines/autism,
# and COPD vs asthma to fix retrieval gaps identified in test run.
# =============================================================

import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHROMA_COLLECTION, RAG_TOP_K

# ── 15 medical knowledge documents ───────────────────────────
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Hypertension (High Blood Pressure)",
        "text": """Hypertension, or high blood pressure, occurs when the force of blood against artery walls is consistently too high, defined as readings at or above 130/80 mmHg. It is a leading risk factor for heart disease, stroke, and kidney failure. Most people with hypertension experience no symptoms, earning it the nickname 'the silent killer.' Risk factors include obesity, excessive salt intake, physical inactivity, smoking, excessive alcohol, stress, family history, and age over 55. Diagnosis is confirmed by multiple blood pressure readings on separate occasions. Lifestyle modifications are the first-line treatment: reducing sodium to under 2,300 mg/day, following the DASH diet rich in fruits and vegetables, exercising 150 minutes per week, limiting alcohol, quitting smoking, and managing stress. When lifestyle changes are insufficient, medications are prescribed. Common classes include ACE inhibitors (e.g., lisinopril), ARBs (e.g., losartan), calcium channel blockers (e.g., amlodipine), and thiazide diuretics. Blood pressure should be monitored regularly at home with a validated cuff. Target BP for most adults is below 130/80 mmHg. Uncontrolled hypertension over years damages blood vessels and organs silently.""",
    },
    {
        "id": "doc_002",
        "topic": "Type 2 Diabetes",
        "text": """Type 2 diabetes is a chronic metabolic condition where the body either resists the effects of insulin or doesn't produce enough to maintain normal glucose levels. It accounts for 90-95% of all diabetes cases. Symptoms include increased thirst, frequent urination, fatigue, blurred vision, slow-healing sores, and frequent infections. Many people are asymptomatic for years. Risk factors include being overweight or obese (especially central adiposity), physical inactivity, family history, age over 45, prediabetes, gestational diabetes history, and certain ethnicities. Diagnosis uses fasting glucose ≥ 126 mg/dL, HbA1c ≥ 6.5%, or a 2-hour oral glucose tolerance test ≥ 200 mg/dL. Management focuses on blood sugar control through lifestyle changes: a balanced diet limiting refined carbohydrates and sugars, regular aerobic exercise, and weight loss of 5-10% body weight. Medications start with metformin, then may include SGLT-2 inhibitors, GLP-1 receptor agonists, DPP-4 inhibitors, sulfonylureas, or insulin. HbA1c should be monitored every 3 months initially, targeting below 7% for most adults. Regular screening for complications including neuropathy, retinopathy, nephropathy, and cardiovascular disease is essential.""",
    },
    {
        "id": "doc_003",
        "topic": "Asthma",
        "text": """Asthma is a chronic inflammatory disease of the airways characterised by episodes of wheezing, breathlessness, chest tightness, and coughing, often worse at night or early morning. The airways become narrowed, swollen, and may produce extra mucus. Triggers vary between individuals and include allergens (pollen, dust mites, pet dander, mold), air pollutants, respiratory infections, exercise, cold air, smoke, strong odours, and emotional stress. Diagnosis is based on symptom history, physical examination, and spirometry measuring FEV1/FVC ratio. Classification ranges from intermittent to severe persistent. Treatment follows a stepwise approach. Reliever (rescue) inhalers containing short-acting beta-agonists (SABA), such as salbutamol, provide rapid bronchodilation during attacks. Controller medications include inhaled corticosteroids (ICS) like budesonide or fluticasone as the cornerstone, with long-acting beta-agonists (LABA), leukotriene receptor antagonists, and biologics (e.g., omalizumab for severe allergic asthma) added stepwise. An Asthma Action Plan helps patients recognise worsening symptoms. Peak flow monitoring at home is useful for tracking control. Well-controlled asthma allows normal daily activity with minimal symptoms.""",
    },
    {
        "id": "doc_004",
        "topic": "Common Cold vs Influenza (Flu)",
        "text": """The common cold and influenza are both respiratory illnesses caused by viruses, but they differ significantly in severity and onset. Colds are usually caused by rhinoviruses and develop gradually. Symptoms include runny or stuffy nose, sore throat, sneezing, and mild cough; fever is rare in adults. Influenza is caused by influenza A or B viruses and comes on abruptly. Symptoms include high fever (38-40°C), severe muscle aches and chills, intense fatigue, headache, dry cough, and sometimes vomiting and diarrhoea. Flu complications can be serious, including pneumonia, hospitalisation, and death, particularly in the elderly, young children, pregnant women, and immunocompromised individuals. Treatment for both is primarily supportive: rest, adequate hydration, and over-the-counter medicines for symptom relief (paracetamol or ibuprofen for fever and aches). Antiviral medications (oseltamivir/Tamiflu, zanamivir) are effective for flu if started within 48 hours of symptom onset. Annual flu vaccination is the most effective preventive measure and is recommended for everyone over 6 months. Cold prevention relies on frequent handwashing and avoiding close contact with infected individuals. Antibiotics are ineffective against both because they are viral infections.""",
    },
    {
        "id": "doc_005",
        "topic": "Mental Health — Anxiety and Depression",
        "text": """Anxiety disorders and depression are the world's most common mental health conditions. Anxiety disorders include generalised anxiety disorder (GAD), panic disorder, social anxiety, and phobias. Symptoms of GAD include persistent excessive worry, restlessness, fatigue, difficulty concentrating, irritability, muscle tension, and sleep disturbances. Depression is characterised by persistent low mood, loss of interest or pleasure in activities, changes in appetite and weight, sleep disturbances, fatigue, feelings of worthlessness or excessive guilt, difficulty concentrating, and in severe cases, thoughts of death or suicide. Both conditions are diagnosed clinically using standardised tools (e.g., PHQ-9 for depression, GAD-7 for anxiety). They often co-exist. Treatment is multimodal. Psychotherapy, particularly cognitive-behavioural therapy (CBT), is evidence-based for both. Medications include SSRIs (e.g., sertraline, escitalopram) and SNRIs as first-line pharmacotherapy. Lifestyle factors — regular exercise, sleep hygiene, reduced alcohol and caffeine, social connection, and mindfulness — significantly support recovery. Crisis intervention is needed for suicidal ideation. If you or someone you know is in crisis, contact a mental health emergency line immediately.""",
    },
    {
        "id": "doc_006",
        "topic": "Nutrition and Healthy Eating",
        "text": """Good nutrition is fundamental to health, energy, immunity, and disease prevention. A balanced diet should include five food groups: fruits and vegetables (aim for at least 5 portions daily, providing vitamins, minerals, fibre, and antioxidants); starchy carbohydrates (whole grains preferred — brown rice, oats, whole wheat — for sustained energy and fibre); proteins (lean meats, fish twice weekly especially oily fish, legumes, eggs, dairy or fortified alternatives); healthy fats (unsaturated from olive oil, avocado, nuts, seeds; limit saturated fat from butter and processed meats; avoid trans fats); and dairy or calcium-rich alternatives for bone health. Hydration is critical — 6 to 8 glasses of water daily is recommended. Key nutrients to monitor include iron (especially for women of childbearing age), vitamin D (supplementation recommended in many populations), calcium, vitamin B12 (critical for vegans), omega-3 fatty acids, and folate (essential in pregnancy). Ultra-processed foods, high in added sugars, salt, and unhealthy fats, are linked to obesity, cardiovascular disease, and diabetes. Portion control matters as much as food choice. The Mediterranean diet consistently ranks among the best evidence-based dietary patterns for longevity and cardiovascular health.""",
    },
    {
        "id": "doc_007",
        "topic": "Vaccines and Immunisation",
        "text": """Vaccines train the immune system to recognise and fight specific pathogens without causing the disease itself. They work by introducing antigens (weakened/killed pathogens, protein subunits, or mRNA encoding a protein) that stimulate production of antibodies and immunological memory. Vaccine types include live-attenuated (MMR, chickenpox), inactivated (flu shot, hepatitis A), subunit/protein (hepatitis B, whooping cough component), mRNA (COVID-19 Pfizer/Moderna), and viral vector (some COVID-19 vaccines). The childhood immunisation schedule covers diseases including diphtheria, tetanus, pertussis (DTaP), polio, Haemophilus influenzae type b, hepatitis B, measles, mumps, rubella (MMR), varicella, rotavirus, and pneumococcal disease. Adult vaccines include annual influenza, Tdap booster every 10 years, shingles vaccine (Shingrix, recommended from age 50), pneumococcal vaccines for over-65s, and HPV vaccine up to age 26 (or 45 with shared decision-making). Herd immunity is achieved when enough of a population is immune that disease spread is interrupted, protecting those who cannot be vaccinated. Vaccine side effects are generally mild (soreness, low-grade fever) and resolve within days. Serious adverse events are extremely rare and monitored through pharmacovigilance systems.""",
    },
    {
        "id": "doc_008",
        "topic": "Sleep Health and Sleep Disorders",
        "text": """Sleep is essential for physical repair, memory consolidation, immune function, metabolism regulation, and mental health. Adults require 7 to 9 hours of quality sleep per night; teenagers need 8 to 10 hours. Chronic sleep deprivation is linked to obesity, type 2 diabetes, cardiovascular disease, impaired immunity, depression, and cognitive decline. Good sleep hygiene practices include maintaining a consistent sleep-wake schedule even on weekends, keeping the bedroom dark, cool, and quiet, avoiding screens (blue light) 1 hour before bed, limiting caffeine after 2 pm, avoiding large meals and alcohol close to bedtime, and using the bed only for sleep and sex. Common sleep disorders include insomnia (difficulty falling or staying asleep, affecting 10-30% of adults), obstructive sleep apnoea (OSA, characterised by repeated breathing interruptions causing snoring and daytime sleepiness, treated with CPAP therapy or lifestyle changes), restless legs syndrome (uncomfortable leg sensations compelling movement at night), and narcolepsy (excessive daytime sleepiness and sudden muscle weakness). Insomnia is best treated with cognitive-behavioural therapy for insomnia (CBT-I) rather than long-term sleeping pills. Melatonin may help with jet lag and circadian rhythm disorders.""",
    },
    {
        "id": "doc_009",
        "topic": "Cardiovascular Disease Prevention",
        "text": """Cardiovascular disease (CVD), including coronary artery disease, heart failure, and stroke, is the leading cause of death globally. The major modifiable risk factors are smoking, high blood pressure, high LDL cholesterol, type 2 diabetes, physical inactivity, obesity, unhealthy diet, excessive alcohol, and psychological stress. Non-modifiable risk factors include age, male sex, and family history. Prevention is classified as primary (preventing a first event) and secondary (preventing recurrence). Primary prevention strategies include not smoking, maintaining blood pressure below 130/80 mmHg, LDL cholesterol below 2.6 mmol/L (or lower in high-risk individuals), HbA1c below 7% if diabetic, BMI 18.5-24.9 kg/m², regular aerobic exercise (150+ minutes/week moderate intensity), a heart-healthy diet (Mediterranean pattern, low in saturated fat and sodium, rich in fibre), and limiting alcohol. Statins (e.g., atorvastatin, rosuvastatin) are prescribed when 10-year CVD risk is elevated, lowering LDL and plaque inflammation. Aspirin for primary prevention is now reserved for specific high-risk cases due to bleeding risk. Blood pressure and cholesterol should be checked regularly from age 40. Knowing the FAST signs of stroke (Face drooping, Arm weakness, Speech difficulty, Time to call emergency) can save lives.""",
    },
    {
        "id": "doc_010",
        "topic": "Antibiotic Use and Resistance",
        "text": """Antibiotics are medicines that kill or inhibit bacteria. They are effective only against bacterial infections and have no effect on viral illnesses such as the common cold, flu, most sore throats, or COVID-19. Overuse and misuse of antibiotics has driven a global crisis of antimicrobial resistance (AMR), where bacteria evolve to survive antibiotic treatment. AMR already causes over 1.2 million deaths annually and is projected to overtake cancer as a leading cause of death by 2050. Responsible antibiotic use means taking antibiotics only when prescribed by a doctor, completing the full prescribed course even if feeling better, never sharing antibiotics or using leftover courses, and never pressuring doctors for antibiotics when not needed. Common bacterial infections that may require antibiotics include strep throat (confirmed by swab), urinary tract infections, pneumonia, skin infections, and Lyme disease. Common classes include penicillins (amoxicillin), cephalosporins, macrolides (azithromycin), tetracyclines (doxycycline), and fluoroquinolones. Side effects include diarrhoea, nausea, rash, and Clostridioides difficile (C. diff) colitis with disrupted gut microbiome. Probiotics during antibiotic treatment may help reduce gastrointestinal side effects.""",
    },
    {
        "id": "doc_011",
        "topic": "Exercise and Physical Activity Guidelines",
        "text": """Regular physical activity is one of the most powerful interventions for health across the lifespan. The World Health Organization recommends adults aged 18-64 accumulate at least 150-300 minutes of moderate-intensity aerobic activity per week (e.g., brisk walking, cycling, swimming) OR 75-150 minutes of vigorous-intensity activity (e.g., running, aerobics, fast cycling), plus muscle-strengthening activities involving all major muscle groups on 2 or more days per week. Children and adolescents need at least 60 minutes of moderate-to-vigorous activity daily. Older adults should add balance and coordination exercises to prevent falls. Benefits of regular exercise include reduced risk of cardiovascular disease (up to 35%), type 2 diabetes (up to 40%), colon and breast cancer (up to 25%), depression and anxiety, osteoporosis, and all-cause mortality. Exercise improves sleep quality, cognitive function, self-esteem, and longevity. Sedentary time should be minimised — prolonged sitting is independently associated with health risks even in active individuals. Starting exercise: begin gradually with 10-15 minute sessions, build up over weeks, choose enjoyable activities for adherence, and consult a doctor before starting vigorous exercise if previously inactive or with chronic conditions.""",
    },
    {
        "id": "doc_012",
        "topic": "First Aid Essentials",
        "text": """Basic first aid knowledge can save lives in emergencies before professional help arrives. For cardiac arrest, call emergency services immediately, then begin hands-only CPR: 30 hard, fast chest compressions (100-120/min) in the centre of the chest, followed by 2 rescue breaths if trained; repeat until help arrives or an AED is available. AEDs (automated external defibrillators) are found in many public places and provide spoken instructions. For choking in adults, perform up to 5 back blows between the shoulder blades, then up to 5 abdominal thrusts (Heimlich manoeuvre); alternate until cleared or unconscious. For burns, cool under running cool water for 20 minutes; do not use ice, butter, or toothpaste; cover with a clean non-fluffy dressing. For bleeding, apply firm direct pressure with a clean cloth; elevate the injured limb; maintain pressure for 10+ minutes without peeking. For suspected fractures, immobilise the area, apply ice wrapped in cloth to reduce swelling, and seek medical care. For anaphylaxis (severe allergic reaction), use an epinephrine auto-injector (EpiPen) if available, call emergency services, lie the person flat with legs raised (unless breathing difficulty), and be ready to give CPR. Never give aspirin to children under 16.""",
    },
    # ── FIX: Dedicated stroke document (test 7 was retrieving CVD doc
    #        where FAST signs appeared only at the very end of text,
    #        beyond the eval node's 600-char truncation window) ──────
    {
        "id": "doc_013",
        "topic": "Stroke — Recognition, FAST Signs, and Emergency Response",
        "text": """A stroke occurs when blood supply to part of the brain is cut off, either by a clot (ischaemic stroke, 85% of cases) or a burst blood vessel (haemorrhagic stroke). Every minute without treatment, approximately 1.9 million neurons die — hence the phrase 'time is brain.' Recognising a stroke immediately is critical. The FAST acronym is the international standard for rapid stroke recognition: F — Face drooping (ask the person to smile; does one side droop?); A — Arm weakness (ask them to raise both arms; does one drift downward?); S — Speech difficulty (is speech slurred, garbled, or hard to understand?); T — Time to call emergency services immediately. Do not wait to see if symptoms improve — call emergency services at once. Additional stroke symptoms include sudden severe headache with no known cause, sudden vision disturbance in one or both eyes, sudden dizziness or loss of balance, and sudden confusion or trouble understanding. Ischaemic strokes can be treated with clot-busting drugs (tPA/alteplase) if given within 4.5 hours of symptom onset, or mechanical thrombectomy up to 24 hours in selected patients. Haemorrhagic strokes require surgical intervention or blood pressure management. Post-stroke rehabilitation involving physiotherapy, speech therapy, and occupational therapy significantly improves recovery outcomes. Stroke risk factors include hypertension (the leading risk factor), atrial fibrillation, smoking, diabetes, high cholesterol, obesity, and physical inactivity.""",
    },
    # ── FIX: Dedicated COPD vs Asthma document (test 3 retrieved the
    #        asthma doc which lacks COPD comparison detail) ───────────
    {
        "id": "doc_014",
        "topic": "Asthma vs COPD — Differences, Diagnosis, and Management",
        "text": """Asthma and Chronic Obstructive Pulmonary Disease (COPD) are both chronic airway diseases causing breathing difficulty, but they differ fundamentally in cause, reversibility, and patient profile. Asthma typically begins in childhood or early adulthood, is driven by allergic inflammation and airway hyper-responsiveness, and is largely reversible with treatment. Triggers include allergens, exercise, cold air, and infections. COPD, which includes chronic bronchitis and emphysema, is almost always caused by long-term cigarette smoking (80-90% of cases) or prolonged exposure to harmful particles or gases. It typically presents after age 40, with progressive irreversible airflow obstruction. Key differences: Asthma — variable airflow obstruction that reverses with bronchodilators; FEV1/FVC ratio normalises post-bronchodilator; symptoms fluctuate. COPD — fixed or incompletely reversible airflow obstruction; FEV1/FVC persistently below 0.70 post-bronchodilator; symptoms are progressive and constant. Symptoms overlap: both cause wheeze, breathlessness, and cough. Asthma cough is often nocturnal; COPD cough is chronic productive morning cough. Asthma treatment: inhaled corticosteroids (ICS) are the cornerstone, plus SABA relievers. COPD treatment: long-acting bronchodilators (LABA, LAMA) are primary; ICS added for frequent exacerbators; smoking cessation is the most important intervention. Some patients have features of both — termed Asthma-COPD Overlap (ACO). Spirometry is essential to distinguish the two conditions. Both conditions benefit from pulmonary rehabilitation and vaccination against influenza and pneumococcus.""",
    },
    # ── FIX: Dedicated vaccine safety document (test 10 asked about
    #        vaccines and autism — the existing doc_007 covers schedule
    #        but not the autism myth, causing a refusal) ──────────────
    {
        "id": "doc_015",
        "topic": "Vaccine Safety — Myths, Side Effects, and the Autism Claim",
        "text": """Vaccine safety is rigorously monitored through pre-licensure clinical trials involving tens of thousands of participants and post-market pharmacovigilance systems including the Vaccine Adverse Event Reporting System (VAERS) and the WHO Global Vaccine Safety Initiative. The claim that vaccines cause autism originates from a 1998 Lancet paper by Andrew Wakefield, which was fully retracted in 2010 after investigations found the data was deliberately falsified and Wakefield had undisclosed financial conflicts of interest. Wakefield subsequently lost his medical licence. Over 20 large-scale epidemiological studies involving millions of children across multiple countries — including a 2019 Danish cohort study of 650,000 children — have found absolutely no link between the MMR vaccine or any vaccine ingredient and autism spectrum disorder. Scientific and regulatory consensus is unambiguous: vaccines do not cause autism. Autism symptoms often become apparent around the same age children receive scheduled vaccines, which explains the perceived (but entirely coincidental) temporal association. Common, expected vaccine side effects include injection-site soreness, redness, mild fever, and fatigue lasting 1-2 days — these indicate normal immune activation, not harm. Serious adverse events such as anaphylaxis occur in approximately 1-2 per million doses and are manageable in clinical settings. The risks of vaccine-preventable diseases — measles encephalitis, polio paralysis, pertussis in infants — vastly outweigh any risk from vaccination. Herd immunity through vaccination protects vulnerable individuals who cannot be vaccinated.""",
    },
]


# ── Build and return ChromaDB collection ─────────────────────

def build_knowledge_base() -> tuple:
    """
    Initialises the SentenceTransformer embedder and ChromaDB collection.
    Returns (embedder, collection) for use in other modules.
    """
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.Client()
    # Fresh collection on every startup (in-memory client)
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(CHROMA_COLLECTION)

    texts       = [d["text"]  for d in DOCUMENTS]
    ids         = [d["id"]    for d in DOCUMENTS]
    metadatas   = [{"topic": d["topic"]} for d in DOCUMENTS]
    embeddings  = embedder.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    print(f"✅ Knowledge base ready: {collection.count()} documents")
    for d in DOCUMENTS:
        print(f"   • {d['topic']}")
    return embedder, collection


def retrieve(question: str, embedder, collection) -> tuple[str, list[str]]:
    """
    Queries ChromaDB and returns (formatted_context, list_of_topic_names).
    FIX: Removed arbitrary text truncation here — truncation was the
    responsibility of the caller (eval_node), not retrieval.
    The full chunk text is preserved so eval_node sees complete evidence.
    """
    q_emb   = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=RAG_TOP_K)
    chunks  = results["documents"][0]
    topics  = [m["topic"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(
        f"[SOURCE: {topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
    )
    return context, topics


# ── Convenience list of all topic names (used by Streamlit sidebar) ──
KB_TOPICS = [d["topic"] for d in DOCUMENTS]