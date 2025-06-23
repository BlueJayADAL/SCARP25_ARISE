import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class WorkoutTherapyFAQMatcher:
    def __init__(self, glove_path: str = None, embedding_dim: int = 100):
        self.embedding_dim = embedding_dim
        self.glove_dict = {}
        self.faq_embeddings = {}
        self.faqs = {}

        # Fitness/therapy specific stop words to ignore
        self.domain_stopwords = {'workout', 'exercise', 'therapy', 'physical', 'fitness'}

        if glove_path:
            self.load_glove_embeddings(glove_path)

    def load_glove_embeddings(self, glove_path: str):
        print(f"Loading GloVe embeddings for workout/therapy FAQ matching...")
        priority_words = {
            'pain', 'muscle', 'joint', 'strength', 'flexibility', 'mobility', 'stretch',
            'rehabilitation', 'injury', 'recovery', 'cardio', 'weights', 'resistance',
            'balance', 'coordination', 'posture', 'breathing', 'relaxation', 'tension',
            'shoulder', 'knee', 'back', 'neck', 'hip', 'ankle', 'wrist', 'elbow',
            'squat', 'lunge', 'pushup', 'plank', 'deadlift', 'press', 'curl', 'row',
            'yoga', 'pilates', 'massage', 'stretching', 'warmup', 'cooldown'
        }

        try:
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    word = parts[0]
                    vector = np.array([float(x) for x in parts[1:]])
                    if len(vector) == self.embedding_dim:
                        if word in priority_words or len(self.glove_dict) < 20000:
                            self.glove_dict[word] = vector
            print(f"Loaded {len(self.glove_dict)} word embeddings (fitness/therapy optimized)")
        except FileNotFoundError:
            print(f"GloVe file not found: {glove_path}")

    def preprocess_fitness_text(self, text: str) -> List[str]:
        text = text.lower()
        text = text.replace('pt', 'physical therapy')
        text = text.replace('rom', 'range of motion')
        text = text.replace('reps', 'repetitions')
        text = text.replace('sets', 'repetitions')
        text = text.replace('lbs', 'pounds')
        text = text.replace('kg', 'kilograms')
        text = re.sub(r'[^\w\s\-/]', '', text)
        words = text.split()

        synonym_map = {
            # Movement verbs
            "affect": "target", "affects": "target", "engage": "target", "engages": "target", "work": "target",
            "works": "target", "targeting": "target", "involves": "target", "involve": "target",
            "hit": "target", "activate": "target", "activates": "target", "build": "strengthen", "builds": "strengthen",
            "grow": "strengthen", "develop": "strengthen",

            # Muscles & body parts
            "muscles": "muscle", "biceps": "bicep", "triceps": "tricep", "glutes": "glute", "quads": "quadriceps",
            "hamstrings": "hamstring", "abs": "core", "abdominals": "core", "delts": "shoulders", "pecs": "chest",
            "calf": "calves", "forearm": "forearms",

            # Exercises
            "curls": "curl", "presses": "press", "raises": "raise", "pullups": "pullup", "pushups": "pushup",
            "deadlifts": "deadlift", "rows": "row", "squats": "squat", "lunges": "lunge", "planks": "plank",
            "crunches": "crunch", "dips": "dip", "burpees": "burpee", "climbers": "mountain_climbers",

            # Equipment
            "weights": "dumbbells", "dumbbell": "dumbbells", "barbells": "barbell", "bands": "resistance_band",
            "balls": "stability_ball", "rollers": "foam_roller", "mat": "exercise_mat", "pullup_bar": "bar",

            # Recovery & therapy
            "therapy": "rehabilitation", "rehab": "rehabilitation", "soreness": "pain", "tightness": "stiffness",
            "warm-up": "warmup", "cool-down": "cooldown", "stretching": "stretch",

            # General
            "routine": "workout", "session": "workout", "exercises": "exercise", "training": "exercise"
        }
        words = [synonym_map.get(w, w) for w in words if w not in self.domain_stopwords and len(w) > 2]
        return words

    def get_sentence_embedding(self, sentence: str) -> np.ndarray:
        words = self.preprocess_fitness_text(sentence)
        vectors = []
        emphasis_words = {"muscle", "target", "work", "engage", "bicep", "glute", "hamstring", "shoulder"}

        for word in words:
            vec = None
            if word in self.glove_dict:
                vec = self.glove_dict[word] * (2.0 if word in emphasis_words else 1.0)
            elif word.endswith("s") and word[:-1] in self.glove_dict:
                vec = self.glove_dict[word[:-1]]
            if vec is not None:
                vectors.append(vec)

        return np.mean(vectors, axis=0) if vectors else np.zeros(self.embedding_dim)

    def add_faq(self, question: str, answer: str, category: str, 
               body_part: str = None, difficulty: str = "beginner", 
               equipment: str = None, duration: str = None):
        faq_id = len(self.faqs)
        self.faqs[faq_id] = {
            'question': question,
            'answer': answer,
            'category': category,
            'body_part': body_part,
            'difficulty': difficulty,
            'equipment': equipment,
            'duration': duration,
            'keywords': self.extract_fitness_keywords(question + " " + answer)
        }
        self.faq_embeddings[faq_id] = self.get_sentence_embedding(question + " " + answer)

    def extract_fitness_keywords(self, text: str) -> List[str]:
        fitness_terms = {
            'exercises': ['squat', 'lunge', 'pushup', 'pullup', 'plank', 'deadlift', 'press', 'curl', 'row'],
            'body_parts': ['shoulder', 'knee', 'back', 'neck', 'hip', 'ankle', 'wrist', 'elbow', 'core', 'bicep', 'tricep', 'glute', 'hamstring'],
            'conditions': ['pain', 'stiffness', 'weakness', 'instability', 'tension', 'soreness'],
            'equipment': ['dumbbell', 'barbell', 'band', 'ball', 'mat', 'foam', 'roller'],
            'modalities': ['stretch', 'massage', 'ice', 'heat', 'ultrasound', 'electrical']
        }
        keywords = []
        text_lower = text.lower()
        for category, terms in fitness_terms.items():
            for term in terms:
                if term in text_lower:
                    keywords.append(term)
        return keywords

    def find_best_match(self, query: str, threshold: float = 0.4, top_k: int = 3,
                       filter_category: str = None, filter_body_part: str = None) -> List[Tuple[int, float, dict]]:
        if not self.faq_embeddings:
            return []

        query_embedding = self.get_sentence_embedding(query)
        query_keywords = self.extract_fitness_keywords(query)
        inferred_body_part = next((k for k in query_keywords if k in [
            'shoulder', 'knee', 'back', 'neck', 'hip', 'ankle', 'wrist', 'elbow', 'core', 'bicep', 'tricep', 'glute', 'hamstring'
        ]), None)

        similarities = []
        for faq_id, faq_embedding in self.faq_embeddings.items():
            faq_data = self.faqs[faq_id]

            if filter_category and faq_data['category'] != filter_category:
                continue
            if filter_body_part and faq_data['body_part'] != filter_body_part:
                continue

            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                faq_embedding.reshape(1, -1)
            )[0][0]

            common_keywords = set(query_keywords) & set(faq_data['keywords'])
            keyword_boost = len(common_keywords) * 0.1

            final_score = similarity + keyword_boost

            if inferred_body_part and faq_data['body_part'] == inferred_body_part:
                final_score += 0.1

            if final_score >= threshold:
                similarities.append((faq_id, final_score, faq_data))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_category_summary(self) -> Dict[str, int]:
        categories = {}
        for faq in self.faqs.values():
            cat = faq['category']
            categories[cat] = categories.get(cat, 0) + 1
        return categories

# Setup comprehensive workout/therapy FAQ database
def setup_workout_therapy_faqs():
    """Setup comprehensive workout and therapy FAQ database"""
    matcher = WorkoutTherapyFAQMatcher(glove_path = "models/glove.6B/glove.6B.300d.txt",embedding_dim=300)
    
    # ========== EXERCISE TECHNIQUE FAQs ==========
    exercise_faqs = [
        # Lower Body Exercises
        ("How do I perform a proper squat?", 
         "Stand with feet shoulder-width apart, toes slightly outward. Lower by pushing hips back and bending knees, keep chest up and knees tracking over toes. Descend until thighs are parallel to floor, then drive through heels to return to start. Keep core engaged throughout.",
         "exercise", "legs", "beginner", "bodyweight", "2-3min"),
        
        ("What's the correct form for lunges?",
         "Step forward into a lunge, lowering hips until both knees are bent at 90 degrees. Front knee should be directly above ankle, back knee nearly touching ground. Push back to starting position. Keep torso upright and core engaged.",
         "exercise", "legs", "beginner", "bodyweight", "2-3min"),
        
        ("How do I do Bulgarian split squats?",
         "Stand 2-3 feet in front of bench, place top of one foot behind you on bench. Lower into lunge position until front thigh is parallel to floor. Push through front heel to return. Keep most weight on front leg.",
         "exercise", "legs", "intermediate", "bench", "3-4min"),
        
        ("How do I perform a deadlift?",
         "Stand with feet hip-width apart, bar over mid-foot. Hinge at hips to grip bar, keep back neutral and chest up. Drive through heels and hips to lift, keeping bar close to body. Reverse the movement to lower with control.",
         "exercise", "back", "intermediate", "barbell", "3-5min"),
        
        ("How do I perform Romanian deadlifts?",
         "Hold barbell with overhand grip, feet hip-width apart. Keep knees slightly bent, hinge at hips pushing them back. Lower bar along legs until you feel hamstring stretch. Drive hips forward to return to start.",
         "exercise", "hamstrings", "intermediate", "barbell", "3-4min"),
        
        ("What's correct hip thrust form?",
         "Sit with back against bench, barbell over hips. Plant feet flat, drive through heels to lift hips up, squeezing glutes at top. Lower with control. Keep core tight and avoid arching back excessively.",
         "exercise", "glutes", "intermediate", "barbell", "3-4min"),
        
        ("How do I do calf raises properly?",
         "Stand on balls of feet on elevated surface, heels hanging off. Lower heels below platform level, then rise up on toes as high as possible. Control the movement, pause at top and bottom.",
         "exercise", "calves", "beginner", "bodyweight", "2-3min"),
        
        # Upper Body Exercises
        ("What's the correct pushup form?",
         "Start in plank position with hands slightly wider than shoulders. Lower body as one unit until chest nearly touches floor, then press back up. Keep core tight, body straight, and avoid sagging hips.",
         "exercise", "chest", "beginner", "bodyweight", "1-2min"),
        
        ("How do I do incline pushups?",
         "Place hands on elevated surface (bench, step). Perform pushup with body at angle. Higher the surface, easier the movement. Maintain straight line from head to heels throughout.",
         "exercise", "chest", "beginner", "bench", "2-3min"),
        
        ("What's proper pull-up technique?",
         "Hang from bar with overhand grip, hands shoulder-width apart. Pull body up until chin clears bar, then lower with control. Engage lats and avoid swinging. Use band assistance if needed.",
         "exercise", "back", "intermediate", "pullup_bar", "3-5min"),
        
        ("How do I perform chin-ups correctly?",
         "Similar to pull-ups but use underhand grip. This targets biceps more. Pull up until chin clears bar, lower slowly. Keep shoulders down and back throughout movement.",
         "exercise", "back", "intermediate", "pullup_bar", "3-5min"),
        
        ("What's correct dumbbell row form?",
         "Hinge at hips, hold dumbbell in one hand. Pull weight to hip, squeezing shoulder blade back. Lower with control. Keep core tight and avoid rotating torso. Support with opposite hand if needed.",
         "exercise", "back", "beginner", "dumbbells", "2-3min"),
        
        ("How do I do overhead press safely?",
         "Stand with feet hip-width apart, hold weights at shoulder level. Press straight up until arms are fully extended overhead. Lower with control. Keep core tight and avoid arching back excessively.",
         "exercise", "shoulders", "intermediate", "dumbbells", "3-4min"),
        
        ("What's proper bench press technique?",
         "Lie on bench, grip barbell slightly wider than shoulders. Lower bar to chest with control, then press up powerfully. Keep feet planted, shoulder blades pulled back, and maintain natural arch.",
         "exercise", "chest", "intermediate", "barbell", "4-5min"),
        
        ("How do I perform dumbbell chest press?",
         "Lie on bench holding dumbbells at chest level. Press weights up and slightly inward until arms are extended. Lower with control, feeling stretch in chest. Keep shoulder blades pulled back.",
         "exercise", "chest", "beginner", "dumbbells", "3-4min"),
        
        ("What's correct bicep curl form?",
         "Stand with dumbbells at sides, elbows close to torso. Curl weights up by flexing biceps, keeping elbows stationary. Lower slowly with control. Avoid swinging or using momentum.",
         "exercise", "arms", "beginner", "dumbbells", "2-3min"),
        
        ("How do I do tricep dips properly?",
         "Sit on edge of bench, hands beside hips. Lower body by bending elbows until shoulders are below elbows. Push back up. Keep legs extended for harder variation, bent for easier.",
         "exercise", "arms", "beginner", "bench", "2-3min"),
        
        ("What's proper lateral raise technique?",
         "Hold dumbbells at sides with slight bend in elbows. Raise arms out to sides until parallel to floor. Lower slowly. Keep wrists neutral and avoid using momentum or swinging.",
         "exercise", "shoulders", "beginner", "dumbbells", "2-3min"),
        
        # Core Exercises
        ("How do I perform a proper plank?",
         "Start in pushup position, lower to forearms. Keep body straight from head to heels, engage core. Hold position without sagging hips or raising butt. Breathe normally throughout.",
         "exercise", "core", "beginner", "bodyweight", "1-2min"),
        
        ("What's correct side plank form?",
         "Lie on side, prop up on forearm with elbow under shoulder. Lift hips to create straight line from head to feet. Hold position, then switch sides. Keep core engaged throughout.",
         "exercise", "core", "beginner", "bodyweight", "2-3min"),
        
        ("How do I do mountain climbers?",
         "Start in plank position. Alternate bringing knees toward chest in running motion. Keep core tight, hips level, and maintain plank position throughout. Move at controlled pace.",
         "exercise", "core", "intermediate", "bodyweight", "2-3min"),
        
        ("What's proper bicycle crunch technique?",
         "Lie on back, hands behind head, knees bent. Bring opposite elbow to knee while extending other leg. Alternate sides in cycling motion. Keep lower back pressed to floor.",
         "exercise", "core", "beginner", "bodyweight", "2-3min"),
        
        ("How do I perform dead bugs correctly?",
         "Lie on back, arms extended up, knees bent at 90 degrees. Lower opposite arm and leg slowly, then return. Keep lower back pressed to floor throughout. Move slowly and controlled.",
         "exercise", "core", "beginner", "bodyweight", "3-4min"),
        
        ("What's correct bird dog form?",
         "Start on hands and knees. Extend opposite arm and leg simultaneously, hold briefly, then return. Keep hips level and core engaged. Avoid rotating torso or sagging back.",
         "exercise", "core", "beginner", "bodyweight", "2-3min"),
        
        # Compound Movements
        ("How do I do burpees properly?",
         "Start standing, drop to squat and place hands on floor. Jump feet back to plank, do pushup, jump feet to squat, then jump up with arms overhead. Land softly and repeat.",
         "exercise", "full_body", "intermediate", "bodyweight", "3-4min"),
        
        ("What's correct thruster technique?",
         "Hold dumbbells at shoulders, squat down, then drive up explosively while pressing weights overhead in one fluid motion. Lower weights to shoulders and repeat.",
         "exercise", "full_body", "advanced", "dumbbells", "4-5min"),
        
        ("How do I perform Turkish get-ups?",
         "Lie on back holding weight overhead. Use multiple steps to stand up while keeping weight stable overhead. Reverse the movement to return to lying position. Very technical movement requiring practice.",
         "exercise", "full_body", "advanced", "kettlebell", "5-8min"),
        
        # Flexibility and Mobility
        ("How do I stretch my hamstrings safely?",
         "Sit with one leg extended, other bent. Reach toward extended foot, keeping back straight. Hold 30 seconds. Alternatively, lie on back and pull leg toward chest with towel or band.",
         "exercise", "hamstrings", "beginner", "bodyweight", "2-3min"),
        
        ("What's proper hip flexor stretch technique?",
         "Kneel in lunge position, push hips forward while keeping back straight. Feel stretch in front of rear leg. Hold 30-60 seconds each side. Can be done standing against wall.",
         "exercise", "hip", "beginner", "bodyweight", "2-3min"),
        
        ("How do I stretch my chest muscles?",
         "Stand in doorway with arm against frame at 90 degrees. Step forward to feel stretch across chest and front shoulder. Hold 30 seconds each arm. Adjust arm height for different angles.",
         "exercise", "chest", "beginner", "bodyweight", "1-2min"),
    ]
    
    
    # ========== INJURY PREVENTION & RECOVERY FAQs ==========
    injury_faqs = [
        # Knee Issues
        ("How can I prevent knee pain during squats?",
         "Ensure proper form: knees track over toes, don't cave inward. Warm up thoroughly, strengthen glutes and hip stabilizers. Start with bodyweight, progress gradually. Consider reducing depth if pain persists.",
         "injury", "knee", "beginner", "bodyweight", "5-10min"),
        
        ("What causes knee pain when running?",
         "Common causes include poor running form, weak hips/glutes, tight IT band, inappropriate footwear, or sudden training increases. Focus on gradual progression, strength training, and proper recovery.",
         "injury", "knee", "beginner", None, "ongoing"),
        
        ("How do I strengthen my knees after injury?",
         "Start with straight leg raises, wall sits, and gentle range of motion. Progress to mini squats, step-ups, and balance exercises. Avoid deep bending initially. Work with physical therapist for serious injuries.",
         "injury", "knee", "intermediate", "bodyweight", "15-20min"),
        
        ("What are signs of serious knee injury?",
         "Severe pain, inability to bear weight, knee giving way, significant swelling, deformity, or locking. Seek immediate medical attention for these symptoms. Don't ignore persistent pain.",
         "injury", "knee", "beginner", None, "immediate"),
        
        # Back Problems
        ("What should I do for lower back pain after lifting?",
         "Apply ice for first 24-48 hours if acute. Gentle movement like walking helps recovery. Avoid bed rest. Focus on hip flexor stretches and core strengthening. Seek medical advice if severe.",
         "injury", "back", "beginner", None, "ongoing"),
        
        ("How can I prevent lower back injury?",
         "Maintain proper lifting form, strengthen core muscles, improve hip mobility, warm up properly, and avoid sudden increases in training intensity. Focus on hip hinge movement patterns.",
         "injury", "back", "beginner", "bodyweight", "daily"),
        
        ("What exercises help chronic lower back pain?",
         "Cat-cow stretches, pelvic tilts, bird dogs, dead bugs, and gentle yoga poses. Strengthen deep core muscles and improve hip flexibility. Avoid exercises that cause pain.",
         "injury", "back", "beginner", "bodyweight", "10-15min"),
        
        ("When should I see a doctor for back pain?",
         "Seek immediate care for severe pain, numbness/tingling in legs, loss of bladder/bowel control, or pain after trauma. See doctor if pain persists beyond 72 hours or worsens significantly.",
         "injury", "back", "beginner", None, "immediate"),
        
        # Shoulder Issues
        ("How do I treat shoulder impingement?",
         "Rest from overhead activities, ice for inflammation, focus on posture correction and scapular stabilization exercises. Gentle range of motion exercises. Physical therapy often beneficial for proper rehabilitation.",
         "injury", "shoulder", "intermediate", None, "2-6weeks"),
        
        ("What causes shoulder pain during bench press?",
         "Common causes include poor form, muscle imbalances, tight chest muscles, weak rear delts, or pressing too heavy. Focus on proper scapular position and balanced training.",
         "injury", "shoulder", "intermediate", "barbell", "ongoing"),
        
        ("How can I prevent rotator cuff injuries?",
         "Strengthen rotator cuff muscles with band exercises, maintain good posture, avoid overhead overuse, warm up properly, and balance pushing with pulling exercises. Focus on scapular stability.",
         "injury", "shoulder", "beginner", "resistance_band", "daily"),
        
        ("What are rotator cuff strengthening exercises?",
         "External rotations with band, wall slides, face pulls, and Y-T-W raises. Start light and focus on form. Perform 2-3 sets of 12-15 reps. Progress slowly to avoid re-injury.",
         "injury", "shoulder", "beginner", "resistance_band", "10-15min"),
        
        # Ankle and Foot
        ("How do I recover from ankle sprain?",
         "R.I.C.E. protocol initially (Rest, Ice, Compression, Elevation). Begin gentle range of motion exercises early. Progress to weight-bearing and balance training. Proper rehabilitation prevents re-injury.",
         "injury", "ankle", "beginner", None, "2-6weeks"),
        
        ("What exercises prevent ankle injuries?",
         "Single-leg balance training, calf raises, ankle circles, and proprioceptive exercises. Strengthen peroneal muscles with band exercises. Focus on landing mechanics in sports.",
         "injury", "ankle", "beginner", "resistance_band", "10-15min"),
        
        ("How do I treat plantar fasciitis?",
         "Stretch calf muscles and plantar fascia, roll foot on tennis ball, ice after activity, wear supportive shoes, and avoid walking barefoot on hard surfaces. Gradual return to activity.",
         "injury", "foot", "beginner", "tennis_ball", "daily"),
        
        # Wrist and Elbow
        ("What causes tennis elbow?",
         "Overuse of forearm muscles from repetitive gripping and wrist extension. Common in racquet sports, weightlifting, and computer work. Rest, ice, and gradual strengthening help recovery.",
         "injury", "elbow", "beginner", None, "ongoing"),
        
        ("How do I prevent wrist pain during pushups?",
         "Use pushup handles or do pushups on fists to maintain neutral wrist position. Strengthen wrists with extensions and flexions. Gradually increase volume to build tolerance.",
         "injury", "wrist", "beginner", "pushup_handles", "ongoing"),
        
        ("What are signs of overtraining?",
         "Persistent fatigue, decreased performance, mood changes, frequent illness, elevated resting heart rate, sleep disturbances, and loss of motivation. Rest and recovery are essential.",
         "injury", None, "intermediate", None, "ongoing"),
        
        # General Injury Prevention
        ("How important is warming up before exercise?",
         "Critical for injury prevention and performance. Increases blood flow, joint mobility, and muscle temperature. Spend 5-10 minutes on dynamic movements that mimic your workout activities.",
         "injury", None, "beginner", "bodyweight", "5-10min"),
        
        ("What's the difference between muscle soreness and injury?",
         "Normal soreness is bilateral, develops 24-48 hours post-exercise, and decreases with movement. Injury pain is often immediate, localized, sharp, and may worsen with movement. When in doubt, rest.",
         "injury", None, "beginner", None, "immediate"),
        
        ("How do I know when to return to exercise after injury?",
         "Full range of motion without pain, strength equal to uninjured side, ability to perform daily activities normally, and clearance from healthcare provider if serious. Don't rush back too early.",
         "injury", None, "intermediate", None, "varies"),
    ]
    
    
    # ========== EQUIPMENT & SETUP FAQs ==========
    equipment_faqs = [
        # Home Gym Basics
        ("What equipment do I need for home workouts?",
         "Start with resistance bands, adjustable dumbbells, and yoga mat. Add stability ball and foam roller as you progress. These cover strength, cardio, flexibility, and recovery needs for complete fitness.",
         "equipment", None, "beginner", "various", None),
        
        ("How do I set up a home gym on a budget?",
         "Prioritize versatile equipment: resistance bands ($20), adjustable dumbbells ($100-200), yoga mat ($30). Add pullup bar ($30) and stability ball ($25). Total under $300 for complete setup.",
         "equipment", None, "beginner", "various", None),
        
        ("What's the best flooring for home gym?",
         "Rubber mats provide cushioning and protect floors. Interlocking foam tiles work for lighter exercises. Avoid carpet for weightlifting. Consider sound dampening if you have neighbors below.",
         "equipment", None, "beginner", "flooring", None),
        
        ("How much space do I need for home workouts?",
         "Minimum 6x6 feet for basic exercises. 8x10 feet ideal for full range of movements. Clear overhead space for jumping and overhead exercises. Consider foldable equipment for small spaces.",
         "equipment", None, "beginner", None, None),
        
        # Resistance Equipment
        ("How do I choose the right resistance band?",
         "Light bands for rehabilitation and beginners, medium for general fitness, heavy for strength training. Loop bands for glute activation, tube bands with handles for full-body exercises. Consider sets with multiple resistances.",
         "equipment", None, "beginner", "resistance_band", "2-3min"),
        
        ("What weight should I start with for dumbbells?",
         "Begin with weights allowing 12-15 repetitions with good form while feeling challenged in last 2-3 reps. Typically 5-15 lbs for beginners, but varies by exercise and individual strength.",
         "equipment", None, "beginner", "dumbbells", None),
        
        ("Are adjustable dumbbells worth it?",
         "Yes for home gyms - save space and money long-term. Look for quick-change mechanisms. Quality options: PowerBlocks, Bowflex SelectTech, or plate-loaded versions. Consider weight range needed.",
         "equipment", None, "beginner", "dumbbells", None),
        
        ("What's better: dumbbells or kettlebells?",
         "Dumbbells better for isolation exercises and bilateral training. Kettlebells excel for dynamic movements and unilateral training. Start with dumbbells if choosing one, add kettlebells later for variety.",
         "equipment", None, "intermediate", "various", None),
        
        ("How do I choose a barbell for home gym?",
         "Standard Olympic barbell (45 lbs, 7 feet) most versatile. Consider shorter bars for limited space. Look for good knurling and spin. Budget $100-300. Add safety bars/rack for heavy lifting.",
         "equipment", None, "intermediate", "barbell", None),
        
        # Cardio Equipment
        ("What's the best cardio equipment for home?",
         "Jump rope most affordable and effective. Stationary bike for low-impact. Treadmill if space/budget allows. Rowing machine provides full-body workout. Consider your preferences and joint health.",
         "equipment", None, "beginner", "cardio", None),
        
        ("Is a treadmill worth buying for home?",
         "Consider usage frequency, space, and budget. Quality treadmills are expensive but durable. Alternatives: outdoor running, walking, or less expensive cardio options may be more practical.",
         "equipment", None, "beginner", "treadmill", None),
        
        ("What should I look for in a stationary bike?",
         "Adjustable seat and handlebars, smooth pedaling action, resistance levels, comfortable seat, and stability. Upright vs recumbent depends on comfort and back issues. Consider smart features if desired.",
         "equipment", None, "beginner", "bike", None),
        
        # Recovery Equipment
        ("What foam roller should I buy?",
         "Medium density for beginners, firmer for experienced users. 36-inch length most versatile. Avoid very firm rollers initially. Textured surfaces provide deeper massage but may be uncomfortable at first.",
         "equipment", None, "beginner", "foam_roller", None),
        
        ("Do I need a massage gun?",
         "Helpful but not essential. Good for targeted muscle release and convenience. More expensive than foam rollers. Look for adjustable speeds, quiet operation, and good battery life if purchasing.",
         "equipment", None, "intermediate", "massage_gun", None),
        
        ("What recovery tools are most important?",
         "Foam roller, lacrosse ball, and resistance bands cover most needs. Add ice packs and heating pad for injury management. Quality sleep and nutrition more important than expensive recovery gadgets.",
         "equipment", None, "beginner", "various", None),
        
        # Specialized Equipment
        ("Is a pull-up bar necessary?",
         "Excellent for upper body strength development. Doorway bars convenient but check weight limits and door frame strength. Wall-mounted or ceiling-mounted options more stable for serious training.",
         "equipment", None, "intermediate", "pullup_bar", None),
        
        ("What type of exercise mat should I get?",
         "6mm thickness good balance of cushioning and stability. Non-slip surface important. 68+ inches length for tall people. Consider eco-friendly materials. Yoga mats work for most floor exercises.",
         "equipment", None, "beginner", "mat", None),
        
        ("Do I need a weight bench?",
         "Adjustable bench adds exercise variety, especially for dumbbell work. Look for sturdy construction, smooth adjustment, and comfortable padding. Flat bench sufficient if budget/space limited.",
         "equipment", None, "intermediate", "bench", None),
        
        ("What about suspension trainers?",
         "TRX-style trainers very versatile for bodyweight strength training. Great for travel and small spaces. Can anchor to door, tree, or ceiling. Hundreds of exercises possible with one tool.",
         "equipment", None, "intermediate", "suspension_trainer", None),
        
        # Technology and Apps
        ("Are fitness apps worth using?",
         "Many excellent free options available. Good for structure, progression tracking, and motivation. Examples: Nike Training Club, Adidas Training, YouTube channels. Paid apps offer more personalization.",
         "equipment", None, "beginner", "app", None),
        
        ("Should I buy a fitness tracker?",
         "Helpful for motivation and tracking progress. Basic step counters affordable. Advanced models track heart rate, sleep, and recovery metrics. Not essential but can increase accountability.",
         "equipment", None, "beginner", "fitness_tracker", None),
        
        ("What's the best way to track workouts?",
         "Simple notebook works well. Smartphone apps offer convenience and analysis. Key: record exercises, sets, reps, and weights consistently. Focus on progressive overload over time.",
         "equipment", None, "beginner", "tracking", None),
        
        # Safety Equipment
        ("Do I need weightlifting gloves?",
         "Help with grip and prevent calluses but not essential. May reduce grip strength development. Chalk or liquid grip alternatives. Focus on proper grip technique first.",
         "equipment", None, "beginner", "gloves", None),
        
        ("Are lifting belts necessary?",
         "Not needed for beginners or light weights. Helpful for heavy squats/deadlifts (85%+ 1RM). Don't rely on belt for all exercises. Focus on natural core strengthening first.",
         "equipment", None, "advanced", "lifting_belt", None),
        
        ("What safety equipment is essential?",
         "First aid kit, water bottle, towel, and phone for emergencies. Safety bars for heavy lifting. Proper footwear with good support. Clear workout space free of obstacles.",
         "equipment", None, "beginner", "safety", None),
    ]
    
    # Physical Therapy FAQs
    # Extended Physical Therapy FAQs - Comprehensive coverage for better GloVe matching
    therapy_faqs = [
        # ========== SHOULDER REHABILITATION ==========
        ("What exercises help with frozen shoulder?",
        "Gentle pendulum swings, wall slides, cross-body stretches, and external rotation with resistance band. Progress gradually and stay within pain-free range. Consistency is key for mobility improvement.",
        "therapy", "shoulder", "beginner", "resistance_band", "10-15min"),
        
        ("How do I treat shoulder impingement syndrome?",
        "Focus on posterior capsule stretching, scapular stabilization exercises, and strengthening the rotator cuff. Avoid overhead activities initially. Use ice for inflammation and heat before stretching.",
        "therapy", "shoulder", "intermediate", "resistance_band", "15-20min"),
        
        ("What are the best rotator cuff strengthening exercises?",
        "External rotation with resistance band, internal rotation, empty can exercise, full can exercise, and prone horizontal abduction. Start with light resistance and progress slowly.",
        "therapy", "shoulder", "beginner", "resistance_band", "12-15min"),
        
        ("How can I improve shoulder blade mobility?",
        "Wall slides, doorway stretches, upper trap stretches, cross-body stretches, and scapular squeezes. Focus on retracting and depressing the shoulder blades throughout daily activities.",
        "therapy", "shoulder", "beginner", "bodyweight", "8-12min"),
        
        ("What helps with shoulder bursitis pain?",
        "Rest from aggravating activities, ice application, gentle range of motion exercises, and anti-inflammatory measures. Avoid sleeping on affected side. Progress to strengthening as pain decreases.",
        "therapy", "shoulder", "beginner", "ice", "ongoing"),
        
        ("How do I rehabilitate after shoulder dislocation?",
        "Begin with gentle pendulum exercises, progress to assisted range of motion, then active range of motion. Strengthen rotator cuff and scapular stabilizers. Avoid aggressive stretching initially.",
        "therapy", "shoulder", "intermediate", "resistance_band", "4-6weeks"),
        
        ("What exercises help thoracic outlet syndrome?",
        "Nerve gliding exercises, scalene stretches, pectoralis stretches, and posture correction exercises. Strengthen deep neck flexors and middle trapezius. Avoid overhead positions initially.",
        "therapy", "shoulder", "intermediate", "bodyweight", "10-15min"),
        
        # ========== BACK AND SPINE REHABILITATION ==========
        ("What's the best treatment for herniated disc?",
        "Extension-based exercises like prone press-ups, walking, and avoiding prolonged sitting. Ice for acute pain, then heat for muscle relaxation. Progress to core strengthening and flexibility.",
        "therapy", "back", "intermediate", "bodyweight", "3-6weeks"),
        
        ("How do I strengthen my core after back injury?",
        "Start with diaphragmatic breathing, dead bugs, bird dogs, and modified planks. Progress to side planks, pallof press, and functional movements. Focus on quality over quantity.",
        "therapy", "back", "beginner", "bodyweight", "12-20min"),
        
        ("What exercises help sciatica pain?",
        "Nerve gliding exercises, piriformis stretches, hamstring stretches, and spinal extension exercises. Avoid forward bending initially. Walking and swimming often provide relief.",
        "therapy", "back", "intermediate", "bodyweight", "15-25min"),
        
        ("How can I improve posture-related back pain?",
        "Strengthen deep neck flexors, rhomboids, and lower trapezius. Stretch hip flexors, pectorals, and upper trapezius. Use ergonomic workstation setup and take frequent breaks.",
        "therapy", "back", "beginner", "resistance_band", "daily"),
        
        ("What helps with muscle spasms in the back?",
        "Apply heat, gentle stretching, deep breathing, and light walking. Avoid bed rest. Use trigger point release techniques and consider massage therapy for persistent spasms.",
        "therapy", "back", "beginner", "heat", "immediate"),
        
        ("How do I rehabilitate after spinal fusion surgery?",
        "Follow surgeon's protocol strictly. Begin with gentle walking, progress to core strengthening, and functional activities. Avoid twisting, bending, and lifting restrictions as specified.",
        "therapy", "back", "advanced", "various", "3-6months"),
        
        # ========== KNEE REHABILITATION ==========
        ("How can I improve ankle mobility after sprain?",
        "Alphabet draws with toe, calf stretches against wall, resistance band exercises for all directions. Balance training on one foot. Progress from non-weight bearing to full weight bearing activities.",
        "therapy", "ankle", "beginner", "resistance_band", "10-20min"),
        
        ("What exercises help with patellofemoral pain syndrome?",
        "Quadriceps strengthening, especially VMO, hip strengthening, IT band stretching, and patellar mobilization. Avoid deep squats and lunges initially. Focus on proper movement patterns.",
        "therapy", "knee", "beginner", "resistance_band", "15-20min"),
        
        ("How do I rehabilitate after ACL reconstruction?",
        "Phase-based approach: initial protection and range of motion, then strengthening, and finally return to sport activities. Follow physical therapist's protocol strictly for optimal outcomes.",
        "therapy", "knee", "advanced", "various", "4-6months"),
        
        ("What helps with meniscus tear recovery?",
        "Range of motion exercises, quadriceps strengthening, hamstring flexibility, and progressive weight bearing. Avoid pivoting and deep squatting initially. Swimming is excellent for conditioning.",
        "therapy", "knee", "intermediate", "pool", "6-12weeks"),
        
        ("How can I strengthen my knees after arthroscopy?",
        "Straight leg raises, mini squats, stationary cycling, and balance training. Progress gradually from non-weight bearing to full weight bearing activities. Ice after exercise sessions.",
        "therapy", "knee", "beginner", "bike", "4-8weeks"),
        
        ("What exercises prevent knee osteoarthritis progression?",
        "Low-impact strengthening, range of motion exercises, weight management, and activity modification. Focus on quadriceps, hamstrings, and hip muscles. Avoid high-impact activities.",
        "therapy", "knee", "intermediate", "pool", "daily"),
        
        # ========== HIP AND PELVIS REHABILITATION ==========
        ("What's good for hip flexor tightness?",
        "Low lunge stretches, couch stretch, standing hip flexor stretch. Strengthen glutes and core. Avoid prolonged sitting. Hold stretches 30-60 seconds, repeat 2-3 times daily.",
        "therapy", "hip", "beginner", "bodyweight", "5-10min"),
        
        ("How do I treat hip bursitis naturally?",
        "Avoid aggravating activities, use ice for inflammation, gentle stretching, and strengthening exercises. Side-lying leg lifts, clamshells, and bridges help strengthen supporting muscles.",
        "therapy", "hip", "beginner", "ice", "2-4weeks"),
        
        ("What exercises help with hip impingement?",
        "Hip flexor stretches, piriformis stretches, and strengthening the deep hip rotators. Avoid deep hip flexion initially. Focus on improving hip mobility and core stability.",
        "therapy", "hip", "intermediate", "resistance_band", "12-18min"),
        
        ("How can I rehabilitate after hip replacement?",
        "Follow surgeon's precautions, gentle range of motion, progressive strengthening, and gait training. Avoid hip flexion beyond 90 degrees, internal rotation, and adduction initially.",
        "therapy", "hip", "advanced", "various", "3-6months"),
        
        ("What helps with sacroiliac joint dysfunction?",
        "Pelvic stabilization exercises, gluteal strengthening, and gentle mobilization techniques. Avoid asymmetrical activities. Use support belt if recommended by healthcare provider.",
        "therapy", "hip", "intermediate", "belt", "4-8weeks"),
        
        ("How do I strengthen glutes after hip injury?",
        "Bridges, clamshells, side-lying leg lifts, and monster walks with resistance band. Progress to single-leg exercises and functional movements. Focus on proper activation patterns.",
        "therapy", "hip", "beginner", "resistance_band", "10-15min"),
        
        # ========== ANKLE AND FOOT REHABILITATION ==========
        ("How do I treat plantar fasciitis effectively?",
        "Calf stretches, plantar fascia stretches, ice rolling, and arch support. Avoid walking barefoot on hard surfaces. Stretch first thing in morning and after prolonged sitting.",
        "therapy", "foot", "beginner", "tennis_ball", "daily"),
        
        ("What exercises help with Achilles tendinitis?",
        "Eccentric calf raises, gentle stretching, and progressive loading exercises. Ice after activity, avoid complete rest. Heel drops and calf stretches are particularly beneficial.",
        "therapy", "ankle", "intermediate", "step", "daily"),
        
        ("How can I improve balance after ankle injury?",
        "Single-leg standing, eyes closed balance, unstable surface training, and dynamic balance exercises. Progress from stable to unstable surfaces and static to dynamic challenges.",
        "therapy", "ankle", "beginner", "balance_pad", "10-15min"),
        
        ("What helps with ankle stiffness?",
        "Ankle pumps, circles, alphabet exercises, and calf stretches. Use heat before stretching and ice after if swollen. Gentle mobilization and range of motion exercises throughout day.",
        "therapy", "ankle", "beginner", "heat", "daily"),
        
        ("How do I prevent recurrent ankle sprains?",
        "Proprioceptive training, strengthening peroneal muscles, balance exercises, and wearing appropriate footwear. Avoid uneven surfaces until fully recovered.",
        "therapy", "ankle", "intermediate", "balance_pad", "ongoing"),
        
        ("What exercises help with Morton's neuroma?",
        "Toe stretches, calf stretches, metatarsal pad use, and avoiding tight shoes. Ice for pain relief and consider activity modification to reduce pressure on affected area.",
        "therapy", "foot", "beginner", "ice", "ongoing"),
        
        # ========== NECK AND CERVICAL SPINE ==========
        ("How do I treat cervical radiculopathy?",
        "Neck decompression exercises, nerve gliding, gentle range of motion, and posture correction. Avoid extreme neck positions and use proper pillow support during sleep.",
        "therapy", "neck", "intermediate", "pillow", "2-6weeks"),
        
        ("What helps with tension headaches from neck problems?",
        "Upper trap stretches, suboccipital releases, posture exercises, and stress management. Apply heat to neck muscles and practice relaxation techniques.",
        "therapy", "neck", "beginner", "heat", "daily"),
        
        ("How can I improve forward head posture?",
        "Chin tucks, deep neck flexor strengthening, upper trap stretches, and ergonomic improvements. Strengthen posterior neck muscles and stretch anterior structures.",
        "therapy", "neck", "beginner", "bodyweight", "daily"),
        
        ("What exercises help whiplash recovery?",
        "Gentle range of motion, isometric exercises, and gradual return to normal activities. Avoid aggressive stretching initially. Use ice for acute pain and heat for muscle tension.",
        "therapy", "neck", "intermediate", "ice", "2-6weeks"),
        
        # ========== WRIST AND ELBOW REHABILITATION ==========
        ("How do I treat carpal tunnel syndrome?",
        "Nerve gliding exercises, wrist stretches, ergonomic modifications, and activity modification. Wear splint at night and avoid repetitive gripping activities.",
        "therapy", "wrist", "beginner", "splint", "4-8weeks"),
        
        ("What helps with tennis elbow recovery?",
        "Eccentric strengthening, stretching, ice after activity, and activity modification. Use counterforce bracing and avoid gripping activities that cause pain.",
        "therapy", "elbow", "intermediate", "brace", "6-12weeks"),
        
        ("How can I rehabilitate after wrist fracture?",
        "Range of motion exercises, strengthening, and functional activities. Progress from gentle movements to normal activities as healing allows. Follow physician's guidelines.",
        "therapy", "wrist", "intermediate", "various", "6-12weeks"),
        
        ("What exercises help golfer's elbow?",
        "Flexor stretches, eccentric strengthening, and activity modification. Use ice after activity and avoid repetitive gripping. Focus on proper technique in sports.",
        "therapy", "elbow", "intermediate", "ice", "6-8weeks"),
    ]

    # ========== EXTENDED RECOVERY & WELLNESS FAQs ==========
    recovery_faqs = [
        # ========== REST AND RECOVERY PRINCIPLES ==========
        ("How long should I rest between workouts?",
        "Allow 48-72 hours for the same muscle groups. You can train different muscle groups on consecutive days. Listen to your body - fatigue, soreness, or decreased performance indicate need for more rest.",
        "recovery", None, "beginner", None, "ongoing"),
        
        ("What are signs I need more recovery time?",
        "Persistent fatigue, declining performance, elevated resting heart rate, mood changes, frequent illness, trouble sleeping, loss of motivation, and excessive muscle soreness lasting more than 72 hours.",
        "recovery", None, "intermediate", None, "ongoing"),
        
        ("How much sleep do I need for proper recovery?",
        "Most adults need 7-9 hours of quality sleep. Athletes may need 8-10 hours. Poor sleep impairs muscle recovery, immune function, and performance. Maintain consistent sleep schedule.",
        "recovery", None, "beginner", None, "daily"),
        
        ("What's the difference between active and passive recovery?",
        "Active recovery involves light movement like walking, gentle yoga, or swimming. Passive recovery is complete rest. Active recovery often promotes better circulation and faster healing.",
        "recovery", None, "beginner", "bodyweight", "20-30min"),
        
        ("How do I know if I'm overtraining?",
        "Watch for decreased performance, chronic fatigue, mood changes, frequent injuries, insomnia, loss of appetite, elevated resting heart rate, and lack of motivation to exercise.",
        "recovery", None, "intermediate", None, "ongoing"),
        
        ("What's the best recovery strategy after intense workouts?",
        "Cool down properly, hydrate, consume protein and carbohydrates within 30 minutes, stretch, use foam rolling, apply ice if needed, and prioritize sleep and stress management.",
        "recovery", None, "intermediate", "various", "30-60min"),
        
        # ========== FOAM ROLLING AND SELF-MASSAGE ==========
        ("What's the best way to foam roll?",
        "Roll slowly over muscle groups, pausing on tender spots for 30-60 seconds. Apply moderate pressure - discomfort is normal but avoid severe pain. Focus on calves, IT band, quads, and glutes.",
        "recovery", None, "beginner", "foam_roller", "10-15min"),
        
        ("How often should I foam roll?",
        "Daily foam rolling is beneficial, especially before workouts for activation and after for recovery. Focus on tight areas and muscles you'll be training that day.",
        "recovery", None, "beginner", "foam_roller", "daily"),
        
        ("Which muscles should I foam roll regularly?",
        "IT band, quadriceps, hamstrings, calves, glutes, upper back, and lats. These areas commonly develop tightness from sitting, training, and daily activities.",
        "recovery", None, "beginner", "foam_roller", "15-20min"),
        
        ("Is foam rolling better before or after workouts?",
        "Both are beneficial. Light rolling before workouts helps activate muscles and improve mobility. Deeper rolling after workouts aids recovery and reduces muscle tension.",
        "recovery", None, "beginner", "foam_roller", "5-15min"),
        
        ("Can foam rolling replace stretching?",
        "No, they serve different purposes. Foam rolling addresses fascial restrictions and muscle quality, while stretching improves flexibility and joint range of motion. Both are valuable.",
        "recovery", None, "intermediate", "foam_roller", "combined"),
        
        ("What's the difference between foam rolling and massage?",
        "Foam rolling is self-administered and focuses on fascial release. Professional massage can address deeper layers, provide relaxation, and target specific problem areas more precisely.",
        "recovery", None, "intermediate", "foam_roller", "varies"),
        
        ("How do I use a lacrosse ball for trigger points?",
        "Place ball between body and wall or floor, apply moderate pressure to trigger points for 30-90 seconds. Move slowly and breathe deeply. Effective for feet, glutes, and upper back.",
        "recovery", None, "beginner", "lacrosse_ball", "5-10min"),
        
        # ========== WARMING UP AND COOLING DOWN ==========
        ("How important is warming up before exercise?",
        "Very important - increases blood flow, joint mobility, and reduces injury risk. Spend 5-10 minutes on dynamic movements that mimic your workout. Include arm circles, leg swings, and bodyweight movements.",
        "recovery", None, "beginner", "bodyweight", "5-10min"),
        
        ("What's the best warm-up routine?",
        "Start with light cardio for 3-5 minutes, then dynamic stretches like leg swings, arm circles, hip circles, and movement-specific exercises. Progress from general to specific movements.",
        "recovery", None, "beginner", "bodyweight", "8-12min"),
        
        ("Why is cooling down important after exercise?",
        "Helps gradually lower heart rate, prevents blood pooling, reduces muscle stiffness, and begins the recovery process. Include light walking and static stretching.",
        "recovery", None, "beginner", "bodyweight", "5-10min"),
        
        ("What should I include in my cool-down routine?",
        "5 minutes of light walking or gentle movement, followed by static stretches for major muscle groups used during workout. Hold stretches for 30-60 seconds each.",
        "recovery", None, "beginner", "bodyweight", "10-15min"),
        
        ("How long should I hold static stretches?",
        "Hold for 30-60 seconds for flexibility improvements. Shorter holds (15-30 seconds) are sufficient for maintenance. Avoid bouncing and stretch to mild discomfort, not pain.",
        "recovery", None, "beginner", "bodyweight", "varies"),
        
        ("What's the difference between dynamic and static stretching?",
        "Dynamic stretching involves movement and is best for warm-ups. Static stretching involves holding positions and is best for cool-downs and flexibility improvement.",
        "recovery", None, "beginner", "bodyweight", "varies"),
        
        # ========== HYDRATION AND NUTRITION FOR RECOVERY ==========
        ("How much water should I drink during exercise?",
        "Drink 17-20 oz 2-3 hours before exercise, 8 oz 20-30 minutes before, and 7-10 oz every 10-20 minutes during exercise. Adjust based on sweat rate and conditions.",
        "recovery", None, "beginner", "water", "ongoing"),
        
        ("What should I eat after a workout for best recovery?",
        "Consume protein and carbohydrates within 30-60 minutes post-workout. Aim for 20-25g protein and 30-60g carbohydrates. Examples: chocolate milk, protein shake with banana, or Greek yogurt with berries.",
        "recovery", None, "beginner", "nutrition", "30-60min"),
        
        ("How does protein intake affect muscle recovery?",
        "Protein provides amino acids necessary for muscle repair and growth. Aim for 1.6-2.2g per kg body weight daily for active individuals. Distribute intake throughout the day.",
        "recovery", None, "intermediate", "nutrition", "daily"),
        
        ("What role do carbohydrates play in recovery?",
        "Carbohydrates replenish muscle glycogen stores depleted during exercise. Important for energy restoration and muscle protein synthesis. Include both simple and complex carbs.",
        "recovery", None, "intermediate", "nutrition", "post-workout"),
        
        ("Are supplements necessary for recovery?",
        "Most nutrients should come from whole foods. Protein powder, creatine, and vitamin D may be beneficial. Consult healthcare provider before starting any supplement regimen.",
        "recovery", None, "intermediate", "supplements", "ongoing"),
        
        ("How does alcohol affect exercise recovery?",
        "Alcohol impairs protein synthesis, disrupts sleep quality, causes dehydration, and interferes with muscle repair. Limit alcohol consumption, especially after intense training sessions.",
        "recovery", None, "intermediate", None, "ongoing"),
        
        # ========== SLEEP AND STRESS MANAGEMENT ==========
        ("How does sleep quality affect recovery?",
        "Sleep is when most muscle repair and growth occurs. Poor sleep increases cortisol, decreases growth hormone, impairs immune function, and slows recovery. Prioritize 7-9 hours nightly.",
        "recovery", None, "beginner", None, "daily"),
        
        ("What can I do to improve sleep quality?",
        "Maintain consistent sleep schedule, create cool dark environment, avoid screens before bed, limit caffeine after 2pm, and establish relaxing bedtime routine.",
        "recovery", None, "beginner", None, "daily"),
        
        ("How does stress impact exercise recovery?",
        "Chronic stress elevates cortisol, impairs immune function, disrupts sleep, and slows muscle repair. Manage stress through relaxation techniques, adequate rest, and lifestyle modifications.",
        "recovery", None, "intermediate", None, "ongoing"),
        
        ("What are effective stress management techniques?",
        "Deep breathing exercises, meditation, yoga, progressive muscle relaxation, regular exercise, adequate sleep, and social support. Find techniques that work best for you.",
        "recovery", None, "beginner", "bodyweight", "10-20min"),
        
        # ========== RECOVERY MODALITIES ==========
        ("Do ice baths help with recovery?",
        "Ice baths may reduce inflammation and muscle soreness, but can also impair adaptation to training. Use strategically for competition recovery, not after every workout.",
        "recovery", None, "advanced", "ice", "10-15min"),
        
        ("What's the best way to use heat therapy?",
        "Heat therapy increases blood flow, reduces muscle stiffness, and promotes relaxation. Use before activity for mobility or after for relaxation. Avoid on acute injuries.",
        "recovery", None, "beginner", "heat", "15-20min"),
        
        ("Are compression garments worth using?",
        "Compression garments may improve circulation and reduce swelling. Benefits are modest but may help with perceived recovery. Most beneficial during travel or extended sitting.",
        "recovery", None, "intermediate", "compression", "as_needed"),
        
        ("How effective are massage guns for recovery?",
        "Massage guns can help reduce muscle tension and improve circulation. Use for 30-60 seconds per muscle group. Don't replace proper warm-up, cool-down, or professional treatment.",
        "recovery", None, "intermediate", "massage_gun", "5-10min"),
        
        ("What's the role of sauna in recovery?",
        "Sauna use may improve circulation, reduce muscle soreness, and promote relaxation. Use 15-20 minutes post-workout. Stay hydrated and listen to your body's response.",
        "recovery", None, "intermediate", "sauna", "15-20min"),
        
        ("How do I create a recovery routine?",
        "Include elements like proper cool-down, hydration, nutrition timing, stretching or foam rolling, stress management, and adequate sleep. Consistency is more important than perfection.",
        "recovery", None, "intermediate", "various", "daily"),
        
        # ========== INJURY PREVENTION THROUGH RECOVERY ==========
        ("How does proper recovery prevent injuries?",
        "Adequate recovery allows tissues to repair, reduces accumulated fatigue, maintains movement quality, and keeps immune system strong. Poor recovery increases injury risk significantly.",
        "recovery", None, "intermediate", None, "ongoing"),
        
        ("What are early warning signs of overuse injuries?",
        "Persistent soreness, stiffness, decreased performance, minor aches that worsen with activity, sleep disturbances, and mood changes. Address early to prevent serious injury.",
        "recovery", None, "intermediate", None, "ongoing"),
        
        ("How do I balance training intensity with recovery needs?",
        "Follow 80/20 rule - 80% of training at moderate intensity, 20% high intensity. Include planned recovery days, periodize training loads, and adjust based on recovery markers.",
        "recovery", None, "advanced", None, "ongoing"),
        
        ("What's the importance of mental recovery?",
        "Mental fatigue affects physical performance and decision-making. Include activities you enjoy, social connections, hobbies outside fitness, and stress management techniques.",
        "recovery", None, "intermediate", None, "ongoing"),
    ]
    
    # Add all FAQs to matcher
    all_faqs = exercise_faqs + injury_faqs + equipment_faqs + therapy_faqs + recovery_faqs
    
    for question, answer, category, body_part, difficulty, equipment, duration in all_faqs:
        matcher.add_faq(question, answer, category, body_part, difficulty, equipment, duration)
    
    return matcher

# Performance monitoring for Raspberry Pi
class WorkoutFAQAnalytics:
    """Track FAQ usage and performance for optimization"""
    
    def __init__(self):
        self.query_log = []
        self.response_times = []
        self.popular_categories = {}
        self.no_match_queries = []
    
    def log_query(self, query: str, matches: List, response_time: float):
        """Log query for analytics"""
        self.query_log.append({
            'timestamp': datetime.now(),
            'query': query,
            'num_matches': len(matches),
            'best_score': matches[0][1] if matches else 0,
            'response_time': response_time
        })
        
        self.response_times.append(response_time)
        
        if matches:
            category = matches[0][2]['category']
            self.popular_categories[category] = self.popular_categories.get(category, 0) + 1
        else:
            self.no_match_queries.append(query)
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.response_times:
            return "No queries logged yet"
        
        avg_response_time = np.mean(self.response_times)
        match_rate = (len(self.query_log) - len(self.no_match_queries)) / len(self.query_log) * 100
        
        return {
            'average_response_time': f"{avg_response_time:.3f}s",
            'match_rate': f"{match_rate:.1f}%",
            'total_queries': len(self.query_log),
            'popular_categories': self.popular_categories,
            'queries_needing_improvement': self.no_match_queries[-5:]  # Last 5 unmatched
        }

# Example usage and testing
if __name__ == "__main__":
    import time
    
    # Setup workout/therapy matcher
    print("Setting up Workout & Therapy FAQ Matcher...")
    matcher = setup_workout_therapy_faqs()
    analytics = WorkoutFAQAnalytics()
    
    # Test queries focused on fitness and therapy
    test_queries = [
    "How do I perform a deadlift?",
    "Is a bicep curl good for arms?",
    "Exercises for stronger shoulders?",
    "What muscles do squats work?",
    "Leg day ideas?",
    "I want to bulk up — any exercises?",
    "Tell me an exercise that works multiple muscles.",
    "I want to work my hamstrings.",
    "Show me how to do a squat.",
    "What should I do if I want bigger arms?",
    "Explain the hammer curl technique.",
    "Can you explain lunges?",
    "Any workout for my thighs?",
    "Give me something to strengthen my arms.",
    "How can I grow my biceps?",
    "Do lunges help with glutes?",
    "What’s a hammer curl?",
    "What should I do for glutes?",
    "Teach me an exercise for balance.",
    "What’s better: curls or hammer curls?",
    "I want to work my core.",
    "What movement helps with core strength?",
    "What is the form for a bicep curl?",
    "What’s a good leg workout?",
    "What muscles do bicep curls work?"
]

    
    print("\nTesting Workout & Therapy FAQ Matching:")
    print("=" * 60)
    
    for query in test_queries:
        start_time = time.time()
        matches = matcher.find_best_match(query, threshold=0.3, top_k=2)
        response_time = time.time() - start_time
        
        analytics.log_query(query, matches, response_time)
        
        print(f"\nQuery: '{query}'")
        print(f"Response time: {response_time:.3f}s")
        
        if matches:
            for i, (faq_id, score, faq_data) in enumerate(matches):
                print(f"  Match {i+1} (Score: {score:.3f}):")
                print(f"    Category: {faq_data['category']} | Body Part: {faq_data['body_part']} | Difficulty: {faq_data['difficulty']}")
                print(f"    Q: {faq_data['question']}")
                print(f"    A: {faq_data['answer'][:]}\n")
                if faq_data['equipment']:
                    print(f"    Equipment: {faq_data['equipment']}")
                if faq_data['duration']:
                    print(f"    Duration: {faq_data['duration']}")
        else:
            print("  No suitable matches found")
    
    # Display analytics
    print(f"\n{'='*60}")
    print("PERFORMANCE ANALYTICS:")
    stats = analytics.get_performance_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"\nFAQ Database Summary:")
    category_summary = matcher.get_category_summary()
    for category, count in category_summary.items():
        print(f"  {category}: {count} FAQs")
    
    print(f"\nTotal FAQs in database: {len(matcher.faqs)}")
    print("\nNote: This example uses dummy embeddings.")
    print("For production, download GloVe embeddings from:")
