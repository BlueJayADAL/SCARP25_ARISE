from ConversationalAI.exercise_similarity_parse import WorkoutTherapyFAQMatcher

# Load the GloVe embeddings and your full FAQ JSON
matcher = WorkoutTherapyFAQMatcher(
    glove_path= "models\glove.6B\glove.6B.300d.txt",
    faq_json_path="ConversationalAI/full_FAQ_extended.json"
)

# List of test queries
test_queries = [
    # ✅ Muscle Growth
    "how do I do a bicep curl",
    "how do i do a deadlift",
    "best way to grow my legs",
    "how do I do a squat",
    "best way to recover"
]


# Run the matcher and print results
print("🔍 FAQ Matcher Test Results:")
for query in test_queries:
    print(f"\n🗣️ Query: {query}")
    print("💬 Response:")
    print(matcher.respond_to_query(query))
print(matcher.get_latency_summary())
