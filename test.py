from swisscom_ai.research_keyphrase.preprocessing.postagging import PosTaggingCoreNLP

# Initialize POS tagger
host = 'localhost'
port = 9000
pos_tagger = PosTaggingCoreNLP(host, port)

# Test tagging
try:
    test_response = pos_tagger.pos_tag_raw_text("This is a test.")
    print("Tagging response:", test_response)
except Exception as e:
    print("Failed to tag test sentence:", e)
