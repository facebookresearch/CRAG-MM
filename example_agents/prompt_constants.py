from pydantic import BaseModel

SYSTEM_PROMPT = """You are a helpful assistant that truthfully answers the user \
question given an image.
Please follow these guidelines when formulating your response:
1. Your response must be grounded in the image and based on factual information.
2. Keep your response concise and to the point. Strive to answer in one sentence.
3. If you are uncertain or don’t know the answer, respond with “I don’t know”.
"""

SYSTEM_MULTI_PROMPT = """You are a helpful assistant that truthfully answers the user \
question given an image. You are able to engage in multi-turn conversations. Answer \
the new question based on the given image and information extracted from the previous \
conversations.
Please follow these guidelines when formulating your response:
1. Your response must be grounded in the image and based on factual information.
2. Build upon previous conversations when responding.
3. Keep your response concise and to the point. Strive to answer in one sentence. 
4. If you are uncertain or don’t know the answer, respond with “I don’t know”.
"""

# Key Message Repeats
KEY_RAG_PROMPT = (
    "Refer to the additional information above when answering the following question if"
    " you are confident that it includes relevant information. "
)
KEY_MESSAGE_HISTORY_PROMPT = "Keep the context of the conversation history in mind. "
KEY_USER_PROMPT = (
    "Keep your response concise and to the point. Strive to answer in one sentence. If"
    " you are uncertain or unable to answer the question, respond with “I don’t"
    " know”.\n"
)


# Query Re-write
class QueryAnalysis(BaseModel):
    entity_name: str
    updated_question: str


QUERY_REWRITE_SYSTEM = """You are a helpful assistant that analyzes questions about \
images and refines them to be more specific."""


QUERY_REWRITE_STEP_1 = """Given an image and the question: "{question}", follow these \
steps carefully:

1. Analyze the question and identify the main entity (name and type) the question is \
asking about. Do not try to answer the question.
2. Look for any corresponding specific entity name visible in the image that matches \
the generic entity in the question.
4. If you cannot confidently identify the specific entity name in the image, be as \
specific as you can when describing the entity.

###Guidelines:
Follow these steps carefully:

**Step 1: Analyze the Query**
- Read the question carefully
- Look for references within the question that are less specific, for example: "this", \
"that vehicle"
- Identify the main entity the question is asking about
- Determine:
  - What type of entity it is (e.g., person, animal, object, location, brand, etc.)
  - What generic noun is being used (e.g., "the man", "the car", "the building", "the \
animal")
  - What specific information the question seeks about this entity

**Step 2: Examine the Image**
- Look at the image provided
- Try to identify the specific name or more precise identifier for the generic entity \
mentioned in the question
- The entity name should be specific, for example: a brand, breed, species, or a \
specific location. Avoid using generic descriptive phrases such as "white car" or \
"gray floor"
- Consider:
  - Visible text, labels, or signs in the image
  - Distinctive features that could identify a specific person, brand, location, or \
object
  - Any contextual clues that might reveal the specific identity
"""

QUERY_REWRITE_PROMPT = """Given the above analysis, extract the name of the main \
entity in the image, match it to a generic noun in this question: "{question}" and \
replace the generic noun in the question with the specific entity name. If you cannot \
confidently identify the specific entity name in the image, return the original \
question unchanged. Format your response as a valid JSON object. Only include the JSON \
object in your response, no extra text.
The JSON response should be formatted as follows:
{{
  "entity_name": "...",
  "updated_question": "..."
}}

###Guidelines:
- Replace generic descriptions of nouns, objects, location, entities with more \
specific entity names extracted from the image.
- Look for references within the question that are less specific, for example: "this", \
"that vehicle".
- If you can confidently identify the specific entity name from the image:
  - Replace the generic noun in the question with the specific name
  - Ensure the grammar remains correct after substitution
- If you cannot confidently determine the specific entity name:
  - Return the original question unchanged
  - Do not guess or make assumptions

**Examples:**
- Original: "What is the man wearing?"
  → If you can identify the man as "Tom Hanks" from the image: "What is Tom Hanks \
wearing?"
  → If you cannot identify him: "What is the man wearing?"

- Original: "What color is the car?"
  → If you can see it's a Tesla Model 3: "What color is the Tesla Model 3?"
  → If you can't determine the specific model: "What color is the car?"

The entity name should contain the main entity identified in the image. \
An entity name should be specific, referring to a brand, breed, species, or a specific \
location. Avoid using generic descriptive phrases such as "white car" or "gray floor". \
If you are not confident about what is in the image, leave the entity_name field \
blank. The updated_question field should contain the updated question with generic \
nouns updated as specific entity names. If you are not confident about identifying the \
specific entity name, use the original question without updates.
"""
