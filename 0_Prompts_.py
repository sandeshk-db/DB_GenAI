# Databricks notebook source
# MAGIC %md
# MAGIC From > https://github.com/NirDiamant/Prompt_Engineering

# COMMAND ----------

# MAGIC %pip install --upgrade langchain mlflow[databricks] langchain[docarray] databricks-sql-connector[sqlalchemy] langchain-community langchain-core langgraph docarray
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain_community.embeddings import DatabricksEmbeddings
from langchain_community.chat_models import ChatDatabricks
import os

from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Set the Databricks token from the secret scope
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("sandesk4", "pat_token")

# Initialize the embedding model with the specified endpoint
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

# Initialize the chat model with the specified endpoint and max tokens
chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", max_tokens = 500)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prompt Basics

# COMMAND ----------

## Basic prompt
basic_prompt = "What is UC in Databricks?"
print(chat_model.invoke(basic_prompt).content)

# COMMAND ----------

## More structured prompt
structured_prompt = PromptTemplate(
    input_variables=["input"],
    template="Provide a definition of {input}, explain its importance, and list three key benefits."
)

chain = structured_prompt | chat_model # Combine the prompt template with the language model
input_variables = {"input": "UC in Databricks"} # Define the input variables
output = chain.invoke(input_variables).content # Invoke the chain with the input variables
print(output)

# COMMAND ----------

### factual check prompts

fact_check_prompt = PromptTemplate(
    input_variables=["statement"],
    template="""Evaluate the following statement for factual accuracy. If it's incorrect, provide the correct information:
    Statement: {statement}
    Evaluation:"""
)

chain = fact_check_prompt | chat_model
print(chain.invoke("The capital of France is Italy.").content)

# COMMAND ----------

## solving a complex problem

problem_solving_prompt = PromptTemplate(
    input_variables=["problem"],
    template="""Solve the following problem step by step:
    Problem: {problem}
    Solution:
    1)"""
)

chain = problem_solving_prompt | chat_model

print(chain.invoke("Calculate the compound interest on $1000 invested for 5 years at an annual rate of 5%, compounded annually.").content)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Types of conversations 

# COMMAND ----------

# Single-turn prompts
prompts = [
    "What is the capital of France?",
    "What is its population?",
    "What is the city's most famous landmark?"
]

print("Single-turn responses:")
for prompt in prompts:
    print(f"Q: {prompt}")
    print(f"A: {chat_model.invoke(prompt).content}\n")


# Multi-turn prompts
print("Multi-turn responses:")
conversation = ConversationChain(llm=chat_model, memory=ConversationBufferMemory())
for prompt in prompts:
    print(f"Q: {prompt}")
    print(f"A: {conversation.predict(input=prompt)}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Zero Shot Prompting

# COMMAND ----------

def create_chain(prompt_template):
    """
    Create a LangChain chain with the given prompt template.
    
    Args:
        prompt_template (str): The prompt template string.
    
    Returns:
        LLMChain: A LangChain chain object.
    """
    prompt = PromptTemplate.from_template(prompt_template)
    return prompt | chat_model
  

direct_task_prompt = """Classify the sentiment of the following text as positive, negative, or neutral.
Do not explain your reasoning, just provide the classification.

Text: {text}

Sentiment:"""

direct_task_chain = create_chain(direct_task_prompt)

# Test the direct task specification
texts = [
    "I absolutely loved the movie! The acting was superb.",
    "The weather today is quite typical for this time of year.",
    "I'm disappointed with the service I received at the restaurant."
]

for text in texts:
    result = direct_task_chain.invoke({"text": text}).content
    print(f"Text: {text}")
    print(f"Sentiment: {result}")

# COMMAND ----------

## Configure how the output should look like !! 

format_spec_prompt = """Generate a short news article about {topic}. 
Structure your response in the following format:

Headline: [A catchy headline for the article]

Lead: [A brief introductory paragraph summarizing the key points]

Body: [2-3 short paragraphs providing more details]

Conclusion: [A concluding sentence or call to action]"""

format_spec_chain = create_chain(format_spec_prompt)

# Test the format specification prompting
topic = "Harappan civilisation!"
result = format_spec_chain.invoke({"topic": topic}).content
print(result)

# COMMAND ----------

## comparative analysis

def compare_prompts(task, prompt_templates):
    """
    Compare different prompt templates for the same task.
    
    Args:
        task (str): The task description or input.
        prompt_templates (dict): A dictionary of prompt templates with their names as keys.
    """
    print(f"Task: {task}\n")
    for name, template in prompt_templates.items():
        chain = create_chain(template)
        result = chain.invoke({"task": task}).content
        print(f"{name} Prompt Result:")
        print(result)
        print("\n" + "-"*50 + "\n")

task = "Explain conciesly the concept of blockchain technology"

prompt_templates = {
    "Basic": "Explain {task}.",
    "Structured": """Explain {task} by addressing the following points:
1. Definition
2. Key features
3. Real-world applications
4. Potential impact on industries"""
}

compare_prompts(task, prompt_templates)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Few-Shot Learning and In-Context Learning

# COMMAND ----------

##Basic fewshots -- One task of classification
def few_shot_sentiment_classification(input_text):

    few_shot_prompt = PromptTemplate(
        input_variables=["input_text"],
        template="""
        Classify the sentiment as Positive, Negative, or Neutral.
        
        Examples:
        Text: I love this product! It's amazing.
        Sentiment: Positive
        
        Text: This movie was terrible. I hated it.
        Sentiment: Negative
        
        Text: The weather today is okay.
        Sentiment: Neutral
        
        Now, classify the following:
        Text: {input_text}
        Sentiment:
        """
    )
    
    chain = few_shot_prompt | chat_model
    result = chain.invoke(input_text).content

    # Clean up the result
    result = result.strip()

    # Extract only the sentiment label
    if ':' in result:
        result = result.split(':')[1].strip()
    
    return result  # This will now return just "Positive", "Negative", or "Neutral"
  
  
test_text = "I can't believe how great this new restaurant is!"
result = few_shot_sentiment_classification(test_text)
print(f"Input: {test_text}")
print(f"Predicted Sentiment: {result}")

# COMMAND ----------

### Advanced few shots - Multiple tasks per prompt

def multi_task_few_shot(input_text, task):
    few_shot_prompt = PromptTemplate(
        input_variables=["input_text", "task"],
        template="""
        Perform the specified task on the given text.
        
        Examples:
        Text: I love this product! It's amazing.
        Task: sentiment
        Result: Positive
        
        Text: Bonjour, comment allez-vous?
        Task: language
        Result: French
        
        Now, perform the following task:
        Text: {input_text}
        Task: {task}
        Result:
        """
    )
    
    chain = few_shot_prompt | chat_model
    return chain.invoke({"input_text": input_text, "task": task}).content

print(multi_task_few_shot("I can't believe how great this is!", "sentiment"))
print(multi_task_few_shot("Guten Tag, wie geht es Ihnen?", "language"))

# COMMAND ----------

### In context learning -- Converting text to pig latin

def in_context_learning(task_description, examples, input_text):
    example_text = "".join([f"Input: {e['input']}\nOutput: {e['output']}\n\n" for e in examples])
    
    in_context_prompt = PromptTemplate(
        input_variables=["task_description", "examples", "input_text"],
        template="""
        Task: {task_description}
        
        Examples:
        {examples}
        
        Now, perform the task on the following input:
        Input: {input_text}
        Output:
        """
    )
    
    chain = in_context_prompt | chat_model
    return chain.invoke({"task_description": task_description, "examples": example_text, "input_text": input_text}).content

task_desc = "Convert the given text to pig latin."
examples = [
    {"input": "hello", "output": "ellohay"},
    {"input": "apple", "output": "appleay"}
]
test_input = "python"

result = in_context_learning(task_desc, examples, test_input)
print(f"Input: {test_input}")
print(f"Output: {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC No major differences between few-shots vs In context learning. Both involves concept of providing few examples in the prompt so that LM understands what to do 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation

# COMMAND ----------

def evaluate_model(model_func, test_cases):
    '''
    Evaluate the model on a set of test cases.

    Args:
    model_func: The function that makes predictions.
    test_cases: A list of dictionaries, where each dictionary contains an "input" text and a "label" for the input.

    Returns:
    The accuracy of the model on the test cases. 
    '''
    correct = 0
    total = len(test_cases)
    
    for case in test_cases:
        input_text = case['input']
        true_label = case['label']
        prediction = model_func(input_text).strip()
        
        is_correct = prediction.lower() == true_label.lower()
        correct += int(is_correct)
        
        print(f"Input: {input_text}")
        print(f"Predicted: {prediction}")
        print(f"Actual: {true_label}")
        print(f"Correct: {is_correct}\n")
    
    accuracy = correct / total
    return accuracy

test_cases = [
    {"input": "This product exceeded my expectations!", "label": "Positive"},
    {"input": "I'm utterly disappointed with the service.", "label": "Negative"},
    {"input": "The temperature today is 72 degrees.", "label": "Neutral"}
]

accuracy = evaluate_model(few_shot_sentiment_classification, test_cases)
print(f"Model Accuracy: {accuracy:.2f}")
