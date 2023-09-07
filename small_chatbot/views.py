from django.shortcuts import render, redirect
from openai import ChatCompletion
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import load_tools, initialize_agent, AgentType
import openai
import requests 
from .forms import *
from django.http import StreamingHttpResponse
from django.conf import settings

# general2
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.utilities.arxiv import ArxivAPIWrapper
from bardapi import Bard


'''	

def once_call(request):
	return render(request, 'small_chatbot/once_call.html', {})

'''
headers = {"Authorization": settings.API_flowise}
def query(payload):
	response = requests.post(settings.API_URL, headers=headers, json=payload)
	return response.json()

def index(request):
	conversation = request.session.get('conversation', [])

	if request.method == 'POST':
		user_input = request.POST.get('user_input')

		# Define your chatbot's predefined prompts
		prompts = []

		# Append user input to the conversation
		if user_input:
			conversation.append({"role": "user", "content": user_input})
			output = query({
				"question": user_input,
				
			})
			conversation.append({"role": "assistant", "content": output["text"]})
		# Append conversation messages to prompts
		prompts.extend(conversation)

		# Set up and invoke the ChatGPT model
		
		
		# Extract chatbot replies from the response

		#chatbot_replies = [message['message']['content'] for message in response['choices'] if message['message']['role'] == 'assistant']

		# Append chatbot replies to the conversation
		
		
		
		#for reply in chatbot_replies:
		#	conversation.append({"role": "assistant", "content": reply})

		# Update the conversation in the session
		request.session['conversation'] = conversation
		

		return render(request, 'small_chatbot/index.html', {'user_input': user_input, 'conversation': conversation})
	else:
		request.session.clear()
		return render(request, 'small_chatbot/index.html', {'conversation': conversation})




def general(request):
	conversation1 = request.session.get('conversation1', [])

	if request.method == 'POST':
		user_input = request.POST.get('user_input')

		# Define your chatbot's predefined prompts
		prompts1 = []

		# Append user input to the conversation
		if user_input:
			conversation1.append({"role": "user", "content": user_input})

		# Append conversation messages to prompts
		prompts1.extend(conversation1)

		# Set up and invoke the ChatGPT model
		response = openai.ChatCompletion.create(
			model="gpt-3.5-turbo",
			messages = prompts1,
			api_key = settings.API_KEY
		)
		
		# Extract chatbot replies from the response

		chatbot_replies = [message['message']['content'] for message in response['choices'] if message['message']['role'] == 'assistant']

		# Append chatbot replies to the conversation
		for reply in chatbot_replies:
			conversation1.append({"role": "assistant", "content": reply})
		reply1=chatbot_replies[-1]
		# Update the conversation in the session
		request.session['conversation1'] = conversation1

		return render(request, 'small_chatbot/general.html', {'user_input': user_input, 'reply1':reply1,
			'chatbot_replies': chatbot_replies, 'conversation1': conversation1})
	else:
		request.session.clear()
		return render(request, 'small_chatbot/general.html', {'conversation1': conversation1})


def bard(request):
	
	conversation4 = request.session.get('conversation4', [])

	if request.method == 'POST':
		user_input = request.POST.get('user_input')

		# Define your chatbot's predefined prompts
		prompts4 = []

		# Append user input to the conversation
		if user_input:
			conversation4.append({"role": "user", "content": user_input})

		# Append conversation messages to prompts
		prompts4.extend(conversation4)

		# Set up and invoke the ChatGPT model
		response = Bard(settings.API_BARD).get_answer(user_input)
		
		# Extract chatbot replies from the response

		chatbot_replies = [message['message']['content'] for message in response['content'] if message['message']['role'] == 'assistant']

		# Append chatbot replies to the conversation
		for reply in chatbot_replies:
			conversation4.append({"role": "assistant", "content": reply})
		reply1=chatbot_replies[-1]
		# Update the conversation in the session
		request.session['conversation4'] = conversation4

		return render(request, 'small_chatbot/bard.html', {'user_input': user_input, 'reply1':reply1,
			'chatbot_replies': chatbot_replies, 'conversation4': conversation4})
	else:
		request.session.clear()
		return render(request, 'small_chatbot/bard.html', {'conversation4': conversation4})




"""

def arxiv(request):
	conversation2 = request.session.get('conversation2', [])

	if request.method == 'POST':
		user_input = request.POST.get('user_input')

		# Define your chatbot's predefined prompts
		prompts2 = []

		# Append user input to the conversation
		if user_input:
			conversation2.append({"role": "user", "content": user_input})
			llm = ChatOpenAI(temperature=0.0, api_key = settings.API_KEY )
			tools = load_tools(["arxiv"],)
			agent_chain = initialize_agent(
 								   tools,
    								llm,
    								agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    								verbose=True,
									)
			response=agent_chain.run(user_input)
			conversation2.append({"role": "assistant", "content": response})

		# Append conversation messages to prompts
		prompts2.extend(conversation2)

		
		request.session['conversation2'] = conversation2

		return render(request, 'small_chatbot/arxiv.html', {'user_input': user_input, 'reply1':reply1,
			'chatbot_replies': chatbot_replies, 'conversation2': conversation2})
	else:
		request.session.clear()
		return render(request, 'small_chatbot/arxiv.html', {'conversation2': conversation2})
"""
def arxiv(request):
	conversation2 = request.session.get('conversation2', [])

	if request.method == 'POST':
		user_input = request.POST.get('user_input')

		# Define your chatbot's predefined prompts
		prompts2 = []

		# Append user input to the conversation
		if user_input:
			conversation2.append({"role": "user", "content": user_input})
			
			arxiv = ArxivAPIWrapper(
				    top_k_results = 5,
				    ARXIV_MAX_QUERY_LENGTH = 300,
				    load_max_docs = 5,
				    load_all_available_meta = False,
				    doc_content_chars_max = 40000
				)
			response=arxiv.run(user_input)
			conversation2.append({"role": "assistant", "content": response})

		# Append conversation messages to prompts
		
		return render(request, 'small_chatbot/arxiv1.html', {'user_input': user_input,
			 'conversation2': conversation2})
	else:
		request.session.clear()
		return render(request, 'small_chatbot/arxiv1.html', {'conversation2': conversation2})


def function_call(request):
	
	form = ChatForm(request.POST or None) 
	if request.method == 'POST':
		topic = str(request.POST.get('topic'))
		num_source = str(request.POST.get('num_source'))

	

		# Append user input to the conversation
		if not num_source:
			num_source=3
		messages=[{"role": "user", "content": topic}]
		functions=[
		{
			"name": "get_source_papers",
			"description": "get the"+num_source+ "sources that talk about"+topic,
			"parameters": {
				"type": "object",
				"properties": {
					"summary": {
						"type": "string",
						"description": "act as en expert: what is"+topic
					},
					
  
				 "papers": {
						"type": "array",
						"items": {
							"type": "string",
							"description": "source that talk about"+topic 
						},
						"description": "list"+num_source+ "sources (with complete elements, Author, date, journal) that talk about"+topic
					}
			},
					"required": ["summary", "papers"]
			}
		}
		
		]
		

		# Set up and invoke the ChatGPT model
		completion = openai.ChatCompletion.create(
			model="gpt-4-0613",
			messages=messages,
			functions=functions,
			function_call={"name": "get_source_papers"},
			api_key= settings.API_KEY
		)
		
		# Extract chatbot replies from the response

		#hatbot_replies = [message['message']['content'] for message in response['choices'] if message['message']['role'] == 'assistant']

		# Append chatbot replies to the conversation
		reply_content = completion.choices[0].message
		out = reply_content["function_call"]["arguments"]
		authors = out.replace("\n", "")
		authors_list=eval(authors)
		papers=authors_list["papers"]


		return render(request, 'small_chatbot/function_call.html', {"papers":papers, "topic":topic, "form":form})
	else:
		
		return render(request, 'small_chatbot/chatform.html', {"form":form})

def general2(request):
	documents = []
	for file in os.listdir("docs"):
		if file.endswith(".pdf"):
			pdf_path = "./docs/" + file
			loader = PyPDFLoader(pdf_path)
			documents.extend(loader.load())
		elif file.endswith('.docx') or file.endswith('.doc'):
			doc_path = "./docs/" + file
			loader = Docx2txtLoader(doc_path)
			documents.extend(loader.load())
		elif file.endswith('.txt'):
			text_path = "./docs/" + file
			loader = TextLoader(text_path)
			documents.extend(loader.load())

	text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
	documents = text_splitter.split_documents(documents)

	vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
	vectordb.persist()

	pdf_qa = ConversationalRetrievalChain.from_llm(
		ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
		vectordb.as_retriever(search_kwargs={'k': 6}),
		return_source_documents=True,
		verbose=False
	)

	yellow = "\033[0;33m"
	green = "\033[0;32m"
	white = "\033[0;39m"

	chat_history = []
	print(f"{yellow}---------------------------------------------------------------------------------")
	print('Welcome to the DocBot. You are now ready to start interacting with your documents')
	print('---------------------------------------------------------------------------------')
	while True:
		query = input(f"{green}Prompt: ")
		if query == "exit" or query == "quit" or query == "q" or query == "f":
			print('Exiting')
			sys.exit()
		if query == '':
			continue
		result = pdf_qa(
			{"question": query, "chat_history": chat_history})
		print(f"{white}Answer: " + result["answer"])
		chat_history.append((query, result["answer"]))