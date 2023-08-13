from django.shortcuts import render, redirect
from openai import ChatCompletion
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import openai
import requests 
from .forms import *

API_URL = "http://localhost:3000/api/v1/prediction/afc66ba4-630f-40e5-9557-8804599b5cb3"


'''	

def once_call(request):
	return render(request, 'small_chatbot/once_call.html', {})

'''

def query(payload):
    response = requests.post(API_URL, json=payload)
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
			messages=prompts1,
			api_key="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"
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
			api_key="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"
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
