from django.shortcuts import render
from openai import ChatCompletion
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import openai
import requests 

def index(request):
	conversation = request.session.get('conversation', [])

	if request.method == 'POST':
		user_input = request.POST.get('user_input')

		# Define your chatbot's predefined prompts
		prompts = []

		# Append user input to the conversation
		if user_input:
			conversation.append({"role": "user", "content": user_input})

		# Append conversation messages to prompts
		prompts.extend(conversation)

		# Set up and invoke the ChatGPT model
		response = openai.ChatCompletion.create(
			model="gpt-3.5-turbo",
			messages=prompts,
			api_key="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"
		)
		
		# Extract chatbot replies from the response

		chatbot_replies = [message['message']['content'] for message in response['choices'] if message['message']['role'] == 'assistant']

		# Append chatbot replies to the conversation
		for reply in chatbot_replies:
			conversation.append({"role": "assistant", "content": reply})

		# Update the conversation in the session
		request.session['conversation'] = conversation

		return render(request, 'small_chatbot/index.html', {'user_input': user_input, 'chatbot_replies': chatbot_replies, 'conversation': conversation})
	else:
		request.session.clear()
		return render(request, 'small_chatbot/index.html', {'conversation': conversation})
