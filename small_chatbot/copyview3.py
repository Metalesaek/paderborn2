from django.shortcuts import render
from django.templatetags.static import static
from django.conf import settings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import threading
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#llm = ChatOpenAI(model_name="gpt-3.5-turbo")
#from .llm_model import qachat




from openai import ChatCompletion

import openai
import requests 


def once_call(request):

	llm = ChatOpenAI(model_name="gpt-4-0613", openai_api_key="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")
	embeddings = OpenAIEmbeddings()
	url = static('chroma2') 
	docsearch = Chroma(
	   persist_directory= url, embedding_function=embeddings
	  )

	memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
	doc=45
	retriever = docsearch.as_retriever()
	# Create the multipurpose chain
	qachat = ConversationalRetrievalChain.from_llm(
	            llm=llm,
	            condense_question_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo'),
	            memory=memory,
	            retriever=retriever, 
	            return_source_documents=True
	        )
	return render(request, 'small_chatbot/once_call.html', {'qachat': qachat})


qachat = None

def index(request):
	global qachat	
	if not qachat:
		llm = ChatOpenAI(model_name="gpt-4-0613", openai_api_key="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")
		embeddings = OpenAIEmbeddings()
		url = static('chroma2') 
		docsearch = Chroma(
		   persist_directory= url, embedding_function=embeddings
		  )

		memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

		retriever = docsearch.as_retriever()
		# Create the multipurpose chain
		qachat = ConversationalRetrievalChain.from_llm(
		            llm=llm,
		            condense_question_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo'),
		            memory=memory,
		            retriever=retriever, 
		            return_source_documents=True
		        )
		conversation = request.session.get('conversation', [])
		#docsearch = Chroma(
	    #persist_directory= url, embedding_function=embeddings
	    #)	

		if request.method == 'POST':
			user_input = request.POST.get('user_input')
		
			# Define your chatbot's predefined prompts
			

			# Append user input to the conversation
			

			# Append conversation messages to prompts
			

			# Set up and invoke the ChatGPT model
			
			# Create the multipurpose chain
			#qachat = settings.qachat
			if user_input:
				conversation.append({"role": "user", "content": user_input})

					# Extract chatbot replies from the response

			reply = qachat(user_input)

			# Append chatbot replies to the conversation
			conversation.append({"role": "assistant", "content": reply["answer"]})

			# Update the conversation in the session
			request.session['conversation'] = conversation

			


			return render(request, 'small_chatbot/index.html', {'user_input': user_input, 'conversation': conversation})
		else:
			request.session.clear()
			return render(request, 'small_chatbot/index.html', {'conversation': conversation})
		
	else:
		conversation = request.session.get('conversation', [])
		#docsearch = Chroma(
	    #persist_directory= url, embedding_function=embeddings
	    #)	

		if request.method == 'POST':
			user_input = request.POST.get('user_input')
		
			# Define your chatbot's predefined prompts
			

			# Append user input to the conversation
			

			# Append conversation messages to prompts
			

			# Set up and invoke the ChatGPT model
			
			# Create the multipurpose chain
			#qachat = settings.qachat
			if user_input:
				conversation.append({"role": "user", "content": user_input})

					# Extract chatbot replies from the response

			reply = qachat(user_input)

			# Append chatbot replies to the conversation
			conversation.append({"role": "assistant", "content": reply["answer"]})

			# Update the conversation in the session
			request.session['conversation'] = conversation

			


			return render(request, 'small_chatbot/index.html', {'user_input': user_input, 'conversation': conversation})
		else:
			request.session.clear()
			return render(request, 'small_chatbot/index.html', {'conversation': conversation})

		

def run_my_function():
    """
    This function starts a thread that runs the `my_function` function once.
    """
    thread = threading.Thread(target=once_call)
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    run_my_function()

