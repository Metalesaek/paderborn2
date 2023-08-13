'''
def startup():
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
	




startup()

'''